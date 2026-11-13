"""The diffusion process: noise schedule, DDIM sampler with classifier-free guidance, and weight EMA.

Training follows DDPM (Ho et al., 2020): add Gaussian noise at a random timestep and train the
network to predict that noise. Sampling uses deterministic DDIM (Song et al., 2021), so a fixed
starting noise gives the same image at every snapshot and 100 steps are enough.
"""

import copy
from collections.abc import Callable

import torch
from torch import nn

from tinydit import constants as C
from tinydit.errors import ConfigError

NoisePredictor = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
PIXEL_MIN, PIXEL_MAX = -1.0, 1.0


class NoiseSchedule:
    """Linear beta schedule and the closed-form forward process q(x_t | x_0)."""

    def __init__(self, timesteps: int = C.TIMESTEPS, beta_start: float = C.BETA_START,
                 beta_end: float = C.BETA_END) -> None:
        if timesteps < 1 or not 0.0 < beta_start < beta_end < 1.0:
            raise ConfigError("need timesteps >= 1 and 0 < beta_start < beta_end < 1")
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        self.alpha_bar = torch.cumprod(1.0 - betas, dim=0).float()

    def alpha_bar_at(self, t: torch.Tensor) -> torch.Tensor:
        """Cumulative signal share at timesteps t, shaped (B, 1, 1, 1) for broadcasting."""
        return self.alpha_bar.to(t.device)[t].view(-1, 1, 1, 1)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) noise."""
        a = self.alpha_bar_at(t)
        return a.sqrt() * x0 + (1.0 - a).sqrt() * noise

    def sample_timesteps(self, batch: int, device: torch.device) -> torch.Tensor:
        """Uniform random timesteps in [0, T)."""
        return torch.randint(0, self.timesteps, (batch,), device=device)


class DDIMSampler:
    """Deterministic DDIM sampling (eta = 0) with classifier-free guidance."""

    def __init__(self, schedule: NoiseSchedule, steps: int = C.SAMPLE_STEPS,
                 guidance: float = C.GUIDANCE_SCALE) -> None:
        if not 1 <= steps <= schedule.timesteps:
            raise ConfigError("steps must be between 1 and the number of timesteps")
        self.schedule = schedule
        self.guidance = guidance
        self.timesteps = torch.linspace(schedule.timesteps - 1, 0, steps).round().long().tolist()

    def guided_noise(self, model: NoisePredictor, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
                     null_class: int) -> torch.Tensor:
        """eps_uncond + w * (eps_cond - eps_uncond), from one batched forward pass."""
        if self.guidance == 1.0:
            return model(x, t, y)
        both = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, torch.full_like(y, null_class)]))
        cond, uncond = both.float().chunk(2)
        return uncond + self.guidance * (cond - uncond)

    @torch.inference_mode()
    def sample(self, model: NoisePredictor, noise: torch.Tensor, y: torch.Tensor, null_class: int,
               keep_every: int = 0) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Denoise from pure noise. Return the images in [-1, 1] and, if asked, the predicted x_0 on the way."""
        x, frames = noise, []
        for i, step in enumerate(self.timesteps):
            t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            eps = self.guided_noise(model, x, t, y, null_class)
            x0 = self._predict_x0(x, t, eps)
            if keep_every and i % keep_every == 0:
                frames.append(x0.cpu())
            x = self._step(x0, eps, self.timesteps[i + 1] if i + 1 < len(self.timesteps) else None, x.device)
        return x.clamp(PIXEL_MIN, PIXEL_MAX), frames

    def _predict_x0(self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a = self.schedule.alpha_bar_at(t)
        return ((x - (1.0 - a).sqrt() * eps) / a.sqrt()).clamp(PIXEL_MIN, PIXEL_MAX)

    def _step(self, x0: torch.Tensor, eps: torch.Tensor, next_step: int | None, device: torch.device) -> torch.Tensor:
        if next_step is None:
            return x0
        a_next = self.schedule.alpha_bar_at(torch.tensor([next_step], device=device))
        return a_next.sqrt() * x0 + (1.0 - a_next).sqrt() * eps


class ExponentialMovingAverage:
    """A slowly updated copy of the weights. Diffusion samples come from this copy."""

    def __init__(self, model: nn.Module, decay: float = C.EMA_DECAY) -> None:
        if not 0.0 <= decay < 1.0:
            raise ConfigError("decay must be in [0, 1)")
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """ema = decay * ema + (1 - decay) * current."""
        for ema_param, param in zip(self.model.parameters(), model.parameters(), strict=True):
            ema_param.lerp_(param.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict:
        """Weights of the averaged model."""
        return self.model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        """Restore the averaged weights."""
        self.model.load_state_dict(state)
