"""A stochastic DDIM sampler (eta = 1) whose steps have a known Gaussian density.

Reinforcement learning needs log p(x_s | x_t) for every step of a rollout. With eta = 1, DDIM
(Song et al., 2021) adds fresh noise at each step, so each step is a Gaussian with mean mu(x_t)
and a fixed standard deviation sigma. The last step (to t = 0) has sigma = 0, so it is
deterministic and gives no log-probability.

Run this sampler in float32: at late steps sigma is about 1e-2, and bf16 rounding of the mean is
as large as the noise, which would make the log-probabilities wrong.
"""

from dataclasses import dataclass

import torch

from tinydit.diffusion import NoiseSchedule
from tinydit.errors import ConfigError
from tinydit.methods.base import PIXEL_MAX, PIXEL_MIN, Predictor


@dataclass
class Rollout:
    """states[i] is x at step i (states[0] is the noise, states[-1] the image)."""

    states: list[torch.Tensor]
    labels: torch.Tensor

    @property
    def images(self) -> torch.Tensor:
        """Final images in [-1, 1]."""
        return self.states[-1].clamp(PIXEL_MIN, PIXEL_MAX)


class StochasticDdim:
    """DDIM with eta = 1 on an evenly spaced subset of the training timesteps."""

    def __init__(self, schedule: NoiseSchedule, steps: int) -> None:
        if not 2 <= steps <= schedule.timesteps:
            raise ConfigError("steps must be between 2 and the number of timesteps")
        self.schedule = schedule
        self.timesteps = torch.linspace(schedule.timesteps - 1, 0, steps).round().long().tolist()
        alpha_bar = schedule.alpha_bar.double()
        self._alpha = [float(alpha_bar[t]) for t in self.timesteps] + [1.0]  # a_bar after the last step is 1

    @property
    def steps(self) -> int:
        """Number of transitions in a rollout."""
        return len(self.timesteps)

    def sigma(self, index: int) -> float:
        """Noise scale of transition `index` (zero for the last one)."""
        a_t, a_s = self._alpha[index], self._alpha[index + 1]
        return ((1 - a_s) / (1 - a_t) * (1 - a_t / a_s)) ** 0.5

    def mean(self, model: Predictor, x: torch.Tensor, index: int, labels: torch.Tensor) -> torch.Tensor:
        """Mean of x_s given x_t for transition `index`; differentiable in the model."""
        a_t, a_s, sigma = self._alpha[index], self._alpha[index + 1], self.sigma(index)
        t = torch.full((len(x),), self.timesteps[index], device=x.device, dtype=torch.long)
        eps = model(x, t, labels).float()
        x0 = (x - (1 - a_t) ** 0.5 * eps) / a_t ** 0.5
        return a_s ** 0.5 * x0 + max(1 - a_s - sigma ** 2, 0.0) ** 0.5 * eps

    def log_prob(self, model: Predictor, x: torch.Tensor, x_next: torch.Tensor, index: int,
                 labels: torch.Tensor) -> torch.Tensor:
        """(B,) Gaussian log-density per pixel (mean over pixels, constants dropped).

        Raises ConfigError for the deterministic last step.
        """
        sigma = self.sigma(index)
        if sigma == 0.0:
            raise ConfigError("the last step is deterministic and has no density")
        return -(x_next - self.mean(model, x, index, labels)).pow(2).flatten(1).mean(dim=1) / (2 * sigma ** 2)

    def trainable_steps(self, every: int) -> list[int]:
        """Every `every`-th transition that has noise."""
        return [i for i in range(0, self.steps, every) if self.sigma(i) > 0.0]

    @torch.no_grad()
    def rollout(self, model: Predictor, noise: torch.Tensor, labels: torch.Tensor,
                generator: torch.Generator | None = None) -> Rollout:
        """Sample from noise and keep every state. A CPU generator makes the rollout repeatable."""
        states = [noise]
        for index in range(self.steps):
            x = self.mean(model, states[-1], index, labels)
            sigma = self.sigma(index)
            if sigma > 0.0:
                fresh = torch.randn(x.shape, generator=generator, dtype=x.dtype).to(x.device)
                x = x + sigma * fresh
            states.append(x)
        return Rollout(states, labels)
