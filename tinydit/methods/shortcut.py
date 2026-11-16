"""Shortcut models: condition on the step size and train big steps to match two small steps
(Frans et al., 2025).

The model s(x_t, t, d) gives the average velocity for one step of size d, from time t to t - d
(t = 1 is pure noise). At d = 0 it is the flow-matching velocity. For d > 0 there is no
closed-form target, so the model teaches itself: one step of size 2d must land where two steps
of size d land. This needs no JVP, only two extra forward passes without gradients.
"""

import math

import torch

from tinydit import constants as C
from tinydit.errors import ConfigError
from tinydit.methods.base import (
    PIXEL_MAX, PIXEL_MIN, GenerativeMethod, MethodKind, Predictor, broadcast, guided, time_grid,
)


class Shortcut(GenerativeMethod):
    """Flow matching on most of the batch, self-consistency on the rest."""

    kind = MethodKind.SHORTCUT
    needs_interval = True
    default_steps = C.SHORTCUT_SAMPLE_STEPS
    loss_label = "Flow-matching + self-consistency MSE (log scale)"

    def __init__(self, guidance: float = C.GUIDANCE_SCALE, bootstrap_share: float = C.SHORTCUT_BOOTSTRAP_SHARE,
                 base_steps: int = C.SHORTCUT_BASE_STEPS) -> None:
        if not 0.0 < bootstrap_share < 1.0:
            raise ConfigError("bootstrap_share must be in (0, 1)")
        if base_steps < 2 or base_steps & (base_steps - 1):
            raise ConfigError("base_steps must be a power of two, at least 2")
        self.guidance = guidance
        self.bootstrap_share = bootstrap_share
        self.base_steps = base_steps

    def loss(self, model: Predictor, x0: torch.Tensor, labels: torch.Tensor, null_class: int) -> torch.Tensor:
        """Mean of the squared errors over both parts of the batch."""
        split = max(1, round(len(x0) * self.bootstrap_share))
        noise = torch.randn_like(x0)
        x_t, t, h, target = self._bootstrap_targets(model, x0[:split], noise[:split], labels[:split])
        x_fm, t_fm, v = self._flow_targets(x0[split:], noise[split:])
        inputs = torch.cat([x_t, x_fm]), torch.cat([t, t_fm]), torch.cat([h, torch.zeros_like(t_fm)])
        prediction = model(inputs[0], inputs[1] * C.TIME_SCALE, labels, inputs[2] * C.TIME_SCALE).float()
        return (prediction - torch.cat([target, v])).pow(2).mean()

    def step_sizes(self, count: int, device: torch.device) -> torch.Tensor:
        """Random d in {1/base, 2/base, ..., 1/2}; the target step is 2d."""
        levels = int(math.log2(self.base_steps))
        exponent = torch.randint(1, levels + 1, (count,), device=device)
        return 2.0 ** -exponent.float()

    def _flow_targets(self, x0: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, ...]:
        t = torch.rand(len(x0), device=x0.device)
        return (1 - broadcast(t)) * x0 + broadcast(t) * noise, t, noise - x0

    @torch.no_grad()
    def _bootstrap_targets(self, model: Predictor, x0: torch.Tensor, noise: torch.Tensor,
                           labels: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """x_t on a grid of multiples of 2d, and the mean of two d-steps as the target for one 2d-step."""
        d = self.step_sizes(len(x0), x0.device)
        slots = torch.floor(torch.rand(len(x0), device=x0.device) / (2 * d)).clamp(max=1 / (2 * d) - 1)
        t = (slots + 1) * 2 * d  # t in {2d, 4d, ..., 1}, so t - 2d >= 0
        x_t = (1 - broadcast(t)) * x0 + broadcast(t) * noise
        inner = torch.where(d <= 1 / self.base_steps, torch.zeros_like(d), d)  # finest step uses the velocity
        first = model(x_t, t * C.TIME_SCALE, labels, inner * C.TIME_SCALE).float()
        x_mid = x_t - broadcast(d) * first
        second = model(x_mid, (t - d) * C.TIME_SCALE, labels, inner * C.TIME_SCALE).float()
        return x_t, t, 2 * d, (first + second) / 2

    @torch.inference_mode()
    def sample(self, model: Predictor, noise: torch.Tensor, labels: torch.Tensor, null_class: int, steps: int,
               keep_every: int = 0) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """`steps` equal steps of size 1 / steps with guidance; frames are the predicted x_0."""
        if steps < 1:
            raise ConfigError("steps must be at least 1")
        grid, x, frames = time_grid(steps, noise.device), noise, []
        size = 1.0 / steps
        for i in range(steps):
            t = grid[i]
            times = torch.full((len(x),), float(t) * C.TIME_SCALE, device=x.device)
            s = guided(model, x, times, labels, null_class, self.guidance, torch.full_like(times, size * C.TIME_SCALE))
            if keep_every and i % keep_every == 0:
                frames.append((x - t * s).clamp(PIXEL_MIN, PIXEL_MAX).cpu())
            x = x - size * s
        return x.clamp(PIXEL_MIN, PIXEL_MAX), frames
