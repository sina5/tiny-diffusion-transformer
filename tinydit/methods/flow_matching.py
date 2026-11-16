"""Flow matching with a straight path from data to noise (Lipman et al., 2023; Liu et al., 2023).

x_t = (1 - t) x_0 + t * noise, with t = 1 at pure noise. The velocity along this path is
v = noise - x_0, and the model learns v. Sampling solves the ODE dx/dt = v from t = 1 to t = 0.
"""

from enum import Enum

import torch
import torch.nn.functional as F

from tinydit import constants as C
from tinydit.errors import ConfigError
from tinydit.methods.base import (
    PIXEL_MAX, PIXEL_MIN, GenerativeMethod, MethodKind, Predictor, broadcast, guided, logit_normal, time_grid,
)


class OdeSolver(Enum):
    """Euler takes one model call per step. Heun takes two (except the last step) and is second order."""

    EULER = "euler"
    HEUN = "heun"


class FlowMatching(GenerativeMethod):
    """Velocity prediction on the linear path, with logit-normal training times."""

    kind = MethodKind.FLOW_MATCHING
    default_steps = C.FLOW_SAMPLE_STEPS
    loss_label = "Velocity MSE (log scale)"

    def __init__(self, guidance: float = C.GUIDANCE_SCALE, solver: OdeSolver = OdeSolver.EULER,
                 logit_mean: float = C.FLOW_LOGIT_MEAN, logit_std: float = C.FLOW_LOGIT_STD) -> None:
        if logit_std <= 0:
            raise ConfigError("logit_std must be positive")
        self.guidance = guidance
        self.solver = solver
        self.logit_mean = logit_mean
        self.logit_std = logit_std

    def loss(self, model: Predictor, x0: torch.Tensor, labels: torch.Tensor, null_class: int) -> torch.Tensor:
        """MSE between the predicted and the true velocity noise - x_0."""
        t = logit_normal(len(x0), self.logit_mean, self.logit_std, x0.device)
        noise = torch.randn_like(x0)
        xt = (1 - broadcast(t)) * x0 + broadcast(t) * noise
        return F.mse_loss(model(xt, t * C.TIME_SCALE, labels).float(), noise - x0)

    @torch.inference_mode()
    def sample(self, model: Predictor, noise: torch.Tensor, labels: torch.Tensor, null_class: int, steps: int,
               keep_every: int = 0) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Solve the ODE in `steps` intervals; frames are the predicted x_0 = x_t - t v."""
        if steps < 1:
            raise ConfigError("steps must be at least 1")
        grid, x, frames = time_grid(steps, noise.device), noise, []
        for i in range(steps):
            t, t_next = grid[i], grid[i + 1]
            v = self.velocity(model, x, t, labels, null_class)
            if keep_every and i % keep_every == 0:
                frames.append((x - t * v).clamp(PIXEL_MIN, PIXEL_MAX).cpu())
            x = self._step(model, x, v, t, t_next, labels, null_class)
        return x.clamp(PIXEL_MIN, PIXEL_MAX), frames

    def velocity(self, model: Predictor, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor,
                 null_class: int) -> torch.Tensor:
        """Guided velocity at the scalar time t for the whole batch."""
        times = torch.full((len(x),), float(t) * C.TIME_SCALE, device=x.device)
        return guided(model, x, times, labels, null_class, self.guidance)

    def _step(self, model: Predictor, x: torch.Tensor, v: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor,
              labels: torch.Tensor, null_class: int) -> torch.Tensor:
        euler = x + (t_next - t) * v
        if self.solver is OdeSolver.EULER or float(t_next) == 0.0:
            return euler
        v_next = self.velocity(model, euler, t_next, labels, null_class)
        return x + (t_next - t) * (v + v_next) / 2
