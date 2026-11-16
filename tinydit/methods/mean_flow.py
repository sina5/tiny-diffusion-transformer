"""MeanFlow: learn the average velocity over an interval, for one-step generation (Geng et al., 2025).

On the flow-matching path z_t = (1 - t) x_0 + t * noise, the average velocity from time r to
time t is u(z_t, r, t) = 1 / (t - r) * integral of v from r to t. One step then maps z_t to
z_r = z_t - (t - r) u. Differentiating (t - r) u = integral of v with respect to t gives the
MeanFlow identity u = v - (t - r) du/dt, where du/dt = v . grad_z u + d_t u is a
Jacobian-vector product (JVP). The training target uses that identity with a stop-gradient.

Classifier-free guidance is part of the target, so sampling needs only one model call per step.

The DiT gets the times in [0, 1] directly (time scale 1, as in the paper), not scaled by 1000 like
the other methods. The target contains du/dt, and with the 1000x scale the time derivative of
the sinusoidal embedding is about 1000x larger; the target then grows with the model and training
diverges (tested: the loss rose from 4 to above 1e4 in the first epochs).
"""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tinydit import constants as C
from tinydit.errors import ConfigError
from tinydit.methods.base import (
    PIXEL_MAX, PIXEL_MIN, GenerativeMethod, MethodKind, Predictor, broadcast, logit_normal, time_grid,
)


class MeanFlow(GenerativeMethod):
    """Average-velocity model u(z, r, t). The DiT gets t and the interval length h = t - r."""

    kind = MethodKind.MEANFLOW
    needs_interval = True
    default_steps = C.MEANFLOW_SAMPLE_STEPS
    loss_label = "Average-velocity MSE (log scale)"

    def __init__(self, guidance: float = C.GUIDANCE_SCALE, unequal_share: float = C.MEANFLOW_UNEQUAL_SHARE,
                 logit_mean: float = C.MEANFLOW_LOGIT_MEAN, logit_std: float = C.MEANFLOW_LOGIT_STD,
                 adaptive_power: float = C.MEANFLOW_ADAPTIVE_POWER, adaptive_eps: float = C.MEANFLOW_ADAPTIVE_EPS,
                 time_scale: float = C.MEANFLOW_TIME_SCALE):
        if not 0.0 <= unequal_share <= 1.0:
            raise ConfigError("unequal_share must be in [0, 1]")
        self.guidance = guidance
        self.unequal_share = unequal_share
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.adaptive_power = adaptive_power
        self.adaptive_eps = adaptive_eps
        self.time_scale = time_scale

    def loss(self, model: Predictor, x0: torch.Tensor, labels: torch.Tensor, null_class: int) -> torch.Tensor:
        """Adaptive-weighted squared error between u and the stop-gradient MeanFlow target."""
        r, t = self.sample_times(len(x0), x0.device)
        noise = torch.randn_like(x0)
        z = (1 - broadcast(t)) * x0 + broadcast(t) * noise
        v = self._guided_target_velocity(model, z, t, noise - x0, labels, null_class)

        target = v - broadcast(t - r) * self.time_derivative(model, z, r, t, v, labels)
        prediction = model(z, t * self.time_scale, labels, (t - r) * self.time_scale).float()
        error = (prediction - target).pow(2).flatten(1).sum(dim=1)
        weight = (error.detach() + self.adaptive_eps).pow(-self.adaptive_power)
        weighted = (weight * error).mean()
        # The weighted loss is close to 1 by design, so report the plain per-pixel MSE; the gradient is unchanged.
        return weighted - weighted.detach() + error.detach().mean() / x0[0].numel()

    def time_derivative(self, model: Predictor, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor, v: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
        """du/dt = v . grad_z u + d_t u, as a JVP. Zero where r = t, since (t - r) du/dt is then zero anyway.

        The target is detached, so the JVP runs without a graph; the caller's forward pass carries the
        gradient. (Backward through the JVP also fails under bf16 autocast.)
        """
        du_dt = torch.zeros(v.shape, device=v.device)
        rows = (r < t).nonzero().squeeze(1)
        if len(rows) == 0:
            return du_dt

        def average_velocity(z_in: torch.Tensor, r_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            return model(z_in, t_in * self.time_scale, labels[rows], (t_in - r_in) * self.time_scale)

        primals = (z[rows], r[rows], t[rows])
        tangents = (v[rows], torch.zeros_like(primals[1]), torch.ones_like(primals[2]))
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):  # fused attention has no forward-mode derivative
            _, partial = torch.func.jvp(average_velocity, primals, tangents)
        du_dt[rows] = partial.float()
        return du_dt

    def sample_times(self, count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """(r, t) with r <= t. A share of 1 - unequal_share has r = t, which is plain flow matching."""
        a = logit_normal(count, self.logit_mean, self.logit_std, device)
        b = logit_normal(count, self.logit_mean, self.logit_std, device)
        t, r = torch.maximum(a, b), torch.minimum(a, b)
        equal = torch.rand(count, device=device) >= self.unequal_share
        return torch.where(equal, t, r), t

    def _guided_target_velocity(self, model: Predictor, z: torch.Tensor, t: torch.Tensor, v: torch.Tensor,
                                labels: torch.Tensor, null_class: int) -> torch.Tensor:
        """v_cfg = w v + (1 - w) u(z, t, t | no class) for labeled samples; v itself for dropped labels."""
        if self.guidance == 1.0:
            return v
        with torch.no_grad():
            times = t * self.time_scale
            uncond = model(z, times, torch.full_like(labels, null_class), torch.zeros_like(times)).float()
        mixed = self.guidance * v + (1 - self.guidance) * uncond
        return torch.where(broadcast(labels != null_class), mixed, v)

    @torch.inference_mode()
    def sample(self, model: Predictor, noise: torch.Tensor, labels: torch.Tensor, null_class: int, steps: int,
               keep_every: int = 0) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """z_r = z_t - (t - r) u(z_t, r, t) on an even grid; one model call per step, no guidance pass."""
        if steps < 1:
            raise ConfigError("steps must be at least 1")
        grid, z, frames = time_grid(steps, noise.device), noise, []
        for i in range(steps):
            t, r = grid[i], grid[i + 1]
            times = torch.full((len(z),), float(t) * self.time_scale, device=z.device)
            u = model(z, times, labels, torch.full_like(times, float(t - r) * self.time_scale)).float()
            z = z - (t - r) * u
            if keep_every and i % keep_every == 0:
                frames.append(z.clamp(PIXEL_MIN, PIXEL_MAX).cpu())
        return z.clamp(PIXEL_MIN, PIXEL_MAX), frames
