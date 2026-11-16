"""DDPM training with DDIM sampling: the baseline of the series (Ho et al., 2020; Song et al., 2021)."""

import torch
import torch.nn.functional as F

from tinydit import constants as C
from tinydit.diffusion import DDIMSampler, NoiseSchedule
from tinydit.methods.base import GenerativeMethod, MethodKind, Predictor


class Ddpm(GenerativeMethod):
    """Predict the added noise at a random discrete timestep; sample with deterministic DDIM."""

    kind = MethodKind.DDPM
    default_steps = C.SAMPLE_STEPS
    loss_label = "Noise-prediction MSE (log scale)"

    def __init__(self, schedule: NoiseSchedule | None = None, guidance: float = C.GUIDANCE_SCALE) -> None:
        self.schedule = schedule or NoiseSchedule()
        self.guidance = guidance

    def loss(self, model: Predictor, x0: torch.Tensor, labels: torch.Tensor, null_class: int) -> torch.Tensor:
        """MSE between the predicted and the true noise."""
        t = self.schedule.sample_timesteps(len(x0), x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(model(self.schedule.add_noise(x0, t, noise), t, labels).float(), noise)

    def sample(self, model: Predictor, noise: torch.Tensor, labels: torch.Tensor, null_class: int, steps: int,
               keep_every: int = 0) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """DDIM with `steps` steps; frames are the predicted x_0."""
        return DDIMSampler(self.schedule, steps, self.guidance).sample(model, noise, labels, null_class, keep_every)
