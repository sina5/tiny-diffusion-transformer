"""The interface that every generative method implements, and helpers they share.

A method owns two things: the training loss (what the DiT must predict at which time) and the
sampler (how to turn noise into an image with that prediction). The trainer, the figures and
the FID evaluation only talk to this interface, so every method trains on the same model,
data, optimizer and EMA.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum

import torch

from tinydit import constants as C

# (x, t, y, h) -> prediction. h is None for methods without an interval input.
Predictor = Callable[..., torch.Tensor]
PIXEL_MIN, PIXEL_MAX = -1.0, 1.0


class MethodKind(Enum):
    """The generative methods this project trains. The value is the run folder name."""

    DDPM = "ddpm"
    FLOW_MATCHING = "flow_matching"
    MEANFLOW = "meanflow"
    SHORTCUT = "shortcut"


class GenerativeMethod(ABC):
    """Training loss plus sampler for one way of learning to generate images."""

    kind: MethodKind
    needs_interval: bool = False  # the DiT gets a second time input (the step length)
    default_steps: int = C.SAMPLE_STEPS
    loss_label: str = "Training loss"

    @abstractmethod
    def loss(self, model: Predictor, x0: torch.Tensor, labels: torch.Tensor, null_class: int) -> torch.Tensor:
        """Scalar training loss for clean images x0 in [-1, 1]. Dropped labels equal null_class."""

    @abstractmethod
    def sample(self, model: Predictor, noise: torch.Tensor, labels: torch.Tensor, null_class: int, steps: int,
               keep_every: int = 0) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Images in [-1, 1] from noise in `steps` steps and, if keep_every > 0, the frames on the way."""


def guided(model: Predictor, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, null_class: int, scale: float,
           h: torch.Tensor | None = None) -> torch.Tensor:
    """Classifier-free guidance: uncond + scale * (cond - uncond), from one batched forward pass."""
    if scale == 1.0:
        return model(x, t, y, h) if h is not None else model(x, t, y)
    both_h = None if h is None else torch.cat([h, h])
    args = (torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, torch.full_like(y, null_class)]))
    both = model(*args, both_h) if h is not None else model(*args)
    cond, uncond = both.float().chunk(2)
    return uncond + scale * (cond - uncond)


def logit_normal(count: int, mean: float, std: float, device: torch.device) -> torch.Tensor:
    """Times in (0, 1) whose logit is normal: more weight on the middle noise levels."""
    return torch.sigmoid(torch.randn(count, device=device) * std + mean)


def broadcast(values: torch.Tensor) -> torch.Tensor:
    """(B,) -> (B, 1, 1, 1), so per-sample times scale images."""
    return values.view(-1, 1, 1, 1)


def time_grid(steps: int, device: torch.device) -> list[torch.Tensor]:
    """steps + 1 evenly spaced scalar times from 1 (noise) to 0 (data)."""
    return list(torch.linspace(1.0, 0.0, steps + 1, device=device))
