"""Scores generated images with a frozen CIFAR-10 classifier."""

from pathlib import Path

import torch
from torch import nn

from tinydit import constants as C
from tinydit.errors import CheckpointError
from tinydit.models.classifiers import CifarResNet18Builder, ViTClassifier, ViTConfig


class ClassifierScore:
    """log p(label | image) and accuracy from a frozen classifier, for images in [-1, 1]."""

    def __init__(self, classifier: nn.Module, mean: tuple[float, ...] = C.CIFAR10_MEAN,
                 std: tuple[float, ...] = C.CIFAR10_STD) -> None:
        self.classifier = classifier.eval().requires_grad_(False)
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    @torch.no_grad()
    def logits(self, images: torch.Tensor) -> torch.Tensor:
        """(B, classes) logits; the classifier was trained on mean/std-normalized [0, 1] images."""
        unit = (images.float().clamp(-1, 1) + 1) / 2
        normalized = (unit - self.mean.to(images.device)) / self.std.to(images.device)
        return self.classifier(normalized).float()

    def log_prob(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """(B,) log-probability of the intended class: the reward."""
        return self.logits(images).log_softmax(dim=1).gather(1, labels[:, None]).squeeze(1)

    def accuracy(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        """Share of images classified as their intended class."""
        return float((self.logits(images).argmax(dim=1) == labels).float().mean())


class ClassifierLoader:
    """Reads the reward ViT and the judge ResNet-18 from their checkpoints."""

    def __init__(self, folder: Path = C.CLASSIFIER_DIR, num_classes: int = C.CIFAR10_NUM_CLASSES) -> None:
        self.folder = folder
        self.num_classes = num_classes

    def reward_vit(self, device: torch.device) -> ClassifierScore:
        """The ViT from the ViT-from-scratch post. Raises CheckpointError if the file is missing."""
        state = self._read(C.REWARD_CHECKPOINT)
        model = ViTClassifier(ViTConfig(**state["config"]))
        model.load_state_dict(state["state_dict"])
        return ClassifierScore(model.to(device))

    def judge_resnet(self, device: torch.device) -> ClassifierScore:
        """The ResNet-18 from scripts/train_judge.py. Raises CheckpointError if the file is missing."""
        model = CifarResNet18Builder(self.num_classes).build()
        model.load_state_dict(self._read(C.JUDGE_CHECKPOINT)["state_dict"])
        return ClassifierScore(model.to(device))

    def _read(self, name: str) -> dict:
        path = self.folder / name
        if not path.exists():
            raise CheckpointError(f"{path} is missing")
        return torch.load(path, map_location="cpu", weights_only=False)
