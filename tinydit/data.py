"""Datasets and input pipelines for CIFAR-10."""

from enum import Enum
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2

from tinydit import constants as C
from tinydit.errors import ConfigError


class Augmentation(Enum):
    """How much the training images are changed before each step."""

    NONE = "none"
    BASIC = "basic"  # random crop with padding, horizontal flip
    STRONG = "strong"  # BASIC plus RandAugment


class TensorImageDataset(Dataset):
    """Images held in memory as one uint8 tensor (N, C, H, W), with a per-sample transform."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform: v2.Transform) -> None:
        if images.ndim != 4 or images.shape[0] != labels.shape[0]:
            raise ConfigError("images must be (N, C, H, W) with one label per image")
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.transform(self.images[index]), int(self.labels[index])


class CifarTransforms:
    """Builds the CIFAR-10 transforms for each augmentation level."""

    def __init__(self, mean: tuple[float, ...] = C.CIFAR10_MEAN, std: tuple[float, ...] = C.CIFAR10_STD):
        self.mean = mean
        self.std = std

    def _finish(self) -> list[v2.Transform]:
        return [v2.ToDtype(torch.float32, scale=True), v2.Normalize(self.mean, self.std)]

    def train(self, augmentation: Augmentation) -> v2.Transform:
        """Return the training transform for the given level."""
        steps: list[v2.Transform] = []
        if augmentation in (Augmentation.BASIC, Augmentation.STRONG):
            steps += [
                v2.RandomCrop(C.CIFAR10_IMAGE_SIZE, padding=C.CIFAR10_CROP_PADDING, padding_mode="reflect"),
                v2.RandomHorizontalFlip(),
            ]
        if augmentation is Augmentation.STRONG:
            steps.append(v2.RandAugment(num_ops=C.RANDAUGMENT_NUM_OPS, magnitude=C.RANDAUGMENT_MAGNITUDE))
        return v2.Compose(steps + self._finish())

    def test(self) -> v2.Transform:
        """Return the evaluation transform (normalization only)."""
        return v2.Compose(self._finish())


class BatchMixer:
    """Applies MixUp or CutMix (chosen at random per batch) and returns soft labels.

    Runs on the device after the batch is moved there, because both methods mix
    images across the batch and cannot run per sample in the loader.
    """

    def __init__(self, num_classes: int, mixup_alpha: float, cutmix_alpha: float) -> None:
        self.transform = v2.RandomChoice([
            v2.MixUp(alpha=mixup_alpha, num_classes=num_classes),
            v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes),
        ])

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mixed images and (B, num_classes) label probabilities."""
        return self.transform(images, labels)


class StratifiedSampler:
    """Draws the same share of images from every class, reproducibly."""

    def __init__(self, fraction: float, seed: int) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ConfigError(f"fraction must be in (0, 1], got {fraction}")
        self.fraction = fraction
        self.seed = seed

    def indices(self, labels: torch.Tensor) -> torch.Tensor:
        """Return sorted indices holding round(fraction * count) images of each class."""
        if self.fraction == 1.0:
            return torch.arange(labels.shape[0])
        generator = torch.Generator().manual_seed(self.seed)
        picks = []
        for cls in labels.unique():
            members = torch.nonzero(labels == cls).flatten()
            keep = max(1, round(self.fraction * members.numel()))
            picks.append(members[torch.randperm(members.numel(), generator=generator)[:keep]])
        return torch.cat(picks).sort().values


class Cifar10Data:
    """Downloads CIFAR-10 and serves train and test loaders.

    Args:
        root: folder for the dataset files.
        batch_size: images per batch.
        num_workers: loader worker processes.
        augmentation: training augmentation level.
        subset: if set, use only the first N images of each split (for smoke runs).
        train_fraction: share of each class to keep in the training split.
        fraction_seed: seed for choosing that share.
    """

    def __init__(
        self,
        root: Path,
        batch_size: int,
        num_workers: int,
        augmentation: Augmentation,
        subset: int | None = None,
        train_fraction: float = 1.0,
        fraction_seed: int = C.DEFAULT_SEED,
    ) -> None:
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.subset = subset
        self.sampler = StratifiedSampler(train_fraction, fraction_seed)
        self.transforms = CifarTransforms()
        self._splits: dict[bool, tuple[torch.Tensor, torch.Tensor]] = {}

    def load(self) -> "Cifar10Data":
        """Download (if needed) and read both splits into memory."""
        for train in (True, False):
            raw = datasets.CIFAR10(str(self.root), train=train, download=True)
            images = torch.from_numpy(raw.data).permute(0, 3, 1, 2).contiguous()
            labels = torch.tensor(raw.targets, dtype=torch.long)
            if self.subset is not None:
                images, labels = images[: self.subset], labels[: self.subset]
            if train:
                keep = self.sampler.indices(labels)
                images, labels = images[keep], labels[keep]
            self._splits[train] = (images, labels)
        return self

    def raw_split(self, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the uint8 images and labels of one split, as loaded."""
        if train not in self._splits:
            raise ConfigError("call load() before reading the data")
        return self._splits[train]

    def train_loader(self) -> DataLoader:
        """Shuffled, augmented training batches."""
        images, labels = self.raw_split(train=True)
        dataset = TensorImageDataset(images, labels, self.transforms.train(self.augmentation))
        return self._loader(dataset, shuffle=True)

    def test_loader(self) -> DataLoader:
        """Test batches in a fixed order."""
        images, labels = self.raw_split(train=False)
        return self._loader(TensorImageDataset(images, labels, self.transforms.test()), shuffle=False)

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=shuffle and len(dataset) >= self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
