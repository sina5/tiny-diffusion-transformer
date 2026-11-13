"""Figures for the diffusion run: sample grids, the progress over epochs, the loss, the denoising path,
and a memorization check against the nearest training images."""

from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image, ImageDraw

from tinydit import constants as C
from tinydit.errors import ResultsError
from tinydit.experiments.diffusion import DiffusionHistory
from tinydit.plotting import LABEL_SIZE, TITLE_SIZE, apply_style, require, save_png

UINT8_MAX = 255
GRID_PAD = 2
PROGRESS_PANELS = 6
PROGRESS_ROWS = 4  # classes shown per progress panel
LABEL_BAND = 18  # pixels above each GIF frame for the epoch label
NEIGHBOR_COLUMNS = 2  # sample, nearest training image
GRID_BACKGROUND = 255


class ImageGrid:
    """Turns (N, C, H, W) images in [-1, 1] into one PIL image."""

    def __init__(self, upscale: int = C.GRID_UPSCALE, pad: int = GRID_PAD) -> None:
        self.upscale = upscale
        self.pad = pad

    @staticmethod
    def to_uint8(images: torch.Tensor) -> np.ndarray:
        """(N, H, W, C) uint8 array."""
        scaled = ((images.float().clamp(-1, 1) + 1) * (UINT8_MAX / 2)).round().to(torch.uint8)
        return scaled.permute(0, 2, 3, 1).numpy()

    def build(self, images: torch.Tensor, columns: int) -> Image.Image:
        """Row-major grid with a white gap between images."""
        pixels = self.to_uint8(images)
        rows = -(-len(pixels) // columns)
        size = pixels.shape[1] * self.upscale
        canvas = Image.new("RGB", (columns * (size + self.pad) + self.pad, rows * (size + self.pad) + self.pad),
                           (GRID_BACKGROUND,) * 3)
        for n, tile in enumerate(pixels):
            row, col = divmod(n, columns)
            image = Image.fromarray(tile).resize((size, size), Image.Resampling.NEAREST)
            canvas.paste(image, (self.pad + col * (size + self.pad), self.pad + row * (size + self.pad)))
        return canvas


class SnapshotStore:
    """Reads the saved snapshot tensors by epoch."""

    def __init__(self, snapshot_dir: Path) -> None:
        self.snapshot_dir = snapshot_dir

    def load(self, epoch: int) -> torch.Tensor:
        """The fixed-noise samples at one epoch."""
        path = self.snapshot_dir / f"epoch_{epoch:04d}.pt"
        if not path.exists():
            raise ResultsError(f"{path} is missing")
        return torch.load(path, weights_only=True).float()


class DiffusionFigures:
    """Writes every figure into one folder."""

    def __init__(self, history: DiffusionHistory, store: SnapshotStore, per_class: int, output_dir: Path) -> None:
        require(history.snapshots)
        self.history = history
        self.store = store
        self.per_class = per_class
        self.output_dir = output_dir
        self.grid = ImageGrid()
        apply_style()

    def final_grid(self) -> Path:
        """All classes (rows) at the last snapshot."""
        epoch = self.history.snapshots[-1]
        path = self.output_dir / "samples_final.png"
        self.grid.build(self.store.load(epoch), self.per_class).save(path)
        return path

    def animation(self) -> Path:
        """A GIF with one frame per snapshot and the epoch written above it."""
        frames = [self._labeled(self.grid.build(self.store.load(e), self.per_class), f"epoch {e}")
                  for e in self.history.snapshots]
        path = self.output_dir / "samples_over_training.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=C.GIF_FRAME_MS, loop=0)
        return path

    def progress(self) -> Path:
        """The same noise at a few epochs, side by side (first classes only, to stay readable)."""
        epochs = self._spread(self.history.snapshots, PROGRESS_PANELS)
        fig = Figure(figsize=C.FIG_SIZE)
        axes = fig.subplots(1, len(epochs), squeeze=False)[0]
        count = PROGRESS_ROWS * self.per_class
        for ax, epoch in zip(axes, epochs, strict=True):
            ax.imshow(self.grid.build(self.store.load(epoch)[:count], self.per_class // 2))
            ax.set_title(f"epoch {epoch}", fontsize=LABEL_SIZE)
            ax.axis("off")
        fig.suptitle("The same starting noise, sampled after more and more training", fontsize=TITLE_SIZE)
        fig.tight_layout()
        return save_png(fig, self.output_dir / "progress.png")

    def loss_curve(self) -> Path:
        """Training loss per epoch on a log scale."""
        fig = Figure(figsize=C.FIG_SIZE)
        ax = fig.subplots()
        ax.plot(self.history.epochs, self.history.losses, color=C.VIT_COLOR, linewidth=2)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Noise-prediction MSE (log scale)")
        ax.set_title("DiT training loss on CIFAR-10", fontsize=TITLE_SIZE)
        fig.tight_layout()
        return save_png(fig, self.output_dir / "loss.png")

    def trajectory(self, frames: list[torch.Tensor], rows: int) -> Path:
        """Predicted clean image at evenly spaced sampling steps, for a few samples."""
        picks = self._spread(list(range(len(frames))), C.TRAJECTORY_FRAMES)
        stacked = torch.stack([frames[i][:rows] for i in picks], dim=1).flatten(0, 1)
        fig = Figure(figsize=C.FIG_SIZE)
        ax = fig.subplots()
        ax.imshow(self.grid.build(stacked, len(picks)))
        ax.set_title("Predicted clean image at 10 points while sampling (left: first step, right: final)", fontsize=TITLE_SIZE)
        ax.axis("off")
        fig.tight_layout()
        return save_png(fig, self.output_dir / "denoising_path.png")

    def neighbors(self, samples: torch.Tensor, train_images: torch.Tensor) -> Path:
        """Each sample next to the training image closest in pixel space (a memorization check)."""
        matches = NearestTrainingImage(train_images).find(samples)
        pairs = torch.stack([samples, matches], dim=1).flatten(0, 1)
        fig = Figure(figsize=C.FIG_SIZE)
        ax = fig.subplots()
        ax.imshow(self.grid.build(pairs, NEIGHBOR_COLUMNS * len(samples) // 2))
        ax.set_title("Samples (left of each pair) and their nearest training image (right)", fontsize=TITLE_SIZE)
        ax.axis("off")
        fig.tight_layout()
        return save_png(fig, self.output_dir / "nearest_training_images.png")

    @staticmethod
    def _spread(items: list[int], count: int) -> list[int]:
        if len(items) <= count:
            return list(items)
        return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]

    @staticmethod
    def _labeled(image: Image.Image, text: str) -> Image.Image:
        canvas = Image.new("RGB", (image.width, image.height + LABEL_BAND), (GRID_BACKGROUND,) * 3)
        canvas.paste(image, (0, LABEL_BAND))
        ImageDraw.Draw(canvas).text((GRID_PAD, GRID_PAD), text, fill=(0, 0, 0))
        return canvas


class NearestTrainingImage:
    """Finds, for each sample, the training image with the smallest L2 distance in pixel space."""

    def __init__(self, train_images: torch.Tensor) -> None:
        self.train = train_images.float().flatten(1)  # in [-1, 1]

    def find(self, samples: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) training images, one per sample."""
        distances = torch.cdist(samples.float().flatten(1), self.train)
        return self.train[distances.argmin(dim=1)].view_as(samples)
