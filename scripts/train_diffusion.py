"""Train a tiny class-conditional DiT on CIFAR-10 with one generative method, then draw the figures.

The DDPM baseline writes to runs/. Every other method writes to runs/<method>/.

Examples:
    uv run python scripts/train_diffusion.py --smoke
    uv run python scripts/train_diffusion.py
    uv run python scripts/train_diffusion.py --method flow_matching
    uv run python scripts/train_diffusion.py --method meanflow --figures-only
"""

import argparse
from pathlib import Path

import torch

from tinydit import constants as C
from tinydit.data import Augmentation, Cifar10Data
from tinydit.experiments.diffusion import (
    SNAPSHOT_DIR, DiffusionSettings, DiffusionTrainer, GpuImageFeeder, method_run_dir,
)
from tinydit.experiments.diffusion_figures import UINT8_MAX, DiffusionFigures, SnapshotStore
from tinydit.methods import MethodKind, build_method

METHOD_TITLES = {MethodKind.DDPM: "DDPM", MethodKind.FLOW_MATCHING: "flow matching",
                 MethodKind.MEANFLOW: "MeanFlow", MethodKind.SHORTCUT: "shortcut model"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", type=MethodKind, default=MethodKind.DDPM, choices=list(MethodKind),
                        metavar="{" + ",".join(k.value for k in MethodKind) + "}")
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="2 epochs on 512 images, to a separate folder")
    parser.add_argument("--figures-only", action="store_true", help="draw the figures from the saved run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(C.DEFAULT_SEED)
    settings = DiffusionSettings(epochs=args.epochs, batch_size=args.batch_size)
    output_dir = method_run_dir(args.method)
    if args.smoke:
        settings, output_dir = settings.smoke(), output_dir / "smoke"
    device = torch.device(args.device)
    data = Cifar10Data(C.DATA_DIR, settings.batch_size, 0, Augmentation.NONE, subset=settings.subset).load()
    images, labels = data.raw_split(train=True)
    feeder = GpuImageFeeder(images, labels, settings.batch_size, device)
    trainer = DiffusionTrainer(settings, feeder, output_dir, device, method=build_method(args.method))
    if args.figures_only:
        print("loaded EMA weights from epoch", trainer.load_ema_only())
    else:
        history = trainer.fit()
        print(f"params={history.params:,} peak_vram={history.peak_vram_mib} MiB "
              f"train_time={sum(history.seconds) / 60:.1f} min")
    draw_figures(trainer, images, output_dir, METHOD_TITLES[args.method])


def draw_figures(trainer: DiffusionTrainer, train_images: torch.Tensor, output_dir: Path, title: str) -> None:
    """Write the sample grid, GIF, progress, loss, denoising path and nearest-neighbor figures."""
    figures = DiffusionFigures(trainer.history, SnapshotStore(output_dir / SNAPSHOT_DIR), trainer.fixed.per_class,
                               output_dir, trainer.method.loss_label, title)
    samples, frames = trainer.sample(keep_every=max(1, trainer.sample_steps // C.TRAJECTORY_FRAMES))
    one_per_class = samples[::trainer.fixed.per_class][:C.NEIGHBOR_SAMPLES]
    paths = [
        figures.final_grid(), figures.animation(), figures.progress(), figures.loss_curve(),
        figures.trajectory([f[::trainer.fixed.per_class] for f in frames], rows=C.TRAJECTORY_ROWS),
        figures.neighbors(one_per_class, train_images.float() / (UINT8_MAX / 2) - 1.0),
    ]
    for path in paths:
        print("wrote", path)


if __name__ == "__main__":
    main()
