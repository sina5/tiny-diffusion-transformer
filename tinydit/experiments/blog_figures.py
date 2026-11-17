"""Figures for the blog posts that compare the four methods and the DDRL runs.

Every number on a chart comes from the result files in runs/ (fid.json, history.json, final.json).
Sample grids are drawn fresh from the saved EMA weights, with the same noise for every panel.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tinydit import constants as C
from tinydit.errors import ResultsError
from tinydit.experiments.diffusion_figures import ImageGrid
from tinydit.plotting import ANNOTATION_SIZE, LABEL_SIZE, TICK_SIZE, TITLE_SIZE, apply_style, require, save_png

SUBTITLE_SIZE = 13
LINE_WIDTH = 2.0
MARKER_SIZE = 8
FIGURE_LEGEND_SPACE = 0.12  # share of the figure height kept free for a legend below the axes
SUPTITLE_Y = 0.99
SUBTITLE_Y = 0.935


def read_json(path: Path) -> dict | list:
    """Parsed JSON. Raises ResultsError if the file is missing."""
    if not path.exists():
        raise ResultsError(f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class FidRow:
    """One FID-10k measurement: a sampler at a step count, with its cost in model calls."""

    sampler: str
    steps: int
    nfe: float
    fid: float


class FidFile:
    """The fid.json of one run."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def rows(self, sampler: str) -> list[FidRow]:
        """Rows of one sampler, sorted by NFE. Raises ResultsError if there are none."""
        rows = [FidRow(r["sampler"], r["steps"], r["nfe"], r["fid"]) for r in read_json(self.path)
                if r["sampler"] == sampler]
        if not rows:
            raise ResultsError(f"{self.path} has no rows for sampler {sampler}")
        return sorted(rows, key=lambda r: r.nfe)

    def row(self, sampler: str, steps: int) -> FidRow:
        """The row for one setting. Raises ResultsError if it was not measured."""
        for row in self.rows(sampler):
            if row.steps == steps:
                return row
        raise ResultsError(f"{self.path} has no row for {sampler} with {steps} steps")


@dataclass(frozen=True)
class Style:
    """How one series looks on every chart."""

    label: str
    color: str
    marker: str = "o"
    linestyle: str = "-"
    label_offset: tuple[int, int] = (0, -18)  # points from the best FID to its label


@dataclass(frozen=True)
class FidSeries:
    """A FID-vs-NFE curve."""

    style: Style
    rows: list[FidRow]
    label_best: bool = True


class FidChart:
    """FID-10k against NFE, both on log axes."""

    def __init__(self, title: str, subtitle: str) -> None:
        self.title = title
        self.subtitle = subtitle
        apply_style()

    def draw(self, series: Sequence[FidSeries], path: Path) -> Path:
        """One line per series; the lowest FID of each series gets a label."""
        require(list(series))
        fig = Figure(figsize=C.FIG_SIZE, layout="constrained")
        ax = fig.subplots()
        for item in series:
            plot_fid(ax, item)
        ax.legend(loc="upper right", fontsize=LABEL_SIZE, frameon=True)
        add_titles(fig, self.title, self.subtitle)
        return save_png(fig, path)


def plot_fid(ax: Axes, series: FidSeries) -> None:
    """Draw one FID curve on log axes and label its best point."""
    nfe = [r.nfe for r in series.rows]
    fid = [r.fid for r in series.rows]
    s = series.style
    ax.plot(nfe, fid, color=s.color, marker=s.marker, linestyle=s.linestyle, linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE, label=s.label)
    best = min(series.rows, key=lambda r: r.fid)
    if series.label_best:
        _label_best(ax, best, s)
    _log_axes(ax)


def _label_best(ax: Axes, best: FidRow, s: Style) -> None:
    ax.annotate(f"{best.fid:.1f}", (best.nfe, best.fid), textcoords="offset points", xytext=s.label_offset,
                ha="center", fontsize=ANNOTATION_SIZE, color="#0b0b0b",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": s.color, "alpha": 0.9})


def _log_axes(ax: Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Model calls per image (NFE, log scale)")
    ax.set_ylabel("FID-10k (log scale, lower is better)")
    ax.yaxis.set_major_formatter(_plain_number)
    ax.yaxis.set_minor_formatter(_plain_number)
    ax.xaxis.set_major_formatter(_plain_number)


def _plain_number(value: float, _position: int | None = None) -> str:
    return f"{value:g}" if value >= 1 else ""


def add_titles(fig: Figure, title: str, subtitle: str) -> None:
    """Title and a one-line subtitle with the context (sample size, seeds)."""
    fig.suptitle(f"{title}\n", fontsize=TITLE_SIZE)
    fig.text(0.5, SUBTITLE_Y, subtitle, ha="center", va="top", fontsize=SUBTITLE_SIZE, color=C.REFERENCE_COLOR,
             transform=fig.transFigure)


@dataclass(frozen=True)
class TrainingHistory:
    """The per-epoch loss of one training run."""

    epochs: list[int]
    losses: list[float]

    @classmethod
    def load(cls, path: Path) -> "TrainingHistory":
        """Read history.json. Raises ResultsError if it is missing."""
        data = read_json(path)
        return cls(data["epochs"], [float("nan") if v is None else float(v) for v in data["losses"]])

    @property
    def non_finite_epochs(self) -> int:
        """Epochs whose logged loss is NaN or infinite."""
        return sum(not math.isfinite(v) for v in self.losses)


class TimeScaleFigure:
    """MeanFlow with time scale 1000 next to time scale 1: the loss per epoch and FID against NFE."""

    def __init__(self, title: str, subtitle: str) -> None:
        self.title = title
        self.subtitle = subtitle
        apply_style()

    def draw(self, runs: Sequence[tuple[Style, TrainingHistory, list[FidRow]]], path: Path) -> Path:
        """Left: the logged loss (log scale, gaps are NaN epochs). Right: FID-10k against NFE."""
        fig = Figure(figsize=C.FIG_SIZE, layout="constrained")
        loss_ax, fid_ax = fig.subplots(1, 2)
        for style, history, rows in runs:
            loss_ax.plot(history.epochs, history.losses, color=style.color, linewidth=LINE_WIDTH, label=style.label)
            plot_fid(fid_ax, FidSeries(style, rows))
        loss_ax.set_yscale("log")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Logged loss (per-pixel MSE, log scale)")
        loss_ax.set_title("Training loss")
        fid_ax.set_title("FID-10k of the final EMA weights")
        handles, labels = loss_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=len(runs), fontsize=LABEL_SIZE)
        add_titles(fig, self.title, self.subtitle)
        return save_png(fig, path)


@dataclass(frozen=True)
class DdrlRun:
    """One fine-tuning run: the evaluations every 25 iterations (FID-2k) and the final row (FID-10k)."""

    name: str
    evaluations: list[dict]
    final: dict

    @classmethod
    def load(cls, run_dir: Path) -> "DdrlRun":
        """Read history.json and final.json. Raises ResultsError if either is missing."""
        history = read_json(run_dir / "history.json")
        return cls(run_dir.name, history["evaluations"], read_json(run_dir / "final.json"))

    def curve(self, key: str) -> tuple[list[int], list[float]]:
        """(iterations, values) of one evaluation metric."""
        return [e["iteration"] for e in self.evaluations], [e[key] for e in self.evaluations]


@dataclass(frozen=True)
class DdrlPanelRow:
    """One row of the curves figure: a title and the runs it shows."""

    title: str
    runs: list[tuple[Style, DdrlRun]]


class DdrlCurvesFigure:
    """Reward and judge accuracy against iteration, one row per learning rate."""

    METRICS = (("reward", "Reward: mean log p(class), reward ViT"), ("judge_accuracy", "Judge ResNet-18 accuracy"))

    def __init__(self, title: str, subtitle: str, base: dict) -> None:
        self.title = title
        self.subtitle = subtitle
        self.base = base
        apply_style()

    def draw(self, rows: Sequence[DdrlPanelRow], path: Path) -> Path:
        """A grid of len(rows) x 2 panels with one shared legend below."""
        require(list(rows))
        fig = Figure(figsize=C.FIG_SIZE, layout="constrained")
        axes = fig.subplots(len(rows), len(self.METRICS), squeeze=False, sharex=True)
        legend: dict[str, object] = {}
        for row, row_axes in zip(rows, axes):
            for (key, label), ax in zip(self.METRICS, row_axes):
                self._panel(ax, row, key, label, legend)
        for ax in axes[-1]:
            ax.set_xlabel("Iteration")
        fig.legend(list(legend.values()), list(legend.keys()), loc="outside lower center", ncols=4,
                   fontsize=TICK_SIZE)
        add_titles(fig, self.title, self.subtitle)
        return save_png(fig, path)

    def _panel(self, ax: Axes, row: DdrlPanelRow, key: str, label: str, legend: dict[str, object]) -> None:
        ax.axhline(self.base[key], color=C.REFERENCE_COLOR, linestyle=":", linewidth=LINE_WIDTH)
        for style, run in row.runs:
            x, y = run.curve(key)
            (line,) = ax.plot(x, y, color=style.color, marker=style.marker, linestyle=style.linestyle,
                              linewidth=LINE_WIDTH, markersize=MARKER_SIZE - 2)
            legend.setdefault(style.label, line)
        legend.setdefault("Base model, 10,000-sample evaluation", ax.get_lines()[0])
        ax.set_title(f"{row.title}: {label}", fontsize=LABEL_SIZE)
        if key.endswith("accuracy"):
            ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")


class FidRewardFigure:
    """Final FID-10k against the final reward, one point per run, each labeled."""

    def __init__(self, title: str, subtitle: str) -> None:
        self.title = title
        self.subtitle = subtitle
        apply_style()

    def draw(self, points: Sequence[tuple[Style, dict]], path: Path, broken: Sequence[tuple[Style, dict]]) -> Path:
        """Points inside the axes; runs listed in `broken` are named in a note, since their FID is off the scale."""
        require(list(points))
        fig = Figure(figsize=C.FIG_SIZE, layout="constrained")
        ax = fig.subplots()
        for style, final in points:
            ax.scatter(final["reward"], final["fid"], s=200, color=style.color, marker=style.marker,
                       edgecolors="white", linewidths=1, zorder=3)
            ax.annotate(style.label, (final["reward"], final["fid"]), textcoords="offset points",
                        xytext=style.label_offset, ha="right" if style.label_offset[0] < 0 else "left",
                        fontsize=ANNOTATION_SIZE)
        ax.set_xlabel("Final reward: mean log p(class) from the reward ViT (higher is better)")
        ax.set_ylabel("Final FID-10k (lower is better)")
        ax.margins(x=0.25, y=0.15)
        if broken:
            note = "Off the scale: " + "; ".join(f"{s.label} (reward {f['reward']:.2f}, FID-10k {f['fid']:.1f})"
                                                 for s, f in broken)
            ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=ANNOTATION_SIZE,
                    bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": C.REFERENCE_COLOR})
        add_titles(fig, self.title, self.subtitle)
        return save_png(fig, path)


@dataclass(frozen=True)
class GridPanel:
    """One panel of a sample grid: a title and the images (class-major, per_class per class)."""

    title: str
    images: torch.Tensor


class SampleGridFigure:
    """Panels side by side; the rows are the CIFAR-10 classes."""

    def __init__(self, title: str, subtitle: str, per_class: int, grid: ImageGrid | None = None) -> None:
        self.title = title
        self.subtitle = subtitle
        self.per_class = per_class
        self.grid = grid or ImageGrid()
        apply_style()

    def draw(self, panels: Sequence[GridPanel], path: Path) -> Path:
        """Each panel shows per_class images for each class; the first panel names the classes."""
        require(list(panels))
        fig = Figure(figsize=C.FIG_SIZE, layout="constrained")
        axes = fig.subplots(1, len(panels), squeeze=False)[0]
        for index, (panel, ax) in enumerate(zip(panels, axes)):
            image = np.asarray(self.grid.build(panel.images, self.per_class))
            ax.imshow(image)
            ax.set_title(panel.title, fontsize=LABEL_SIZE)
            ax.set_xticks([])
            ax.grid(False)
            self._class_ticks(ax, image.shape[0], show=index == 0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        add_titles(fig, self.title, self.subtitle)
        return save_png(fig, path)

    @staticmethod
    def _class_ticks(ax: Axes, height: int, show: bool) -> None:
        rows = len(C.CIFAR10_CLASSES)
        ax.set_yticks([(r + 0.5) * height / rows for r in range(rows)] if show else [])
        if show:
            ax.set_yticklabels(C.CIFAR10_CLASSES, fontsize=TICK_SIZE)
        ax.tick_params(length=0)


def first_per_class(images: torch.Tensor, stored_per_class: int, keep: int) -> torch.Tensor:
    """From class-major images with stored_per_class per class, keep the first `keep` of each class."""
    if keep > stored_per_class:
        raise ResultsError(f"only {stored_per_class} images per class are stored, not {keep}")
    classes = len(images) // stored_per_class
    return torch.cat([images[c * stored_per_class:c * stored_per_class + keep] for c in range(classes)])


class ComparisonNoise:
    """The same class-major noise and labels for every panel of a comparison grid."""

    def __init__(self, per_class: int = C.COMPARE_PER_CLASS, seed: int = C.COMPARE_SEED,
                 num_classes: int = C.CIFAR10_NUM_CLASSES, image_size: int = C.CIFAR10_IMAGE_SIZE,
                 channels: int = 3) -> None:
        self.per_class = per_class
        self.seed = seed
        self.num_classes = num_classes
        self.shape = (channels, image_size, image_size)

    def make(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(noise, labels) with labels grouped class after class."""
        generator = torch.Generator().manual_seed(self.seed)
        labels = torch.arange(self.num_classes).repeat_interleave(self.per_class)
        return torch.randn(len(labels), *self.shape, generator=generator), labels
