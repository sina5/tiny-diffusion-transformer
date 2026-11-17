"""Draw the figures for the blog posts on flow matching, MeanFlow, shortcut models and DDRL.

Charts read only the result files in runs/. Sample grids load the saved EMA weights and sample
the same noise for every panel, so they need the checkpoints and, ideally, a GPU.

Examples:
    uv run python scripts/make_blog_figures.py                 # everything, to runs/blog_figures/
    uv run python scripts/make_blog_figures.py --charts-only   # no sampling
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from tinydit import constants as C
from tinydit.experiments.blog_figures import (
    ComparisonNoise, DdrlCurvesFigure, DdrlPanelRow, DdrlRun, FidChart, FidFile, FidRewardFigure, FidRow, FidSeries,
    GridPanel, SampleGridFigure, Style, TimeScaleFigure, TrainingHistory, first_per_class, read_json,
)
from tinydit.experiments.diffusion import DiffusionSettings, DiffusionTrainer, GpuImageFeeder, method_run_dir
from tinydit.methods import FlowMatching, GenerativeMethod, MeanFlow, MethodKind, build_method

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = C.SERIES_COLORS
DIVERGED_DIR = C.RUNS_DIR / "meanflow_time1000_diverged"
DDRL_DIR = C.RUNS_DIR / "ddrl"
DIVERGED_TIME_SCALE = 1000.0
DDRL_SNAPSHOT_PER_CLASS = 8
WIDE_PER_CLASS = 8  # samples per class when a grid has only two panels
FID_SUBTITLE = "FID-10k: 10,000 class-balanced samples against the 50,000 training images. One training run per method."
GRID_SUBTITLE = "The same noise and labels in every panel. Guidance scale 2; a guided step is two calls."
LABEL_ABOVE = (0, 14)
SCATTER_LABEL = (10, 6)

DDRL_STYLES = {
    "reward_only": Style("Reward only", BLUE, "o", "-", SCATTER_LABEL),
    "reward_only_seed1": Style("Reward only, seed 1", BLUE, "o", "--", (10, -18)),
    "reverse_kl_beta0.1": Style("Reverse KL, beta 0.1", ORANGE, "s", "-", SCATTER_LABEL),
    "data_beta0.01": Style("Data, beta 0.01", AQUA, "^", "-", SCATTER_LABEL),
    "data_beta0.01_seed1": Style("Data, beta 0.01, seed 1", AQUA, "^", "--", SCATTER_LABEL),
    "data_beta0.1": Style("Data, beta 0.1", YELLOW, "v", "-", SCATTER_LABEL),
    "data_beta1": Style("Data, beta 1", MAGENTA, "D", "-", (-12, 6)),
    "reward_only_lr0.0001": Style("Reward only, lr 1e-4", BLUE, "X", ":", SCATTER_LABEL),
    "data_beta0.01_lr0.0001": Style("Data, beta 0.01, lr 1e-4", AQUA, "P", ":", SCATTER_LABEL),
}
LOW_LR_RUNS = ["reward_only", "reward_only_seed1", "reverse_kl_beta0.1", "data_beta0.01", "data_beta0.01_seed1",
               "data_beta0.1", "data_beta1"]
HIGH_LR_RUNS = ["reward_only_lr0.0001", "data_beta0.01_lr0.0001"]


@dataclass(frozen=True)
class Sampler:
    """A trained run and the sampler settings to draw with. `fid_sampler` is its name in fid.json."""

    name: str
    fid_sampler: str
    method: GenerativeMethod
    run_dir: Path


def samplers() -> dict[str, Sampler]:
    """Every trained model the figures read from."""
    return {
        "ddpm": Sampler("DDPM (DDIM)", "ddim", build_method(MethodKind.DDPM), method_run_dir(MethodKind.DDPM)),
        "fm": Sampler("Flow matching (Euler)", "euler", FlowMatching(), method_run_dir(MethodKind.FLOW_MATCHING)),
        "meanflow": Sampler("MeanFlow", "meanflow", MeanFlow(), method_run_dir(MethodKind.MEANFLOW)),
        "meanflow1000": Sampler("MeanFlow, time scale 1000", "meanflow", MeanFlow(time_scale=DIVERGED_TIME_SCALE),
                                DIVERGED_DIR),
        "shortcut": Sampler("Shortcut model", "shortcut", build_method(MethodKind.SHORTCUT),
                            method_run_dir(MethodKind.SHORTCUT)),
    }


class GridMaker:
    """Samples the same noise with several trained models and turns each into a titled panel."""

    def __init__(self, device: torch.device, seed: int = C.COMPARE_SEED) -> None:
        self.device = device
        self.seed = seed
        self.samplers = samplers()

    def panel(self, key: str, steps: int, per_class: int) -> GridPanel:
        """One panel; the title gives the steps, the NFE and the FID-10k from that run's fid.json."""
        sampler = self.samplers[key]
        row = FidFile(sampler.run_dir / "fid.json").row(sampler.fid_sampler, steps)
        noise, labels = ComparisonNoise(per_class, self.seed).make()
        images, _ = self._trainer(sampler).generate(noise, labels, steps)
        step_word = "step" if steps == 1 else "steps"
        return GridPanel(f"{sampler.name}\n{steps} {step_word}, NFE {row.nfe:g}, FID {row.fid:.1f}", images)

    def _trainer(self, sampler: Sampler) -> DiffusionTrainer:
        dummy = torch.zeros(1, 3, C.CIFAR10_IMAGE_SIZE, C.CIFAR10_IMAGE_SIZE, dtype=torch.uint8)
        feeder = GpuImageFeeder(dummy, torch.zeros(1, dtype=torch.long), 1, self.device)
        trainer = DiffusionTrainer(DiffusionSettings(), feeder, sampler.run_dir, self.device, method=sampler.method)
        trainer.load_ema_only()
        return trainer


@dataclass(frozen=True)
class GridSpec:
    """One sample-grid figure: file, title, subtitle, samples per class and (sampler key, steps) panels."""

    file_name: str
    title: str
    subtitle: str
    per_class: int
    panels: list[tuple[str, int]]


GRID_SPECS = [
    GridSpec("grid_flow_matching_vs_ddpm.png", "DDPM and flow matching at the same number of model calls",
             GRID_SUBTITLE, C.COMPARE_PER_CLASS, [("ddpm", 4), ("fm", 4), ("ddpm", 16), ("fm", 16)]),
    GridSpec("grid_meanflow_time_scale.png", "MeanFlow in one step: time scale 1000 against time scale 1",
             "The same noise and labels in both panels. Guidance is part of the MeanFlow target, so one step is one call.",
             WIDE_PER_CLASS, [("meanflow1000", 1), ("meanflow", 1)]),
    GridSpec("grid_nfe2.png", "Two model calls per image", GRID_SUBTITLE, C.COMPARE_PER_CLASS,
             [("ddpm", 1), ("fm", 1), ("meanflow", 2), ("shortcut", 1)]),
    GridSpec("grid_nfe8.png", "Eight model calls per image", GRID_SUBTITLE, C.COMPARE_PER_CLASS,
             [("ddpm", 4), ("fm", 4), ("meanflow", 8), ("shortcut", 4)]),
]


class BlogFigures:
    """Writes every figure of the four posts into one folder."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.samplers = samplers()

    def _fid(self, key: str, sampler: str | None = None) -> list[FidRow]:
        entry = self.samplers[key]
        return FidFile(entry.run_dir / "fid.json").rows(sampler or entry.fid_sampler)

    def fid_flow_vs_ddpm(self) -> Path:
        """Post 1: DDPM against flow matching with both solvers."""
        series = [FidSeries(Style("DDPM, DDIM", BLUE, "o", label_offset=LABEL_ABOVE), self._fid("ddpm")),
                  FidSeries(Style("Flow matching, Euler", ORANGE, "s"), self._fid("fm")),
                  FidSeries(Style("Flow matching, Heun", ORANGE, "D", "--"), self._fid("fm", "heun"), label_best=False)]
        chart = FidChart("Flow matching reaches a given FID with fewer model calls than DDPM",
                         FID_SUBTITLE + " Guidance scale 2: a guided step is two calls.")
        return chart.draw(series, self.output_dir / "fid_flow_matching_vs_ddpm.png")

    def fid_all_methods(self) -> Path:
        """Posts 2 and 3: the four methods on one chart."""
        series = [FidSeries(Style("DDPM, DDIM", BLUE, "o", label_offset=LABEL_ABOVE), self._fid("ddpm")),
                  FidSeries(Style("Flow matching, Euler", ORANGE, "s"), self._fid("fm")),
                  FidSeries(Style("MeanFlow (1 call per step)", AQUA, "^"), self._fid("meanflow")),
                  FidSeries(Style("Shortcut model (2 calls per step)", YELLOW, "v", label_offset=LABEL_ABOVE),
                            self._fid("shortcut"))]
        chart = FidChart("The flow-map methods win up to 32 calls; flow matching wins from 64",
                         FID_SUBTITLE + " Same DiT, data, optimizer and 200 epochs.")
        return chart.draw(series, self.output_dir / "fid_all_methods.png")

    def meanflow_time_scale(self) -> Path:
        """Post 2: the diverged run with time scale 1000 next to the fixed run."""
        broken = TrainingHistory.load(DIVERGED_DIR / "history.json")
        fixed = TrainingHistory.load(self.samplers["meanflow"].run_dir / "history.json")
        runs = [(Style("Time scale 1000 (first run)", MAGENTA, "o"), broken, self._fid("meanflow1000")),
                (Style("Time scale 1 (fixed)", AQUA, "^"), fixed, self._fid("meanflow"))]
        subtitle = (f"Same settings and seed; only the time input changes. The first run logged NaN in "
                    f"{broken.non_finite_epochs} of {len(broken.epochs)} epochs (gaps in the left line).")
        figure = TimeScaleFigure("MeanFlow: scaling the time input by 1000 makes the loss diverge", subtitle)
        return figure.draw(runs, self.output_dir / "meanflow_time_scale.png")

    def ddrl_charts(self) -> list[Path]:
        """Post 4: reward and judge curves, and final FID against final reward."""
        base = read_json(DDRL_DIR / "base" / "final.json")
        runs = {name: DdrlRun.load(DDRL_DIR / name) for name in DDRL_STYLES}
        rows = [DdrlPanelRow("lr 1e-5", [(DDRL_STYLES[n], runs[n]) for n in LOW_LR_RUNS]),
                DdrlPanelRow("lr 1e-4", [(DDRL_STYLES[n], runs[n]) for n in HIGH_LR_RUNS])]
        curves = DdrlCurvesFigure(
            "Reward fine-tuning: the reward ViT and the held-out judge mostly agree, until lr 1e-4",
            "Evaluated every 25 iterations on the same 2,000 samples (stochastic DDIM, 20 steps, no guidance). "
            "Dotted line: base model, 10,000 samples.", base)
        points = [(Style("Base model", C.REFERENCE_COLOR, "*", "-", SCATTER_LABEL), base)]
        points += [(DDRL_STYLES[n], runs[n].final) for n in LOW_LR_RUNS + ["data_beta0.01_lr0.0001"]]
        scatter = FidRewardFigure("Data regularization with a small beta gives the highest reward and the lowest FID",
                                  "Final evaluation on 10,000 samples after 300 iterations. Seed 0 unless named.")
        broken = [(DDRL_STYLES["reward_only_lr0.0001"], runs["reward_only_lr0.0001"].final)]
        return [curves.draw(rows, self.output_dir / "ddrl_curves.png"),
                scatter.draw(points, self.output_dir / "ddrl_fid_vs_reward.png", broken)]

    def ddrl_samples(self, per_class: int = C.COMPARE_PER_CLASS) -> Path:
        """Post 4: the saved evaluation samples (same noise) of the base model and four runs."""
        picks = [("reward_only", 0, "Base model"), ("reward_only", 300, "Reward only"),
                 ("data_beta0.01", 300, "Data, beta 0.01"), ("data_beta0.01_lr0.0001", 300, "Data, beta 0.01, lr 1e-4"),
                 ("reward_only_lr0.0001", 300, "Reward only, lr 1e-4")]
        panels = []
        for run, iteration, title in picks:
            images = torch.load(DDRL_DIR / run / "snapshots" / f"iter_{iteration:04d}.pt", weights_only=True).float()
            panels.append(GridPanel(f"{title}\niteration {iteration}",
                                    first_per_class(images, DDRL_SNAPSHOT_PER_CLASS, per_class)))
        figure = SampleGridFigure("The same noise before and after 300 iterations of reward fine-tuning",
                                  "Stochastic DDIM, 20 steps, no guidance. Seed 0 runs.", per_class)
        return figure.draw(panels, self.output_dir / "ddrl_samples.png")

    def grids(self, maker: GridMaker) -> list[Path]:
        """Sample grids at equal NFE."""
        paths = []
        for spec in GRID_SPECS:
            panels = [maker.panel(key, steps, spec.per_class) for key, steps in spec.panels]
            figure = SampleGridFigure(spec.title, spec.subtitle, spec.per_class)
            paths.append(figure.draw(panels, self.output_dir / spec.file_name))
        return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=C.BLOG_FIGURES_DIR)
    parser.add_argument("--charts-only", action="store_true", help="skip the sample grids (no checkpoints needed)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures = BlogFigures(args.output)
    paths = [figures.fid_flow_vs_ddpm(), figures.fid_all_methods(), figures.meanflow_time_scale(),
             *figures.ddrl_charts(), figures.ddrl_samples()]
    if not args.charts_only:
        paths += figures.grids(GridMaker(torch.device(args.device)))
    for path in paths:
        print("wrote", path.relative_to(C.REPO_ROOT) if path.is_relative_to(C.REPO_ROOT) else path.name)


if __name__ == "__main__":
    main()
