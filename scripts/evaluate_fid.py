"""Compute FID-10k for a trained method at several step counts and append the rows to fid.json.

The first run downloads the FID Inception weights (about 100 MB) and caches the CIFAR-10
training statistics in runs/.

Examples:
    uv run python scripts/evaluate_fid.py --method ddpm --steps 10 20 50 100
    uv run python scripts/evaluate_fid.py --method flow_matching --steps 10 20 50 100
    uv run python scripts/evaluate_fid.py --method flow_matching --solver heun --steps 5 10 25 50
    uv run python scripts/evaluate_fid.py --method meanflow --steps 1 2 4
"""

import argparse

import torch

from tinydit import constants as C
from tinydit.data import Augmentation, Cifar10Data
from tinydit.errors import ConfigError
from tinydit.evaluation.fid import FidEvaluator, InceptionFeatures
from tinydit.experiments.diffusion import DiffusionSettings, DiffusionTrainer, GpuImageFeeder, method_run_dir
from tinydit.experiments.fid_sweep import FID_RESULTS_FILE, FidResultsFile, FidSweep
from tinydit.methods import FlowMatching, GenerativeMethod, MethodKind, OdeSolver, build_method

DEFAULT_SAMPLER = {MethodKind.DDPM: "ddim", MethodKind.FLOW_MATCHING: OdeSolver.EULER.value,
                   MethodKind.MEANFLOW: "meanflow", MethodKind.SHORTCUT: "shortcut"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", type=MethodKind, required=True, choices=list(MethodKind),
                        metavar="{" + ",".join(k.value for k in MethodKind) + "}")
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--solver", type=OdeSolver, choices=list(OdeSolver), default=None,
                        help="flow matching only: euler (default) or heun")
    parser.add_argument("--samples", type=int, default=C.FID_SAMPLES)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build(args: argparse.Namespace) -> tuple[GenerativeMethod, str]:
    """The method and the name of its sampler. Raises ConfigError for --solver on another method."""
    if args.solver is None:
        return build_method(args.method), DEFAULT_SAMPLER[args.method]
    if args.method is not MethodKind.FLOW_MATCHING:
        raise ConfigError("--solver applies to flow_matching only")
    return FlowMatching(solver=args.solver), args.solver.value


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    method, sampler = build(args)
    data = Cifar10Data(C.DATA_DIR, C.BATCH_SIZE, 0, Augmentation.NONE).load()
    images, labels = data.raw_split(train=True)
    output_dir = method_run_dir(args.method)
    trainer = DiffusionTrainer(DiffusionSettings(), GpuImageFeeder(images[:1], labels[:1], 1, device), output_dir,
                               device, method=method)
    print("loaded EMA weights from epoch", trainer.load_ema_only())
    evaluator = FidEvaluator(InceptionFeatures(device).load(), C.RUNS_DIR / C.FID_REFERENCE_FILE)
    sweep = FidSweep(trainer, evaluator, evaluator.reference(images), samples=args.samples)
    results = FidResultsFile(output_dir / FID_RESULTS_FILE)
    for steps in args.steps:
        result = sweep.run(steps, sampler)
        results.add(result)
        print(f"{result.method} {sampler} steps={steps} nfe={result.nfe:g} fid={result.fid:.2f} "
              f"({result.seconds:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
