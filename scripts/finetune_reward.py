"""Fine-tune the DDPM model to raise a classifier reward, with or without a regularizer.

Starts from the EMA weights of the DDPM run (runs/checkpoint.pt) and writes to
runs/ddrl/<run name>/. Needs the reward ViT and the judge in runs/classifiers/
(scripts/train_judge.py makes the judge).

Examples:
    uv run python scripts/finetune_reward.py --regularizer none
    uv run python scripts/finetune_reward.py --regularizer reverse_kl --beta 0.1
    uv run python scripts/finetune_reward.py --regularizer data --beta 0.1
    uv run python scripts/finetune_reward.py --regularizer none --lr 1e-4 --seed 1   # runs/ddrl/reward_only_lr0.0001_seed1/
    uv run python scripts/finetune_reward.py --regularizer data --iterations 4 --eval-every 2 --smoke
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from tinydit import constants as C
from tinydit.data import Augmentation, Cifar10Data
from tinydit.evaluation.fid import FidEvaluator, InceptionFeatures
from tinydit.experiments.diffusion import CHECKPOINT_FILE, GpuImageFeeder
from tinydit.models.dit import DiT, DiTConfig
from tinydit.rl.ddrl import Regularizer, RewardFineTuner, RlSettings
from tinydit.rl.evaluation import RlEvaluator
from tinydit.rl.reward import ClassifierLoader

FINAL_FILE = "final.json"
SMOKE_EVAL_SAMPLES = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regularizer", type=Regularizer, choices=list(Regularizer), default=Regularizer.DATA)
    parser.add_argument("--beta", type=float, default=None, help="regularizer weight (default per regularizer)")
    parser.add_argument("--iterations", type=int, default=C.RL_ITERATIONS)
    parser.add_argument("--eval-every", type=int, default=C.RL_EVAL_EVERY)
    parser.add_argument("--lr", type=float, default=C.RL_LR)
    parser.add_argument("--seed", type=int, default=C.DEFAULT_SEED, help="seed for rollouts and data batches")
    parser.add_argument("--smoke", action="store_true", help="few eval samples, to runs/ddrl/smoke/")
    parser.add_argument("--base-only", action="store_true", help="evaluate the base model to runs/ddrl/base/")
    parser.add_argument("--classifier-dir", type=Path, default=C.CLASSIFIER_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_base(device: torch.device) -> tuple[DiT, DiTConfig]:
    """The DDPM model with its EMA weights."""
    state = torch.load(C.RUNS_DIR / CHECKPOINT_FILE, map_location=device, weights_only=False)
    config = DiTConfig(**state["config"])
    model = DiT(config).to(device)
    model.load_state_dict(state["ema"])
    return model, config


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.allow_tf32 = False  # keep step means exact enough for log-probabilities (see sampler)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    default_beta = C.RL_KL_BETA if args.regularizer is Regularizer.REVERSE_KL else C.RL_DATA_BETA
    settings = RlSettings(args.regularizer, default_beta if args.beta is None else args.beta, args.iterations,
                          args.lr, eval_every=args.eval_every, seed=args.seed)
    data = Cifar10Data(C.DATA_DIR, C.BATCH_SIZE, 0, Augmentation.NONE).load()
    (train_x, train_y), (test_x, test_y) = data.raw_split(True), data.raw_split(False)
    policy, config = load_base(device)
    classifiers = ClassifierLoader(args.classifier_dir)
    reward, judge = classifiers.reward_vit(device), classifiers.judge_resnet(device)
    fid = FidEvaluator(InceptionFeatures(device).load(), C.RUNS_DIR / C.FID_REFERENCE_FILE)
    samples = SMOKE_EVAL_SAMPLES if args.smoke else C.RL_EVAL_SAMPLES
    evaluator = RlEvaluator(reward, judge, fid, fid.reference(train_x), test_x, test_y, device, samples)
    final_samples = SMOKE_EVAL_SAMPLES if args.smoke else C.RL_FINAL_FID_SAMPLES
    if args.base_only:
        write_final(evaluator.evaluate(policy, 0, fid_samples=final_samples), C.RUNS_DIR / "ddrl" / "base")
        return
    output_dir = C.RUNS_DIR / "ddrl" / ("smoke" if args.smoke else settings.name)
    reference = copy.deepcopy(policy) if args.regularizer is Regularizer.REVERSE_KL else None
    tuner = RewardFineTuner(settings, policy, reward, evaluator, GpuImageFeeder(train_x, train_y, C.BATCH_SIZE, device),
                            output_dir, device, config.null_class, reference)
    history = tuner.fit()
    final = evaluator.evaluate(policy, settings.iterations, fid_samples=final_samples)
    final["seconds"] = sum(history.seconds)
    write_final(final, output_dir)


def write_final(row: dict, output_dir: Path) -> None:
    """Write the final evaluation (FID-10k on a full run) next to the history."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / FINAL_FILE).write_text(json.dumps(row, indent=2), encoding="utf-8")
    print("final", row)


if __name__ == "__main__":
    main()
