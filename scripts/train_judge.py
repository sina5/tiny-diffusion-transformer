"""Train the ResNet-18 judge for the reward fine-tuning runs (about 5 minutes on one GPU).

Example:
    uv run python scripts/train_judge.py
"""

import argparse

import torch

from tinydit import constants as C
from tinydit.data import Augmentation, Cifar10Data
from tinydit.models.classifiers import CifarResNet18Builder
from tinydit.rl.judge import JudgeTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=C.JUDGE_EPOCHS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(C.DEFAULT_SEED)
    data = Cifar10Data(C.DATA_DIR, C.JUDGE_BATCH_SIZE, C.JUDGE_WORKERS, Augmentation.BASIC).load()
    trainer = JudgeTrainer(CifarResNet18Builder(C.CIFAR10_NUM_CLASSES).build(), torch.device(args.device), args.epochs)
    accuracy = trainer.fit(data.train_loader(), data.test_loader())
    trainer.save(C.CLASSIFIER_DIR / C.JUDGE_CHECKPOINT, accuracy)
    print(f"judge test accuracy {accuracy:.4f}")


if __name__ == "__main__":
    main()
