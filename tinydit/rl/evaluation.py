"""Measures a fine-tuned model on fixed noise: the reward it was trained on and the signs of
reward hacking (a second classifier, FID and the diffusion loss on held-out images)."""

from pathlib import Path

import torch
from torch import nn

from tinydit import constants as C
from tinydit.diffusion import NoiseSchedule
from tinydit.evaluation.fid import FeatureStats, FidEvaluator
from tinydit.rl.reward import ClassifierScore
from tinydit.rl.sampler import StochasticDdim

UINT8_MAX = 255.0
SNAPSHOT_DIR = "snapshots"
GRID_PER_CLASS = 8  # the first images of each class are saved as the sample grid


class RlEvaluator:
    """Samples the same noise (and the same step noise) at every evaluation."""

    def __init__(self, reward: ClassifierScore, judge: ClassifierScore, fid: FidEvaluator, reference: FeatureStats,
                 test_images: torch.Tensor, test_labels: torch.Tensor, device: torch.device, samples: int = C.RL_EVAL_SAMPLES,
                 steps: int = C.RL_SAMPLE_STEPS, num_classes: int = C.CIFAR10_NUM_CLASSES,
                 seed: int = C.RL_EVAL_SEED, schedule: NoiseSchedule | None = None) -> None:
        self.reward = reward
        self.judge = judge
        self.fid = fid
        self.reference = reference
        self.device = device
        self.samples = samples
        self.seed = seed
        self.schedule = schedule or NoiseSchedule()
        self.sampler = StochasticDdim(self.schedule, steps)
        self.image_shape = tuple(test_images.shape[1:])
        self.num_classes = num_classes
        self.test = test_images[:samples].float() / (UINT8_MAX / 2) - 1.0
        self.test_labels = test_labels[:samples]
        generator = torch.Generator().manual_seed(seed)
        self.test_t = torch.randint(0, self.schedule.timesteps, (len(self.test),), generator=generator)
        self.test_noise = torch.randn(self.test.shape, generator=generator)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, iteration: int, output_dir: Path | None = None,
                 fid_samples: int | None = None) -> dict:
        """Reward, both accuracies, FID and the held-out diffusion loss; saves a grid if output_dir is set."""
        images = self.generate(model, fid_samples or self.samples)
        labels = self.labels_for(len(images)).to(self.device)
        row = {"iteration": iteration,
               "reward": float(self.reward.log_prob(images.to(self.device), labels).mean()),
               "reward_accuracy": self.reward.accuracy(images.to(self.device), labels),
               "judge_accuracy": self.judge.accuracy(images.to(self.device), labels),
               "fid": self.fid.score(images, self.reference),
               "test_diffusion_loss": self.held_out_loss(model)}
        if output_dir is not None:
            self._save_grid(images, output_dir / SNAPSHOT_DIR / f"iter_{iteration:04d}.pt")
        return row

    def labels_for(self, count: int) -> torch.Tensor:
        """Class-balanced labels, grouped class after class."""
        per_class = -(-count // self.num_classes)
        return torch.arange(self.num_classes).repeat_interleave(per_class)[:count]

    def generate(self, model: nn.Module, count: int) -> torch.Tensor:
        """count images (CPU, [-1, 1]) from fixed noise and fixed step noise."""
        generator = torch.Generator().manual_seed(self.seed)
        labels = self.labels_for(count)
        chunks = []
        for start in range(0, count, C.FID_BATCH):
            batch = labels[start:start + C.FID_BATCH].to(self.device)
            noise = torch.randn(len(batch), *self.image_shape, generator=generator).to(self.device)
            chunks.append(self.sampler.rollout(model, noise, batch, generator).images.cpu())
        return torch.cat(chunks)

    def held_out_loss(self, model: nn.Module) -> float:
        """DDPM loss on test images with fixed timesteps and noise. A rise means the model left the data."""
        total = 0.0
        for start in range(0, len(self.test), C.FID_BATCH):
            stop = start + C.FID_BATCH
            x0, t = self.test[start:stop].to(self.device), self.test_t[start:stop].to(self.device)
            noise = self.test_noise[start:stop].to(self.device)
            labels = self.test_labels[start:stop].to(self.device)
            prediction = model(self.schedule.add_noise(x0, t, noise), t, labels).float()
            total += float((prediction - noise).pow(2).mean()) * len(x0)
        return total / len(self.test)

    def _save_grid(self, images: torch.Tensor, path: Path) -> None:
        per_class = -(-len(images) // self.num_classes)
        picks = torch.cat([images[c * per_class:c * per_class + GRID_PER_CLASS] for c in range(self.num_classes)])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(picks.half(), path)

