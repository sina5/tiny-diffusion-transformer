"""Reward fine-tuning of the DDPM model, with three ways to keep it close to the data.

Every iteration samples groups of rollouts that share a label, scores them with a frozen
classifier and normalizes the reward inside each group (the advantage A). The policy-gradient
loss is -A log p_theta(x_s | x_t), summed over every second transition (DDRL, Ye et al., 2025,
Algorithm 1; on-policy, so no importance ratio is needed). The regularizer is one of:

- none: reward only. The model is free to drift; this is the reward-hacking baseline.
- reverse KL: the KL from each policy step to the same step of the frozen base model
  (the regularizer of KL-regularized methods such as Flow-GRPO).
- data (DDRL): the ordinary diffusion loss on real training images, weighted by beta. The paper
  shows that this is a forward KL to the data distribution, so no reference model is needed.
  This project uses one random timestep per real image (the paper's "reduced" variant).
"""

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import torch
from torch import nn

from tinydit import constants as C
from tinydit.errors import ConfigError
from tinydit.experiments.diffusion import GpuImageFeeder
from tinydit.methods import Ddpm
from tinydit.rl.evaluation import RlEvaluator
from tinydit.rl.reward import ClassifierScore
from tinydit.rl.sampler import StochasticDdim

HISTORY_FILE = "history.json"
POLICY_FILE = "policy.pt"


class Regularizer(Enum):
    """What keeps the fine-tuned model close to the data."""

    NONE = "none"
    REVERSE_KL = "reverse_kl"
    DATA = "data"


@dataclass(frozen=True)
class RlSettings:
    """One fine-tuning run."""

    regularizer: Regularizer = Regularizer.DATA
    beta: float = C.RL_DATA_BETA  # weight of the regularizer (ignored for NONE)
    iterations: int = C.RL_ITERATIONS
    lr: float = C.RL_LR
    groups: int = C.RL_GROUPS
    group_size: int = C.RL_GROUP_SIZE
    sample_steps: int = C.RL_SAMPLE_STEPS
    train_every: int = C.RL_TRAIN_EVERY
    eval_every: int = C.RL_EVAL_EVERY
    seed: int = C.DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ConfigError("group_size must be at least 2 to normalize rewards inside a group")
        if self.beta < 0 or self.iterations < 1 or self.eval_every < 1:
            raise ConfigError("beta must be non-negative; iterations and eval_every positive")

    @property
    def name(self) -> str:
        """Run folder name, such as data_beta0.1, with _lr and _seed suffixes for non-default values."""
        name = "reward_only" if self.regularizer is Regularizer.NONE else f"{self.regularizer.value}_beta{self.beta:g}"
        if self.lr != C.RL_LR:
            name += f"_lr{self.lr:g}"
        if self.seed != C.DEFAULT_SEED:
            name += f"_seed{self.seed}"
        return name


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """(r - group mean) / group std, for rewards laid out group after group."""
    grouped = rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    return (centered / (grouped.std(dim=1, keepdim=True) + C.ADVANTAGE_EPS)).flatten()


@dataclass
class RlHistory:
    """Training reward per iteration and the evaluation rows."""

    settings: dict = field(default_factory=dict)
    train_reward: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Write as JSON."""
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


class RewardFineTuner:
    """Runs the iterations, evaluates every few of them and saves the history and the policy."""

    def __init__(self, settings: RlSettings, policy: nn.Module, reward: ClassifierScore, evaluator: RlEvaluator,
                 feeder: GpuImageFeeder, output_dir: Path, device: torch.device, null_class: int,
                 reference: nn.Module | None = None) -> None:
        """Raises ConfigError if the reverse-KL run has no reference model."""
        if settings.regularizer is Regularizer.REVERSE_KL and reference is None:
            raise ConfigError("the reverse-KL regularizer needs the frozen base model")
        self.settings = settings
        self.policy = policy
        self.reward = reward
        self.evaluator = evaluator
        self.feeder = feeder
        self.output_dir = output_dir
        self.device = device
        self.null_class = null_class
        self.reference = reference.eval().requires_grad_(False) if reference is not None else None
        self.sampler = StochasticDdim(evaluator.schedule, settings.sample_steps)
        self.data_loss = Ddpm(evaluator.schedule)
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=settings.lr, weight_decay=0.0)
        self.history = RlHistory(settings={**asdict(settings), "regularizer": settings.regularizer.value})
        self._real = iter(())

    def fit(self) -> RlHistory:
        """Evaluate the base model, then train and evaluate every eval_every iterations."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._evaluate(0)
        for iteration in range(1, self.settings.iterations + 1):
            began = time.perf_counter()
            reward = self._iteration()
            self.history.train_reward.append(reward)
            self.history.seconds.append(time.perf_counter() - began)
            if iteration % self.settings.eval_every == 0 or iteration == self.settings.iterations:
                self._evaluate(iteration)
        torch.save({"policy": self.policy.state_dict(), "settings": self.history.settings},
                   self.output_dir / POLICY_FILE)
        return self.history

    def _iteration(self) -> float:
        labels = torch.randint(0, self.null_class, (self.settings.groups,), device=self.device)
        labels = labels.repeat_interleave(self.settings.group_size)
        noise = torch.randn(len(labels), *self.evaluator.image_shape, device=self.device)
        self.policy.eval()
        rollout = self.sampler.rollout(self.policy, noise, labels)
        rewards = self.reward.log_prob(rollout.images, labels)
        advantages = group_advantages(rewards, self.settings.group_size)
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        steps = self.sampler.trainable_steps(self.settings.train_every)
        for index in steps:
            self._policy_step(rollout.states[index], rollout.states[index + 1], index, labels, advantages, len(steps))
        if self.settings.regularizer is Regularizer.DATA:
            (self.settings.beta * self._real_data_loss()).backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), C.GRAD_CLIP_NORM)
        self.optimizer.step()
        return float(rewards.mean())

    def _policy_step(self, x: torch.Tensor, x_next: torch.Tensor, index: int, labels: torch.Tensor,
                     advantages: torch.Tensor, count: int) -> None:
        """Backward of -A log p (plus the KL term) for one transition; gradients add up over steps."""
        mean = self.sampler.mean(self.policy, x, index, labels)
        sigma = self.sampler.sigma(index)
        log_prob = -(x_next - mean).pow(2).flatten(1).mean(dim=1) / (2 * sigma ** 2)
        loss = -(advantages * log_prob).mean()
        if self.settings.regularizer is Regularizer.REVERSE_KL:
            with torch.no_grad():
                ref_mean = self.sampler.mean(self.reference, x, index, labels)
            # KL of two Gaussians with the same sigma, per pixel
            loss = loss + self.settings.beta * ((mean - ref_mean).pow(2).flatten(1).mean(dim=1) / (2 * sigma ** 2)).mean()
        (loss / count).backward()

    def _real_data_loss(self) -> torch.Tensor:
        try:
            x0, labels = next(self._real)
        except StopIteration:
            self._real = iter(self.feeder)
            x0, labels = next(self._real)
        drop = torch.rand(len(labels), device=self.device) < C.LABEL_DROP
        labels = torch.where(drop, torch.full_like(labels, self.null_class), labels)
        return self.data_loss.loss(self.policy, x0, labels, self.null_class)

    def _evaluate(self, iteration: int) -> None:
        self.policy.eval()
        row = self.evaluator.evaluate(self.policy, iteration, self.output_dir)
        self.history.evaluations.append(row)
        self.history.save(self.output_dir / HISTORY_FILE)
        print(f"[{self.settings.name}] iter {iteration:4d} " + " ".join(
            f"{k}={v:.4f}" for k, v in row.items() if k != "iteration"), flush=True)
