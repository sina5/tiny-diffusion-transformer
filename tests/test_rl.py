from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tinydit.diffusion import NoiseSchedule
from tinydit.errors import CheckpointError, ConfigError
from tinydit.experiments.diffusion import GpuImageFeeder
from tinydit.models.classifiers import ViTClassifier, ViTConfig
from tinydit.models.dit import DiT, DiTConfig
from tinydit.rl.ddrl import Regularizer, RewardFineTuner, RlSettings, group_advantages
from tinydit.rl.judge import JudgeTrainer
from tinydit.rl.reward import ClassifierLoader, ClassifierScore
from tinydit.rl.sampler import StochasticDdim

CPU = torch.device("cpu")
TINY = DiTConfig(image_size=8, patch_size=2, dim=16, depth=2, heads=2, num_classes=3, time_freq_dim=8)
TINY_VIT = ViTConfig(image_size=8, patch_size=4, num_classes=3, dim=16, depth=1, num_heads=2)


def _noisy_dit() -> DiT:
    """A DiT whose output is not exactly zero, so gradients and log-probabilities are non-trivial."""
    model = DiT(TINY)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


# --- stochastic sampler ----------------------------------------------------------


@pytest.fixture
def sampler() -> StochasticDdim:
    return StochasticDdim(NoiseSchedule(), steps=6)


def test_only_the_last_step_is_deterministic(sampler: StochasticDdim) -> None:
    sigmas = [sampler.sigma(i) for i in range(sampler.steps)]
    assert sigmas[-1] == 0.0 and all(s > 0 for s in sigmas[:-1])


def test_trainable_steps_skip_the_deterministic_last_step(sampler: StochasticDdim) -> None:
    assert sampler.trainable_steps(2) == [0, 2, 4]
    assert sampler.trainable_steps(1) == [0, 1, 2, 3, 4]


def test_rollout_keeps_every_state_and_repeats_with_a_generator(sampler: StochasticDdim) -> None:
    model, noise, labels = _noisy_dit(), torch.randn(2, 3, 8, 8), torch.tensor([0, 1])
    a = sampler.rollout(model, noise, labels, torch.Generator().manual_seed(1))
    b = sampler.rollout(model, noise, labels, torch.Generator().manual_seed(1))
    assert len(a.states) == sampler.steps + 1 and torch.equal(a.images, b.images)


def test_log_prob_is_highest_at_the_mean(sampler: StochasticDdim) -> None:
    model, x, labels = _noisy_dit(), torch.randn(2, 3, 8, 8), torch.tensor([0, 1])
    mean = sampler.mean(model, x, 1, labels)
    at_mean = sampler.log_prob(model, x, mean, 1, labels)
    away = sampler.log_prob(model, x, mean + 0.1, 1, labels)
    assert torch.allclose(at_mean, torch.zeros(2)) and torch.all(away < at_mean)


def test_log_prob_refuses_the_deterministic_step(sampler: StochasticDdim) -> None:
    x = torch.randn(1, 3, 8, 8)
    with pytest.raises(ConfigError):
        sampler.log_prob(_noisy_dit(), x, x, sampler.steps - 1, torch.tensor([0]))


def test_sampler_needs_two_steps() -> None:
    with pytest.raises(ConfigError):
        StochasticDdim(NoiseSchedule(), steps=1)


# --- advantages and settings -------------------------------------------------------


def test_group_advantages_are_standardized_inside_each_group() -> None:
    advantages = group_advantages(torch.tensor([1.0, 2.0, 3.0, 10.0, 10.0, 40.0]), group_size=3).view(2, 3)
    assert torch.allclose(advantages.mean(dim=1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(advantages.std(dim=1), torch.ones(2), atol=1e-4)


def test_equal_rewards_give_zero_advantage() -> None:
    assert torch.count_nonzero(group_advantages(torch.ones(4), group_size=2)) == 0


@pytest.mark.parametrize(("settings", "name"), [
    (RlSettings(Regularizer.NONE), "reward_only"),
    (RlSettings(Regularizer.DATA, beta=0.1), "data_beta0.1"),
    (RlSettings(Regularizer.REVERSE_KL, beta=0.01), "reverse_kl_beta0.01"),
    (RlSettings(Regularizer.NONE, lr=1e-4, seed=1), "reward_only_lr0.0001_seed1"),
])
def test_run_names(settings: RlSettings, name: str) -> None:
    assert settings.name == name


@pytest.mark.parametrize("kwargs", [{"group_size": 1}, {"beta": -1.0}, {"iterations": 0}])
def test_settings_reject_bad_values(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        RlSettings(**kwargs)


# --- rewards and classifiers --------------------------------------------------------


def test_classifier_reward_is_a_log_probability() -> None:
    score = ClassifierScore(ViTClassifier(TINY_VIT))
    images, labels = torch.rand(4, 3, 8, 8) * 2 - 1, torch.tensor([0, 1, 2, 0])
    assert torch.all(score.log_prob(images, labels) <= 0) and 0.0 <= score.accuracy(images, labels) <= 1.0


def test_vit_classifier_round_trips_its_state_dict() -> None:
    model = ViTClassifier(TINY_VIT)
    assert ViTClassifier(TINY_VIT).load_state_dict(model.state_dict()).missing_keys == []


def test_loader_reports_a_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError):
        ClassifierLoader(tmp_path).reward_vit(CPU)


def test_judge_trainer_trains_and_saves(tmp_path: Path) -> None:
    data = DataLoader(TensorDataset(torch.randn(16, 3, 8, 8), torch.arange(16) % 3), batch_size=8)
    trainer = JudgeTrainer(nn.Sequential(nn.Flatten(), nn.Linear(192, 3)), CPU, epochs=1)
    accuracy = trainer.fit(data, data)
    trainer.save(tmp_path / "judge.pt", accuracy)
    assert 0.0 <= accuracy <= 1.0 and (tmp_path / "judge.pt").exists()


# --- fine-tuner --------------------------------------------------------------------


class FakeEvaluator:
    """Stands in for RlEvaluator: no FID network, records each call."""

    def __init__(self) -> None:
        self.schedule = NoiseSchedule()
        self.image_shape = (3, 8, 8)
        self.calls: list[int] = []

    def evaluate(self, model: nn.Module, iteration: int, output_dir: Path | None = None) -> dict:
        self.calls.append(iteration)
        return {"iteration": iteration, "fid": 0.0}


def _tuner(tmp_path: Path, regularizer: Regularizer, reference: nn.Module | None = None):
    settings = RlSettings(regularizer, beta=0.1, iterations=2, eval_every=1, groups=2, group_size=2,
                          sample_steps=4, lr=1e-3)
    images = torch.randint(0, 256, (8, 3, 8, 8), dtype=torch.uint8)
    feeder = GpuImageFeeder(images, torch.arange(8) % 3, 4, CPU)
    evaluator = FakeEvaluator()
    tuner = RewardFineTuner(settings, _noisy_dit(), ClassifierScore(ViTClassifier(TINY_VIT)), evaluator, feeder,
                            tmp_path, CPU, TINY.null_class, reference)
    return tuner, evaluator


@pytest.mark.parametrize("regularizer", list(Regularizer))
def test_fine_tuner_changes_the_policy_and_saves_history(tmp_path: Path, regularizer: Regularizer) -> None:
    reference = _noisy_dit() if regularizer is Regularizer.REVERSE_KL else None
    tuner, evaluator = _tuner(tmp_path, regularizer, reference)
    before = [p.detach().clone() for p in tuner.policy.parameters()]
    history = tuner.fit()
    changed = any(not torch.equal(a, b) for a, b in zip(before, tuner.policy.parameters(), strict=True))
    assert changed and evaluator.calls == [0, 1, 2] and len(history.train_reward) == 2
    assert (tmp_path / "history.json").exists() and (tmp_path / "policy.pt").exists()


def test_reverse_kl_needs_a_reference(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        _tuner(tmp_path, Regularizer.REVERSE_KL)
