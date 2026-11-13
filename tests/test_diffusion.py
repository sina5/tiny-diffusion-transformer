from pathlib import Path

import pytest
import torch

from tinydit.diffusion import DDIMSampler, ExponentialMovingAverage, NoiseSchedule
from tinydit.errors import CheckpointError, ConfigError
from tinydit.experiments.diffusion import (
    SNAPSHOT_DIR, DiffusionSettings, DiffusionTrainer, GpuImageFeeder,
)
from tinydit.experiments.diffusion_figures import (
    DiffusionFigures, ImageGrid, NearestTrainingImage, SnapshotStore,
)
from tinydit.models.dit import DiT, DiTConfig

CPU = torch.device("cpu")
TINY = DiTConfig(image_size=8, patch_size=2, dim=16, depth=2, heads=2, num_classes=3, time_freq_dim=8)


def test_dit_output_matches_input_shape() -> None:
    x = torch.randn(4, 3, 8, 8)
    assert DiT(TINY)(x, torch.tensor([0, 1, 2, 999]), torch.tensor([0, 1, 2, 3])).shape == x.shape


def test_dit_starts_by_predicting_zero_noise() -> None:
    """adaLN-Zero and the zero output layer make the untrained output exactly zero."""
    out = DiT(TINY)(torch.randn(2, 3, 8, 8), torch.tensor([5, 50]), torch.tensor([0, 3]))
    assert torch.count_nonzero(out) == 0


def test_dit_unpatchify_inverts_patch_order() -> None:
    model, image = DiT(TINY), torch.randn(1, 3, 8, 8)
    patches = image.unfold(2, 2, 2).unfold(3, 2, 2)  # (1, C, g, g, p, p)
    tokens = patches.permute(0, 2, 3, 4, 5, 1).reshape(1, 16, 12)
    assert torch.equal(model._unpatchify(tokens), image)


@pytest.mark.parametrize("kwargs", [{"image_size": 9}, {"dim": 15}, {"time_freq_dim": 7}])
def test_dit_config_rejects_bad_shapes(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        DiTConfig(**{**TINY.__dict__, **kwargs})


def test_schedule_signal_share_falls_from_one_toward_zero() -> None:
    alpha_bar = NoiseSchedule().alpha_bar
    assert alpha_bar[0] > 0.999 and alpha_bar[-1] < 1e-3 and torch.all(alpha_bar[1:] < alpha_bar[:-1])


def test_add_noise_keeps_unit_variance_for_unit_data() -> None:
    schedule = NoiseSchedule()
    x0, noise = torch.randn(4096, 1, 1, 1), torch.randn(4096, 1, 1, 1)
    xt = schedule.add_noise(x0, torch.full((4096,), 500), noise)
    assert abs(xt.std().item() - 1.0) < 0.05


@pytest.mark.parametrize("args", [(0, 1e-4, 0.02), (10, 0.02, 1e-4)])
def test_schedule_rejects_bad_settings(args: tuple) -> None:
    with pytest.raises(ConfigError):
        NoiseSchedule(*args)


def test_ddim_with_the_true_noise_recovers_the_clean_image() -> None:
    """If the model predicts the exact noise, one DDIM pass returns x_0."""
    schedule, x0 = NoiseSchedule(), torch.rand(2, 3, 4, 4) * 2 - 1
    noise = torch.randn_like(x0)
    start = schedule.add_noise(x0, torch.full((2,), schedule.timesteps - 1), noise)

    def oracle(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = schedule.alpha_bar_at(t)
        return (x - a.sqrt() * x0.repeat(len(x) // len(x0), 1, 1, 1)) / (1 - a).sqrt()

    images, frames = DDIMSampler(schedule, steps=10, guidance=1.0).sample(oracle, start, torch.zeros(2), 0, 5)
    assert torch.allclose(images, x0, atol=1e-3) and len(frames) == 2


def test_guidance_combines_conditional_and_unconditional_predictions() -> None:
    def model(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y.float().view(-1, 1, 1, 1).expand_as(x)  # predicts its label

    sampler = DDIMSampler(NoiseSchedule(), steps=2, guidance=3.0)
    eps = sampler.guided_noise(model, torch.zeros(1, 1, 1, 1), torch.zeros(1, dtype=torch.long), torch.tensor([2]), 0)
    assert eps.item() == pytest.approx(6.0)  # 0 + 3 * (2 - 0)


def test_sampler_rejects_more_steps_than_timesteps() -> None:
    with pytest.raises(ConfigError):
        DDIMSampler(NoiseSchedule(timesteps=10), steps=11)


def test_ema_moves_toward_the_current_weights() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    ema = ExponentialMovingAverage(model, decay=0.9)
    torch.nn.init.ones_(model.weight)
    ema.update(model)
    assert ema.model.weight.item() == pytest.approx(0.1)


def test_feeder_yields_full_batches_in_range() -> None:
    feeder = GpuImageFeeder(torch.randint(0, 256, (10, 3, 8, 8), dtype=torch.uint8), torch.arange(10), 4, CPU)
    batches = list(feeder)
    assert len(batches) == 2 and all(x.shape == (4, 3, 8, 8) for x, _ in batches)
    assert all(x.min() >= -1 and x.max() <= 1 for x, _ in batches)


def _trainer(tmp_path: Path, epochs: int = 2, config: DiTConfig = TINY) -> DiffusionTrainer:
    images = torch.randint(0, 256, (16, 3, 8, 8), dtype=torch.uint8)
    settings = DiffusionSettings(epochs=epochs, batch_size=8, snapshot_every=1, sample_steps=2)
    trainer = DiffusionTrainer(settings, GpuImageFeeder(images, torch.arange(16) % 3, 8, CPU), tmp_path, CPU, config)
    return trainer


def test_trainer_writes_snapshots_and_resumes(tmp_path: Path) -> None:
    history = _trainer(tmp_path).fit()
    assert history.snapshots == [1, 2] and (tmp_path / SNAPSHOT_DIR / "epoch_0002.pt").exists()
    resumed = _trainer(tmp_path, epochs=3).fit()
    assert resumed.epochs == [1, 2, 3] and resumed.snapshots == [1, 2, 3]


def test_trainer_refuses_a_checkpoint_from_another_config(tmp_path: Path) -> None:
    _trainer(tmp_path, epochs=1).fit()
    with pytest.raises(CheckpointError):
        _trainer(tmp_path, config=DiTConfig(**{**TINY.__dict__, "depth": 1})).fit()


def test_figures_are_written(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    trainer.fit()
    figures = DiffusionFigures(trainer.history, SnapshotStore(tmp_path / SNAPSHOT_DIR), trainer.fixed.per_class, tmp_path)
    samples, frames = trainer.sample(keep_every=1)
    paths = [figures.final_grid(), figures.animation(), figures.progress(), figures.loss_curve(),
             figures.trajectory(frames, rows=2), figures.neighbors(samples[:4], samples[4:12])]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_image_grid_has_the_expected_size() -> None:
    grid = ImageGrid(upscale=2, pad=1).build(torch.zeros(6, 3, 4, 4), columns=3)
    assert grid.size == (3 * 9 + 1, 2 * 9 + 1)


def test_nearest_training_image_finds_the_exact_copy() -> None:
    train = torch.rand(5, 3, 4, 4)
    assert torch.equal(NearestTrainingImage(train).find(train[[3, 1]]), train[[3, 1]])
