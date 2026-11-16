from pathlib import Path

import pytest
import torch

from tinydit import constants as C
from tinydit.errors import CheckpointError, ConfigError
from tinydit.experiments.diffusion import DiffusionSettings, DiffusionTrainer, GpuImageFeeder, method_run_dir
from tinydit.methods import Ddpm, FlowMatching, MeanFlow, MethodKind, OdeSolver, Shortcut, build_method
from tinydit.methods.base import guided, time_grid
from tinydit.models.dit import DiT, DiTConfig

CPU = torch.device("cpu")
TINY = DiTConfig(image_size=8, patch_size=2, dim=16, depth=2, heads=2, num_classes=3, time_freq_dim=8)
TINY_INTERVAL = DiTConfig(**{**TINY.__dict__, "interval_input": True})


@pytest.fixture
def x0() -> torch.Tensor:
    return torch.rand(4, 3, 4, 4) * 2 - 1


def straight_path_oracle(x0: torch.Tensor, time_scale: float = C.TIME_SCALE):
    """The exact velocity (x_t - x_0) / t of the linear path; on that path it is also the average velocity."""
    def model(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        return (x - x0) / (t / time_scale).view(-1, 1, 1, 1)
    return model


# --- shared helpers -------------------------------------------------------------


def test_time_grid_runs_from_noise_to_data() -> None:
    grid = time_grid(4, CPU)
    assert len(grid) == 5 and grid[0].item() == 1.0 and grid[-1].item() == 0.0


def test_guidance_passes_the_interval_to_both_halves() -> None:
    def model(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return (y.float() + h).view(-1, 1, 1, 1).expand_as(x)

    out = guided(model, torch.zeros(1, 1, 1, 1), torch.zeros(1), torch.tensor([2]), 0, 3.0, h=torch.tensor([1.0]))
    assert out.item() == pytest.approx(1.0 + 3.0 * 2.0)  # uncond (0 + 1) + 3 * ((2 + 1) - (0 + 1))


@pytest.mark.parametrize("kind", list(MethodKind))
def test_build_method_returns_that_kind(kind: MethodKind) -> None:
    assert build_method(kind).kind is kind


def test_run_dir_keeps_ddpm_at_the_top_level(tmp_path: Path) -> None:
    assert method_run_dir(MethodKind.DDPM, tmp_path) == tmp_path
    assert method_run_dir(MethodKind.MEANFLOW, tmp_path) == tmp_path / "meanflow"


# --- DiT interval input ---------------------------------------------------------


def test_dit_with_interval_input_keeps_the_image_shape() -> None:
    x = torch.randn(2, 3, 8, 8)
    assert DiT(TINY_INTERVAL)(x, torch.tensor([10.0, 500.0]), torch.tensor([0, 3]), torch.tensor([0.0, 250.0])).shape \
        == x.shape


@pytest.mark.parametrize(("config", "h"), [(TINY, torch.zeros(1)), (TINY_INTERVAL, None)])
def test_dit_rejects_a_missing_or_unexpected_interval(config: DiTConfig, h: torch.Tensor | None) -> None:
    with pytest.raises(ConfigError):
        DiT(config)(torch.randn(1, 3, 8, 8), torch.zeros(1), torch.zeros(1, dtype=torch.long), h)


# --- flow matching ------------------------------------------------------------


@pytest.mark.parametrize("solver", list(OdeSolver))
def test_flow_matching_with_the_true_velocity_recovers_the_image(x0: torch.Tensor, solver: OdeSolver) -> None:
    """The linear path is straight, so even one Euler step is exact for the true velocity."""
    method = FlowMatching(guidance=1.0, solver=solver)
    images, frames = method.sample(straight_path_oracle(x0), torch.randn_like(x0), torch.zeros(4), 0, 3, 1)
    assert torch.allclose(images, x0, atol=1e-5) and len(frames) == 3


def test_flow_matching_loss_is_zero_for_the_true_velocity(x0: torch.Tensor) -> None:
    assert FlowMatching().loss(straight_path_oracle(x0), x0, torch.zeros(4), 0).item() < 1e-8


def test_heun_uses_two_calls_per_step_except_the_last(x0: torch.Tensor) -> None:
    calls = []

    def model(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return torch.zeros_like(x)

    FlowMatching(guidance=1.0, solver=OdeSolver.HEUN).sample(model, x0, torch.zeros(4), 0, 4)
    assert len(calls) == 2 * 4 - 1


@pytest.mark.parametrize("method", [FlowMatching(), MeanFlow(), Shortcut()])
def test_samplers_reject_zero_steps(method, x0: torch.Tensor) -> None:
    with pytest.raises(ConfigError):
        method.sample(straight_path_oracle(x0), x0, torch.zeros(4), 0, 0)


# --- MeanFlow -----------------------------------------------------------------


def test_meanflow_times_are_ordered_with_the_expected_share_equal() -> None:
    r, t = MeanFlow(unequal_share=0.25).sample_times(20_000, CPU)
    assert torch.all(r <= t) and torch.all((t > 0) & (t < 1))
    assert (r == t).float().mean().item() == pytest.approx(0.75, abs=0.02)


@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_meanflow_time_derivative_matches_finite_differences(scale: float) -> None:
    """du/dt along the path (z moves with v, t moves with 1, r is fixed), for a smooth toy u."""
    def model(z: torch.Tensor, t: torch.Tensor, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return z.pow(2) * torch.sin(t / scale).view(-1, 1, 1, 1) + (h / scale).view(-1, 1, 1, 1) * z

    z, v = torch.randn(3, 1, 2, 2, dtype=torch.float64), torch.randn(3, 1, 2, 2, dtype=torch.float64)
    r, t = torch.tensor([0.1, 0.5, 0.2], dtype=torch.float64), torch.tensor([0.6, 0.5, 0.9], dtype=torch.float64)
    labels, eps = torch.zeros(3, dtype=torch.long), 1e-6
    exact = MeanFlow(time_scale=scale).time_derivative(model, z, r, t, v, labels)
    numeric = (model(z + eps * v, (t + eps) * scale, labels, (t + eps - r) * scale)
               - model(z - eps * v, (t - eps) * scale, labels, (t - eps - r) * scale)) / (2 * eps)
    assert torch.allclose(exact[[0, 2]], numeric[[0, 2]].float(), atol=1e-4)
    assert torch.count_nonzero(exact[1]) == 0  # r = t: the derivative is not needed


def test_meanflow_one_step_with_the_true_average_velocity_recovers_the_image(x0: torch.Tensor) -> None:
    oracle = straight_path_oracle(x0, C.MEANFLOW_TIME_SCALE)
    images, _ = MeanFlow().sample(oracle, torch.randn_like(x0), torch.zeros(4), 0, 1)
    assert torch.allclose(images, x0, atol=1e-5)


def test_meanflow_loss_is_zero_for_the_true_average_velocity(x0: torch.Tensor) -> None:
    """On a straight path u = v everywhere, so du/dt = 0 and the target is v."""
    oracle = straight_path_oracle(x0, C.MEANFLOW_TIME_SCALE)
    loss = MeanFlow(guidance=1.0).loss(oracle, x0, torch.zeros(4, dtype=torch.long), 0)
    assert loss.item() < 1e-8


def test_meanflow_guided_target_keeps_v_for_dropped_labels() -> None:
    def model(z: torch.Tensor, t: torch.Tensor, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.full_like(z, 10.0)  # the unconditional velocity

    v = torch.ones(2, 1, 1, 1)
    target = MeanFlow(guidance=3.0)._guided_target_velocity(model, v, torch.ones(2), v, torch.tensor([1, 0]), 0)
    assert target[0].item() == pytest.approx(3.0 * 1 + (1 - 3.0) * 10) and target[1].item() == 1.0


def test_meanflow_rejects_a_bad_share() -> None:
    with pytest.raises(ConfigError):
        MeanFlow(unequal_share=1.5)


# --- shortcut models --------------------------------------------------------------


def test_shortcut_step_sizes_are_powers_of_two_down_to_the_finest() -> None:
    d = Shortcut(base_steps=8).step_sizes(1000, CPU)
    assert set(d.tolist()) == {1 / 2, 1 / 4, 1 / 8}


def test_shortcut_bootstrap_targets_stay_on_the_grid(x0: torch.Tensor) -> None:
    x_t, t, two_d, target = Shortcut()._bootstrap_targets(straight_path_oracle(x0), x0, torch.randn_like(x0),
                                                          torch.zeros(4))
    slots = t / two_d
    assert torch.all(t - two_d >= 0) and torch.all(t <= 1) and torch.allclose(slots, slots.round())


def test_shortcut_bootstrap_target_is_the_velocity_on_a_straight_path(x0: torch.Tensor) -> None:
    noise = torch.randn_like(x0)
    *_, target = Shortcut()._bootstrap_targets(straight_path_oracle(x0), x0, noise, torch.zeros(4))
    assert torch.allclose(target, noise - x0, atol=1e-4)


def test_shortcut_one_step_with_the_true_velocity_recovers_the_image(x0: torch.Tensor) -> None:
    images, _ = Shortcut(guidance=1.0).sample(straight_path_oracle(x0), torch.randn_like(x0), torch.zeros(4), 0, 1)
    assert torch.allclose(images, x0, atol=1e-5)


@pytest.mark.parametrize("kwargs", [{"bootstrap_share": 0.0}, {"base_steps": 6}, {"base_steps": 1}])
def test_shortcut_rejects_bad_settings(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        Shortcut(**kwargs)


# --- trainer with each method -----------------------------------------------------


def _trainer(tmp_path: Path, method, config: DiTConfig | None = None, epochs: int = 1) -> DiffusionTrainer:
    images = torch.randint(0, 256, (16, 3, 8, 8), dtype=torch.uint8)
    settings = DiffusionSettings(epochs=epochs, batch_size=8, snapshot_every=1, sample_steps=2)
    feeder = GpuImageFeeder(images, torch.arange(16) % 3, 8, CPU)
    return DiffusionTrainer(settings, feeder, tmp_path, CPU, config, method)


@pytest.mark.parametrize("method", [Ddpm(), FlowMatching(), MeanFlow(), Shortcut()])
def test_trainer_runs_each_method(tmp_path: Path, method) -> None:
    config = TINY_INTERVAL if method.needs_interval else TINY
    history = _trainer(tmp_path, method, config).fit()
    assert history.snapshots == [1] and all(torch.isfinite(torch.tensor(history.losses)))


def test_trainer_rejects_a_config_without_the_interval_input(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        _trainer(tmp_path, MeanFlow(), TINY)


def test_trainer_refuses_a_checkpoint_from_another_method(tmp_path: Path) -> None:
    _trainer(tmp_path, FlowMatching(), TINY).fit()
    with pytest.raises(CheckpointError):
        _trainer(tmp_path, Ddpm(), TINY, epochs=2).fit()
