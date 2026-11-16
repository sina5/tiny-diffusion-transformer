from pathlib import Path

import numpy as np
import pytest
import torch

from tinydit.errors import ResultsError
from tinydit.evaluation.fid import FeatureStats, InceptionFeatures
from tinydit.experiments.fid_sweep import CallCounter, FidResult, FidResultsFile


@pytest.fixture
def features() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(500, 4))


def test_identical_statistics_have_zero_distance(features: np.ndarray) -> None:
    stats = FeatureStats.from_features(features)
    assert stats.distance(stats) == pytest.approx(0.0, abs=1e-6)


def test_a_mean_shift_adds_its_squared_length(features: np.ndarray) -> None:
    """With equal covariances, FID is the squared distance between the means."""
    a = FeatureStats.from_features(features)
    b = FeatureStats(a.mean + np.array([3.0, 0, 0, 0]), a.cov)
    assert a.distance(b) == pytest.approx(9.0, abs=1e-4)


def test_stats_round_trip_through_a_file(tmp_path: Path, features: np.ndarray) -> None:
    stats = FeatureStats.from_features(features)
    stats.save(tmp_path / "ref.npz")
    loaded = FeatureStats.load(tmp_path / "ref.npz")
    assert np.array_equal(loaded.mean, stats.mean) and np.array_equal(loaded.cov, stats.cov)


def test_stats_need_two_images() -> None:
    with pytest.raises(ResultsError):
        FeatureStats.from_features(np.zeros((1, 4)))


def test_features_need_load_first() -> None:
    with pytest.raises(ResultsError):
        InceptionFeatures(torch.device("cpu")).extract(torch.zeros(1, 3, 8, 8))


def test_call_counter_counts_images_not_calls() -> None:
    counter = CallCounter(torch.nn.Identity())
    counter(torch.zeros(3, 1))
    counter(torch.zeros(5, 1))
    assert counter.images == 8


def _result(steps: int, fid: float) -> FidResult:
    return FidResult("flow_matching", "euler", steps, 2.0 * steps, fid, 100, 1.0)


def test_results_file_replaces_the_same_setting_and_sorts(tmp_path: Path) -> None:
    results = FidResultsFile(tmp_path / "fid.json")
    results.add(_result(50, 9.0))
    results.add(_result(10, 20.0))
    results.add(_result(50, 8.0))
    rows = results.load()
    assert [r["steps"] for r in rows] == [10, 50] and rows[1]["fid"] == 8.0
