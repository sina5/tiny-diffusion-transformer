"""Tests for the blog figure helpers: result readers, grid helpers and that charts write PNGs."""

import json
from pathlib import Path

import pytest
import torch

from tinydit.errors import ResultsError
from tinydit.experiments.blog_figures import (
    ComparisonNoise, DdrlRun, FidChart, FidFile, FidSeries, GridPanel, SampleGridFigure, Style, TrainingHistory,
    first_per_class,
)


@pytest.fixture
def fid_file(tmp_path: Path) -> FidFile:
    rows = [{"method": "m", "sampler": "euler", "steps": s, "nfe": 2.0 * s, "fid": 100.0 / s, "samples": 10,
             "seconds": 1.0} for s in (4, 1, 2)]
    rows.append({"method": "m", "sampler": "heun", "steps": 1, "nfe": 2.0, "fid": 50.0, "samples": 10, "seconds": 1.0})
    path = tmp_path / "fid.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return FidFile(path)


def test_fid_rows_are_sorted_by_nfe_and_filtered_by_sampler(fid_file: FidFile) -> None:
    assert [r.steps for r in fid_file.rows("euler")] == [1, 2, 4]


def test_fid_row_finds_one_setting(fid_file: FidFile) -> None:
    assert fid_file.row("heun", 1).fid == 50.0


@pytest.mark.parametrize("sampler, steps", [("euler", 8), ("ddim", 1)])
def test_fid_row_raises_for_a_missing_setting(fid_file: FidFile, sampler: str, steps: int) -> None:
    with pytest.raises(ResultsError):
        fid_file.row(sampler, steps)


def test_missing_results_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ResultsError):
        DdrlRun.load(tmp_path / "nothing")


def test_training_history_counts_nan_epochs(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text('{"epochs": [1, 2, 3], "losses": [1.0, NaN, null]}', encoding="utf-8")
    assert TrainingHistory.load(path).non_finite_epochs == 2


def test_first_per_class_keeps_the_first_images_of_each_class() -> None:
    images = torch.arange(6).float().view(6, 1, 1, 1)  # two classes, three each
    assert first_per_class(images, 3, 2).flatten().tolist() == [0.0, 1.0, 3.0, 4.0]


def test_first_per_class_rejects_more_than_stored() -> None:
    with pytest.raises(ResultsError):
        first_per_class(torch.zeros(4, 1, 1, 1), 2, 3)


def test_comparison_noise_is_repeatable_and_class_major() -> None:
    noise_a, labels = ComparisonNoise(per_class=2, seed=1).make()
    noise_b, _ = ComparisonNoise(per_class=2, seed=1).make()
    assert torch.equal(noise_a, noise_b)
    assert labels[:4].tolist() == [0, 0, 1, 1]


def test_fid_chart_writes_a_png(fid_file: FidFile, tmp_path: Path) -> None:
    path = FidChart("t", "s").draw([FidSeries(Style("a", "#2a78d6"), fid_file.rows("euler"))], tmp_path / "c.png")
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_sample_grid_writes_a_png(tmp_path: Path) -> None:
    panel = GridPanel("p", torch.zeros(20, 3, 32, 32))
    path = SampleGridFigure("t", "s", per_class=2).draw([panel], tmp_path / "g.png")
    assert path.exists()
