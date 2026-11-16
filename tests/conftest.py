"""Shared fixtures. Tests run on CPU with tiny models and random tensors."""

import pytest
import torch


@pytest.fixture(autouse=True)
def seed() -> None:
    torch.manual_seed(0)
