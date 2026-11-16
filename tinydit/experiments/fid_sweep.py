"""FID at several step counts for one trained method, with the real cost of each setting.

The cost is the number of function evaluations (NFE): DiT calls per image, counted by a
wrapper around the model. A guided step counts as two, because it runs the conditional and the
unconditional prediction. Results go to a JSON file that later runs append to.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from tinydit import constants as C
from tinydit.evaluation.fid import FeatureStats, FidEvaluator
from tinydit.experiments.diffusion import DiffusionTrainer

FID_RESULTS_FILE = "fid.json"


class CallCounter(nn.Module):
    """Wraps a model and counts how many images pass through it."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.images = 0

    def forward(self, x: torch.Tensor, *args: torch.Tensor) -> torch.Tensor:
        self.images += len(x)
        return self.model(x, *args)


@dataclass(frozen=True)
class FidResult:
    """One row of the sweep."""

    method: str
    sampler: str
    steps: int
    nfe: float
    fid: float
    samples: int
    seconds: float


class FidSweep:
    """Generates class-balanced samples from fixed noise and scores each step count."""

    def __init__(self, trainer: DiffusionTrainer, evaluator: FidEvaluator, reference: FeatureStats,
                 samples: int = C.FID_SAMPLES, batch: int = C.FID_BATCH, seed: int = C.FID_SEED) -> None:
        self.trainer = trainer
        self.evaluator = evaluator
        self.reference = reference
        self.samples = samples
        self.batch = batch
        self.seed = seed

    def run(self, steps: int, sampler: str) -> FidResult:
        """FID of `samples` images generated with `steps` steps."""
        counter = CallCounter(self.trainer.ema.model)
        model, self.trainer.ema.model = self.trainer.ema.model, counter
        began = time.perf_counter()
        try:
            images = self._generate(steps)
        finally:
            self.trainer.ema.model = model
        seconds = time.perf_counter() - began
        fid = self.evaluator.score(images, self.reference)
        return FidResult(self.trainer.method.kind.value, sampler, steps, counter.images / self.samples, fid,
                         self.samples, seconds)

    def _generate(self, steps: int) -> torch.Tensor:
        config = self.trainer.config
        generator = torch.Generator().manual_seed(self.seed)
        labels = torch.arange(self.samples) % config.num_classes
        chunks = []
        for start in range(0, self.samples, self.batch):
            count = min(self.batch, self.samples - start)
            noise = torch.randn(count, config.channels, config.image_size, config.image_size, generator=generator)
            chunks.append(self.trainer.generate(noise, labels[start:start + count], steps)[0])
        return torch.cat(chunks)


class FidResultsFile:
    """The JSON list of FidResult rows for one run; a new row replaces an old one with the same setting."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[dict]:
        """All rows, or an empty list."""
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def add(self, result: FidResult) -> None:
        """Insert or replace the row for (sampler, steps, samples)."""
        key = (result.sampler, result.steps, result.samples)
        rows = [r for r in self.load() if (r["sampler"], r["steps"], r["samples"]) != key]
        rows.append(asdict(result))
        rows.sort(key=lambda r: (r["sampler"], r["steps"]))
        self.path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
