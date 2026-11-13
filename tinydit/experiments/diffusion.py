"""Train a tiny class-conditional DiT on CIFAR-10 and record how its samples improve.

Every few epochs the trainer samples the same noise with the EMA weights and saves the grid,
so the snapshots can be compared directly. A checkpoint is saved at each snapshot, so a long
run can stop and resume.
"""

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from tinydit import constants as C
from tinydit.diffusion import DDIMSampler, ExponentialMovingAverage, NoiseSchedule
from tinydit.errors import CheckpointError
from tinydit.models.dit import DiT, DiTConfig

UINT8_MAX = 255.0
HISTORY_FILE = "history.json"
CHECKPOINT_FILE = "checkpoint.pt"
SNAPSHOT_DIR = "snapshots"


@dataclass(frozen=True)
class DiffusionSettings:
    """Everything that changes between a full run and a smoke run."""

    epochs: int = C.EPOCHS
    batch_size: int = C.BATCH_SIZE
    lr: float = C.LR
    snapshot_every: int = C.SNAPSHOT_EVERY
    sample_steps: int = C.SAMPLE_STEPS
    subset: int | None = None
    seed: int = C.DEFAULT_SEED

    def smoke(self) -> "DiffusionSettings":
        """A run that checks the pipeline in about a minute."""
        return replace(self, epochs=C.SMOKE_EPOCHS, subset=C.SMOKE_SUBSET, snapshot_every=1,
                       sample_steps=C.SMOKE_SAMPLE_STEPS)


class GpuImageFeeder:
    """Holds the training images on the device and yields shuffled, flipped batches in [-1, 1]."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int, device: torch.device) -> None:
        self.images = images.to(device)
        self.labels = labels.to(device)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.images) // self.batch_size

    def __iter__(self):
        order = torch.randperm(len(self.images), device=self.images.device)
        for i in range(len(self)):
            pick = order[i * self.batch_size:(i + 1) * self.batch_size]
            x = self.images[pick].float() / (UINT8_MAX / 2) - 1.0
            flip = torch.rand(len(pick), device=x.device) < 0.5
            yield torch.where(flip[:, None, None, None], x.flip(3), x), self.labels[pick]


@dataclass
class DiffusionHistory:
    """Per-epoch loss and timing, plus the epochs that have a snapshot."""

    epochs: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    snapshots: list[int] = field(default_factory=list)
    params: int = 0
    peak_vram_mib: float | None = None

    def save(self, path: Path) -> None:
        """Write as JSON."""
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DiffusionHistory":
        """Read from JSON, or start empty if the file is missing."""
        return cls(**json.loads(path.read_text(encoding="utf-8"))) if path.exists() else cls()


class FixedNoise:
    """The same starting noise and labels for every snapshot: one row per class."""

    def __init__(self, config: DiTConfig, per_class: int = C.SAMPLES_PER_CLASS, seed: int = C.SAMPLE_SEED):
        generator = torch.Generator().manual_seed(seed)
        count = config.num_classes * per_class
        self.noise = torch.randn(count, config.channels, config.image_size, config.image_size, generator=generator)
        self.labels = torch.arange(config.num_classes).repeat_interleave(per_class)
        self.per_class = per_class


class DiffusionTrainer:
    """Trains the DiT with the DDPM loss, keeps an EMA copy and writes snapshots."""

    def __init__(self, settings: DiffusionSettings, feeder: GpuImageFeeder, output_dir: Path, device: torch.device,
                 config: DiTConfig | None = None) -> None:
        self.settings = settings
        self.feeder = feeder
        self.output_dir = output_dir
        self.device = device
        self.config = config or DiTConfig()
        self.model = DiT(self.config).to(device)
        self.ema = ExponentialMovingAverage(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=settings.lr, weight_decay=C.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(1.0, (step + 1) / C.WARMUP_STEPS))
        self.schedule = NoiseSchedule()
        self.sampler = DDIMSampler(self.schedule, settings.sample_steps)
        self.fixed = FixedNoise(self.config)
        self.history = DiffusionHistory(params=sum(p.numel() for p in self.model.parameters()))

    @property
    def checkpoint_path(self) -> Path:
        """Where the resumable state lives."""
        return self.output_dir / CHECKPOINT_FILE

    def fit(self) -> DiffusionHistory:
        """Train to settings.epochs, resuming from the last checkpoint if there is one."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        start = self._resume()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(start + 1, self.settings.epochs + 1):
            began = time.perf_counter()
            loss = self._run_epoch()
            self._record(epoch, loss, time.perf_counter() - began)
            if epoch % self.settings.snapshot_every == 0 or epoch == self.settings.epochs:
                self._snapshot(epoch)
        return self.history

    def _run_epoch(self) -> float:
        self.model.train()
        total = torch.zeros((), device=self.device)
        for x0, labels in self.feeder:
            loss = self._loss(x0, labels)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), C.GRAD_CLIP_NORM)
            self.optimizer.step()
            self.scheduler.step()
            self.ema.update(self.model)
            total += loss.detach()
        return float(total / max(1, len(self.feeder)))

    def _loss(self, x0: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        t = self.schedule.sample_timesteps(len(x0), self.device)
        noise = torch.randn_like(x0)
        drop = torch.rand(len(labels), device=self.device) < C.LABEL_DROP
        labels = torch.where(drop, torch.full_like(labels, self.config.null_class), labels)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            predicted = self.model(self.schedule.add_noise(x0, t, noise), t, labels)
        return F.mse_loss(predicted.float(), noise)

    def _record(self, epoch: int, loss: float, seconds: float) -> None:
        self.history.epochs.append(epoch)
        self.history.losses.append(loss)
        self.history.seconds.append(seconds)
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / C.BYTES_PER_MIB
            self.history.peak_vram_mib = max(peak, self.history.peak_vram_mib or 0.0)
        print(f"[dit] epoch {epoch:4d} loss={loss:.4f} ({seconds:.1f}s)", flush=True)

    def sample(self, keep_every: int = 0) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sample the fixed noise with the EMA weights; images in [-1, 1] on the CPU."""
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            images, frames = self.sampler.sample(self.ema.model, self.fixed.noise.to(self.device),
                                                 self.fixed.labels.to(self.device), self.config.null_class, keep_every)
        return images.float().cpu(), frames

    def _snapshot(self, epoch: int) -> None:
        images, _ = self.sample()
        snapshot_dir = self.output_dir / SNAPSHOT_DIR
        snapshot_dir.mkdir(exist_ok=True)
        torch.save(images.half(), snapshot_dir / f"epoch_{epoch:04d}.pt")
        self.history.snapshots.append(epoch)
        self.history.save(self.output_dir / HISTORY_FILE)
        torch.save({"epoch": epoch, "model": self.model.state_dict(), "ema": self.ema.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                    "rng": torch.get_rng_state(), "config": asdict(self.config)}, self.checkpoint_path)

    def _resume(self) -> int:
        if not self.checkpoint_path.exists():
            return 0
        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if DiTConfig(**state["config"]) != self.config:
            raise CheckpointError(f"{self.checkpoint_path} was trained with a different DiT config")
        self.model.load_state_dict(state["model"])
        self.ema.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])
        self.history = DiffusionHistory.load(self.output_dir / HISTORY_FILE)  # saved with the checkpoint
        print(f"resumed from epoch {state['epoch']}", flush=True)
        return int(state["epoch"])

    def load_ema_only(self) -> int:
        """Load the EMA weights for figures without training; return the checkpoint epoch."""
        if not self.checkpoint_path.exists():
            raise CheckpointError(f"{self.checkpoint_path} is missing; train first")
        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.ema.load_state_dict(state["ema"])
        self.history = DiffusionHistory.load(self.output_dir / HISTORY_FILE)
        return int(state["epoch"])
