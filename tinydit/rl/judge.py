"""Trains the ResNet-18 judge: a second CIFAR-10 classifier that the fine-tuning never sees.

If the reward ViT's accuracy on generated images rises but the judge's does not, the model has
learned to fool the ViT rather than to draw the class better.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from tinydit import constants as C


class JudgeTrainer:
    """SGD with Nesterov momentum and a cosine learning rate, with bf16 autocast on CUDA."""

    def __init__(self, model: nn.Module, device: torch.device, epochs: int = C.JUDGE_EPOCHS,
                 lr: float = C.JUDGE_LR) -> None:
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=C.JUDGE_MOMENTUM, nesterov=True,
                                         weight_decay=C.JUDGE_WEIGHT_DECAY)

    def fit(self, train: DataLoader, test: DataLoader) -> float:
        """Train for all epochs; return the final test accuracy."""
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs * len(train))
        accuracy = 0.0
        for epoch in range(1, self.epochs + 1):
            loss = self._epoch(train, scheduler)
            accuracy = self.evaluate(test)
            print(f"[judge] epoch {epoch:3d} loss={loss:.4f} test_acc={accuracy:.4f}", flush=True)
        return accuracy

    def _epoch(self, train: DataLoader, scheduler) -> float:
        self.model.train()
        total, seen = 0.0, 0
        for images, labels in train:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                loss = F.cross_entropy(self.model(images).float(), labels)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            scheduler.step()
            total, seen = total + loss.item() * len(labels), seen + len(labels)
        return total / max(1, seen)

    @torch.no_grad()
    def evaluate(self, test: DataLoader) -> float:
        """Test accuracy."""
        self.model.eval()
        correct, seen = 0, 0
        for images, labels in test:
            logits = self.model(images.to(self.device)).float()
            correct += int((logits.argmax(dim=1).cpu() == labels).sum())
            seen += len(labels)
        return correct / max(1, seen)

    def save(self, path: Path, accuracy: float) -> None:
        """Write the weights and the test accuracy."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "test_accuracy": accuracy}, path)
