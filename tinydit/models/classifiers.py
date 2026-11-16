"""CIFAR-10 classifiers that score generated images: a ViT (the reward) and a ResNet-18 (the judge).

The ViT is the "patch 4, 2x epochs" model from the ViT-from-scratch post (93.1% test accuracy,
repository vit-cifar10-from-scratch, module vitcifar.models.vit). The layer names here match that
module, so its checkpoint loads without changes. The ResNet-18 is a different architecture, so
an image that fools the ViT does not automatically fool it.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

from tinydit.errors import ConfigError

INIT_STD = 0.02
RESNET_STEM_CHANNELS = 64


@dataclass(frozen=True)
class ViTConfig:
    """Shape of the classifier ViT, as stored in its checkpoint."""

    image_size: int
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    num_heads: int
    mlp_ratio: float = 4.0
    in_channels: int = 3
    dropout: float = 0.0
    drop_path_rate: float = 0.0  # training only; kept so the stored config loads

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size or self.dim % self.num_heads:
            raise ConfigError("image_size must split into patches and dim into heads")

    @property
    def num_patches(self) -> int:
        """Patch tokens, without the CLS token."""
        return (self.image_size // self.patch_size) ** 2


class _PatchEmbedding(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels, config.dim, config.patch_size, stride=config.patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(images).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class _Block(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        hidden = int(config.dim * config.mlp_ratio)
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = _Attention(config.dim, config.num_heads)
        self.norm2 = nn.LayerNorm(config.dim)
        self.mlp = nn.Sequential(nn.Linear(config.dim, hidden), nn.GELU(), nn.Dropout(config.dropout),
                                 nn.Linear(hidden, config.dim), nn.Dropout(config.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTClassifier(nn.Module):
    """ViT with a CLS token; inference only (no drop path)."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = _PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches + 1, config.dim))
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.depth))
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, num_classes) logits for normalized images."""
        tokens = self.patch_embed(images)
        x = torch.cat([self.cls_token.expand(len(tokens), -1, -1), tokens], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))


class CifarResNet18Builder:
    """torchvision ResNet-18 with a 3x3 stride-1 stem and no max-pool, for 32x32 images."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def build(self) -> nn.Module:
        """A randomly initialized model."""
        model = resnet18(weights=None, num_classes=self.num_classes)
        model.conv1 = nn.Conv2d(3, RESNET_STEM_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
