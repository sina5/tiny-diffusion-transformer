"""A small class-conditional Diffusion Transformer (DiT, Peebles and Xie, 2023).

The network is a ViT that takes a noisy image, the timestep and a class label, and predicts
the noise that was added. Timestep and class condition every block through adaptive layer
norm with zero-initialized gates ("adaLN-Zero"), so each block starts as the identity.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from tinydit import constants as C
from tinydit.errors import ConfigError

ADALN_BLOCK_CHUNKS = 6  # shift, scale, gate for attention and for the MLP
ADALN_FINAL_CHUNKS = 2  # shift, scale before the output projection
LAYER_NORM_EPS = 1e-6


@dataclass(frozen=True)
class DiTConfig:
    """Shape of the DiT. The extra class index num_classes is the "no class" token."""

    image_size: int = C.CIFAR10_IMAGE_SIZE
    channels: int = 3
    patch_size: int = C.PATCH_SIZE
    dim: int = C.DIM
    depth: int = C.DEPTH
    heads: int = C.HEADS
    mlp_ratio: float = C.MLP_RATIO
    num_classes: int = C.CIFAR10_NUM_CLASSES
    time_freq_dim: int = C.TIME_FREQ_DIM

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size:
            raise ConfigError("image_size must be a multiple of patch_size")
        if self.dim % self.heads:
            raise ConfigError("dim must be a multiple of heads")
        if self.time_freq_dim % 2:
            raise ConfigError("time_freq_dim must be even")

    @property
    def grid_size(self) -> int:
        """Patches per side."""
        return self.image_size // self.patch_size

    @property
    def null_class(self) -> int:
        """Label index that means "no class" (for classifier-free guidance)."""
        return self.num_classes


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply a per-sample affine change to (B, T, D) tokens."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal features of the timestep, then a two-layer MLP."""

    def __init__(self, freq_dim: int, dim: int, max_period: float = C.MAX_PERIOD) -> None:
        super().__init__()
        half = freq_dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(max_period) * torch.arange(half) / half), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.float()[:, None] * self.freqs[None]
        return self.mlp(torch.cat([angles.cos(), angles.sin()], dim=1))


class DiTBlock(nn.Module):
    """Pre-norm transformer block whose norms and residual gates depend on the condition."""

    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(config.dim, elementwise_affine=False, eps=LAYER_NORM_EPS)
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.proj = nn.Linear(config.dim, config.dim)
        self.norm2 = nn.LayerNorm(config.dim, elementwise_affine=False, eps=LAYER_NORM_EPS)
        hidden = int(config.dim * config.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(config.dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, config.dim))
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(config.dim, ADALN_BLOCK_CHUNKS * config.dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(cond).chunk(ADALN_BLOCK_CHUNKS, dim=1)
        x = x + gate1.unsqueeze(1) * self._attention(modulate(self.norm1(x), shift1, scale1))
        return x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).view(batch, tokens, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class DiT(nn.Module):
    """Predicts the noise in x_t, given the timestep t and the class label y."""

    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(config.channels, config.dim, config.patch_size, stride=config.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.grid_size ** 2, config.dim))
        self.time_embed = TimestepEmbedding(config.time_freq_dim, config.dim)
        self.class_embed = nn.Embedding(config.num_classes + 1, config.dim)  # + the "no class" token
        self.blocks = nn.ModuleList(DiTBlock(config) for _ in range(config.depth))
        self.final_norm = nn.LayerNorm(config.dim, elementwise_affine=False, eps=LAYER_NORM_EPS)
        self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(config.dim, ADALN_FINAL_CHUNKS * config.dim))
        self.out = nn.Linear(config.dim, config.patch_size ** 2 * config.channels)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        for block in self.blocks:  # adaLN-Zero: every block starts as the identity
            nn.init.zeros_(block.adaln[-1].weight)
            nn.init.zeros_(block.adaln[-1].bias)
        for layer in (self.final_adaln[-1], self.out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        cond = self.time_embed(t) + self.class_embed(y)
        for block in self.blocks:
            tokens = block(tokens, cond)
        shift, scale = self.final_adaln(cond).chunk(ADALN_FINAL_CHUNKS, dim=1)
        return self._unpatchify(self.out(modulate(self.final_norm(tokens), shift, scale)))

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        c, p, g = self.config.channels, self.config.patch_size, self.config.grid_size
        patches = tokens.view(tokens.shape[0], g, g, p, p, c)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(tokens.shape[0], c, g * p, g * p)
