"""Frechet Inception Distance (FID, Heusel et al., 2017) between generated images and CIFAR-10.

The feature extractor is the FID Inception network from pytorch-fid (the same weights as the
original TensorFlow FID), so the numbers can be compared with other pytorch-fid results. The
reference statistics come from the 50,000 training images and are cached after the first run.
The project computes FID from 10,000 samples (FID-10k), which reads higher than FID-50k.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from scipy import linalg

from tinydit import constants as C
from tinydit.errors import ResultsError

UINT8_MAX = 255.0
MIN_IMAGES = 2  # a covariance needs at least two images
SQRTM_EPS = 1e-6
IMAG_TOLERANCE = 1e-3


@dataclass(frozen=True)
class FeatureStats:
    """Mean and covariance of Inception features."""

    mean: np.ndarray
    cov: np.ndarray

    @classmethod
    def from_features(cls, features: np.ndarray) -> "FeatureStats":
        """Raises ResultsError for fewer than two feature rows."""
        if len(features) < MIN_IMAGES:
            raise ResultsError("FID needs at least two images")
        return cls(features.mean(axis=0), np.cov(features, rowvar=False))

    def save(self, path: Path) -> None:
        """Write as .npz."""
        np.savez(path, mean=self.mean, cov=self.cov)

    @classmethod
    def load(cls, path: Path) -> "FeatureStats":
        """Read from .npz."""
        with np.load(path) as data:
            return cls(data["mean"], data["cov"])

    def distance(self, other: "FeatureStats") -> float:
        """The Frechet distance |mu1 - mu2|^2 + Tr(S1 + S2 - 2 (S1 S2)^(1/2)) between the two Gaussians.

        Written out here because pytorch-fid 0.3.0 passes an argument that SciPy 1.18 removed.
        """
        covmean = linalg.sqrtm(self.cov @ other.cov)
        if not np.isfinite(covmean).all():  # singular product: nudge both covariances, as pytorch-fid does
            offset = np.eye(len(self.cov)) * SQRTM_EPS
            covmean = linalg.sqrtm((self.cov + offset) @ (other.cov + offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=IMAG_TOLERANCE):
                raise ResultsError("matrix square root has a large imaginary part")
            covmean = covmean.real
        diff = self.mean - other.mean
        return float(diff @ diff + np.trace(self.cov) + np.trace(other.cov) - 2 * np.trace(covmean))


class InceptionFeatures:
    """Pool3 features of images in [-1, 1]. Builds the network (and downloads its weights) on load()."""

    def __init__(self, device: torch.device, batch_size: int = C.FID_BATCH, dims: int = C.FID_DIMS) -> None:
        self.device = device
        self.batch_size = batch_size
        self.dims = dims
        self.network: InceptionV3 | None = None

    def load(self) -> "InceptionFeatures":
        """Build the network in eval mode."""
        self.network = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]]).to(self.device).eval()
        return self

    @torch.inference_mode()
    def extract(self, images: torch.Tensor) -> np.ndarray:
        """(N, dims) float64 features. Raises ResultsError if load() was not called."""
        if self.network is None:
            raise ResultsError("call load() before extract()")
        rows = []
        for batch in images.split(self.batch_size):
            unit = (batch.float().to(self.device).clamp(-1, 1) + 1) / 2  # the network wants [0, 1]
            rows.append(self.network(unit)[0].flatten(1).double().cpu())
        return torch.cat(rows).numpy()


class FidEvaluator:
    """Scores generated images against cached CIFAR-10 training statistics."""

    def __init__(self, features: InceptionFeatures, reference_path: Path) -> None:
        self.features = features
        self.reference_path = reference_path

    def reference(self, train_images_uint8: torch.Tensor) -> FeatureStats:
        """Training-set statistics, computed once and then read from reference_path."""
        if self.reference_path.exists():
            return FeatureStats.load(self.reference_path)
        scaled = train_images_uint8.float() / (UINT8_MAX / 2) - 1.0
        stats = FeatureStats.from_features(self.features.extract(scaled))
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        stats.save(self.reference_path)
        return stats

    def score(self, samples: torch.Tensor, reference: FeatureStats) -> float:
        """FID of samples in [-1, 1] against the reference statistics."""
        return FeatureStats.from_features(self.features.extract(samples)).distance(reference)
