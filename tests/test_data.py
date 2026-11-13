import pytest
import torch

from tinydit.data import Augmentation, BatchMixer, CifarTransforms, StratifiedSampler, TensorImageDataset
from tinydit.errors import ConfigError


def _image() -> torch.Tensor:
    return torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)


@pytest.mark.parametrize("augmentation", list(Augmentation))
def test_train_transform_keeps_shape_and_returns_float(augmentation: Augmentation) -> None:
    out = CifarTransforms().train(augmentation)(_image())
    assert out.shape == (3, 32, 32) and out.dtype == torch.float32


def test_no_augmentation_matches_test_transform() -> None:
    image, transforms = _image(), CifarTransforms()
    assert torch.equal(transforms.train(Augmentation.NONE)(image), transforms.test()(image))


def test_dataset_returns_transformed_image_and_int_label() -> None:
    dataset = TensorImageDataset(torch.zeros(2, 3, 4, 4, dtype=torch.uint8), torch.tensor([1, 0]), lambda x: x + 1)
    image, label = dataset[0]
    assert image.max() == 1 and label == 1 and len(dataset) == 2


def test_dataset_rejects_label_count_mismatch() -> None:
    with pytest.raises(ConfigError):
        TensorImageDataset(torch.zeros(2, 3, 4, 4), torch.tensor([1]), lambda x: x)


def test_stratified_sampler_takes_the_same_share_of_each_class() -> None:
    labels = torch.tensor([0] * 20 + [1] * 40)
    picks = StratifiedSampler(0.25, seed=1).indices(labels)
    assert torch.bincount(labels[picks]).tolist() == [5, 10]


def test_stratified_sampler_is_reproducible() -> None:
    labels = torch.arange(100) % 4
    assert torch.equal(StratifiedSampler(0.3, 7).indices(labels), StratifiedSampler(0.3, 7).indices(labels))


def test_stratified_sampler_keeps_everything_at_full_fraction() -> None:
    assert StratifiedSampler(1.0, 0).indices(torch.tensor([2, 0, 1])).tolist() == [0, 1, 2]


@pytest.mark.parametrize("fraction", [0.0, 1.2])
def test_stratified_sampler_rejects_out_of_range_fraction(fraction: float) -> None:
    with pytest.raises(ConfigError):
        StratifiedSampler(fraction, 0)


def test_batch_mixer_returns_soft_labels_that_sum_to_one() -> None:
    images, labels = torch.randn(6, 3, 8, 8), torch.tensor([0, 1, 2, 0, 1, 2])
    mixed, targets = BatchMixer(num_classes=3, mixup_alpha=0.8, cutmix_alpha=1.0)(images, labels)
    assert mixed.shape == images.shape and targets.shape == (6, 3)
    assert torch.allclose(targets.sum(dim=1), torch.ones(6))
