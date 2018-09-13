import os

import pytest
from itertools import combinations
import numpy as np

from nachbarstrom import LocalImgDataProvider
from nachbarstrom.img_augmentation import SegmentationAugmentor
from .test_img_augmentation import imgaug_augmentor


@pytest.fixture
def local_img_provider(imgaug_augmentor: SegmentationAugmentor):
    basedir = "imgs"
    full_basedir = os.path.join(os.path.dirname(__file__), basedir)
    return LocalImgDataProvider(basedir=full_basedir, augmentor=imgaug_augmentor)


def test_provider_returns_y_with_even_masks(local_img_provider: LocalImgDataProvider):
    _, y = local_img_provider(1)
    _, _, _, masks_num = y.shape

    def is_even(x):
        return x % 2 == 0

    assert is_even(masks_num)


def test_provider_y_sum_to_one(local_img_provider: LocalImgDataProvider):
    _, y = local_img_provider(1)
    mask_1 = y[0, ..., 0]
    mask_2 = y[0, ..., 1]

    def sum_to_one(x, y) -> bool:
        return (x + y == 1).all()

    assert sum_to_one(mask_1, mask_2)


def test_provider_augments_images(local_img_provider: LocalImgDataProvider):
    X, y = local_img_provider(4)
    for img1, img2 in combinations(X, 2):
        assert not np.array_equal(img1, img2)

    for masks1, masks2 in combinations(y, 2):
        assert not np.array_equal(masks1, masks2)


def test_provider_has_channels(local_img_provider: LocalImgDataProvider):
    assert local_img_provider.channels == 3


def test_provider_has_n_class(local_img_provider: LocalImgDataProvider):
    assert local_img_provider.n_class == 2
