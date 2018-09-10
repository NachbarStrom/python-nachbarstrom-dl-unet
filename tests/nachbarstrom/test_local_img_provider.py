import os

import pytest

from nachbarstrom import LocalImgDataProvider


@pytest.fixture
def local_img_provider():
    basedir = "imgs"
    full_basedir = os.path.join(os.path.dirname(__file__), basedir)
    return LocalImgDataProvider(basedir=full_basedir)


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
