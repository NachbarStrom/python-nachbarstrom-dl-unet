import pytest
from itertools import combinations

from typing import Sequence

from nachbarstrom.img_augmentation import ImgAugAugmentor, SegmentationAugmentor
import numpy as np

IMG_DIM = 32
CHANNELS = 3
ONE_IMG = np.random.random((IMG_DIM, IMG_DIM, CHANNELS))
ONE_MASK = np.random.binomial(1, 0.5, (IMG_DIM, IMG_DIM, 1)).astype(float)


@pytest.fixture
def imgaug_augmentor():
    """Low 'elastic_transf_sigma' to reduce compute duration."""
    return ImgAugAugmentor(random_seed=0, elastic_transf_sigma=1)


def test_numpy_equality_works():
    assert np.array_equal(ONE_IMG, ONE_IMG)
    assert not np.array_equal(ONE_IMG, np.zeros(ONE_IMG.shape))


def test_augment_one_image_and_mask(imgaug_augmentor: SegmentationAugmentor):
    transformed_img, transformed_masks = \
        imgaug_augmentor.transform_image(image=ONE_IMG, masks=ONE_MASK)
    assert not np.array_equal(ONE_IMG, transformed_img)
    assert not np.array_equal(ONE_MASK, transformed_masks)


def test_augment_several_images_and_masks(imgaug_augmentor: SegmentationAugmentor):
    imgs = np.array((ONE_IMG, ONE_IMG, ONE_IMG))
    masks_list = np.array((ONE_MASK, ONE_MASK, ONE_MASK))
    for img, masks in zip(imgs, masks_list):
        transf_img, transf_masks = \
            imgaug_augmentor.transform_image(image=img, masks=masks)
        assert not np.array_equal(transf_img, img)
        assert not np.array_equal(transf_masks, masks)


def each_array_is_different(list_of_arrays: Sequence[np.ndarray]) -> bool:
    """Auxiliary function"""
    for array1, array2 in combinations(list_of_arrays, 2):
        if np.array_equal(array1, array2):
            return False
    return True


def test_augmented_imgs_and_masks_are_different(imgaug_augmentor: SegmentationAugmentor):
    transf_imgs = []
    transf_masks_list = []
    for _ in range(4):
        transf_img, transf_masks = \
            imgaug_augmentor.transform_image(image=ONE_IMG, masks=ONE_IMG)
        transf_imgs.append(transf_img)
        transf_masks_list.append(transf_masks)

    assert each_array_is_different(transf_imgs)
    assert each_array_is_different(transf_masks_list)


def test_augmented_masks_are_binary(imgaug_augmentor: SegmentationAugmentor):
    for _ in range(4):
        _, masks = imgaug_augmentor.transform_image(image=ONE_IMG, masks=ONE_MASK)
        assert np.array_equal(masks, masks.astype(bool))
