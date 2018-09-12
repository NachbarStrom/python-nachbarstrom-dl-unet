import pytest
from itertools import combinations

from nachbarstrom.img_augmentation.segmentation_augmentator import ImgAugAugmentor, SegmentationAugmentor
import numpy as np

IMG_DIM = 32
CHANNELS = 3
ONE_IMG = np.random.random((IMG_DIM, IMG_DIM, CHANNELS))


@pytest.fixture
def imgaug_augmentor():
    return ImgAugAugmentor()


def test_numpy_equality_works():
    assert np.array_equal(ONE_IMG, ONE_IMG)
    assert not np.array_equal(ONE_IMG, np.zeros(ONE_IMG.shape))


def test_augment_one_image_and_mask(imgaug_augmentor: SegmentationAugmentor):
    masks = ONE_IMG
    transformed_img, transformed_masks = \
        imgaug_augmentor.transform_image(image=ONE_IMG, masks=masks)
    assert not np.array_equal(ONE_IMG, transformed_img)
    assert not np.array_equal(masks, transformed_masks)


def test_augment_several_images_and_masks(imgaug_augmentor: SegmentationAugmentor):
    imgs = np.array((ONE_IMG, ONE_IMG, ONE_IMG))
    masks_list = imgs
    for img, masks in zip(imgs, masks_list):
        transf_img, transf_masks = \
            imgaug_augmentor.transform_image(image=img, masks=masks)
        assert not np.array_equal(transf_img, img)
        assert not np.array_equal(transf_masks, masks)


def test_augmented_imgs_are_different(imgaug_augmentor: SegmentationAugmentor):
    transf_imgs = []
    for _ in range(5):
        transf_img, _ = imgaug_augmentor.transform_image(image=ONE_IMG, masks=ONE_IMG)
        transf_imgs.append(transf_img)
    for img1, img2 in combinations(transf_imgs, 2):
        assert not np.array_equal(img1, img2)
