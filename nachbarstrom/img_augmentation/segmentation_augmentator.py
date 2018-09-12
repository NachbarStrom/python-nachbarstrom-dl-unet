from imgaug import augmenters as iaa
import numpy as np
from typing import Tuple

from typeguard import typechecked


class SegmentationAugmentor:
    def transform_image(self, image: np.ndarray, masks: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Transforms the input image into its augmented version.
        :param image: image of shape (dim1, dim2, channels)
        :param masks: masks with shape (dim1, dim2, num_masks)
        :return: tuple (transformed_image, transformed_masks)
        """
        raise NotImplementedError


class ImgAugAugmentor(SegmentationAugmentor):
    """Name comes from the library used: 'imgaug'. """

    _ROTATION_RANGE = -15, 15  # in degrees
    _TRANSLATE_PERCENT = {
        "x": (-0.025, 0.025),
        "y": (-0.025, 0.025),
    }

    @typechecked
    def __init__(self, random_seed: int = None):
        self._augmentor = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Crop(percent=(0, 0.1)),
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            iaa.ContrastNormalization((0.999, 1.001)),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent=self._TRANSLATE_PERCENT,
                rotate=self._ROTATION_RANGE,
                shear=(-8, 8),
                mode="symmetric",
            )
        ], random_order=True).to_deterministic()
        if random_seed:
            self._augmentor.reseed(random_seed, deterministic_too=True)

    @typechecked
    def transform_image(self, image: np.ndarray, masks: np.ndarray)-> \
            Tuple[np.ndarray, np.ndarray]:
        assert len(image.shape) == len(masks.shape) == 3, \
            f"Input shapes should have dim 3, but have dims " \
            f"{len(image.shape)} and {len(masks.shape)}."
        self._augmentor.reseed(deterministic_too=True)
        images_augmented = self._augmentor.augment_image(image)
        masks_augmented = self._augmentor.augment_image(masks)
        assert image.shape == images_augmented.shape
        assert masks.shape == masks_augmented.shape
        return images_augmented, masks_augmented
