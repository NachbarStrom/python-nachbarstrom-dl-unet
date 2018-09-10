import os

from itertools import cycle

from typing import Sequence

import numpy as np
from PIL import Image

from tf_unet.image_util import BaseDataProvider


class LocalImgDataProvider(BaseDataProvider):
    channels = 3

    def __init__(self, a_min=0, a_max=255, basedir: str = None):
        super().__init__(a_min, a_max)
        assert os.path.isdir(basedir), f"{basedir} does not exist."

        self._basedir = basedir
        imgs_fnames = sorted(os.listdir(basedir))
        assert len(imgs_fnames) % 3 == 0, f"Number of imgs ({len(imgs_fnames)})" \
                                          f"is not a multiple of 3."
        self._fnames_gen = cycle(self._chunk(imgs_fnames, 3))

    def _next_data(self):
        original_fname, suitable_fname, _ = next(self._fnames_gen)
        original_img = self._fname_to_rgb_img_array(original_fname)
        suitable_img = self._fname_to_binary_img_array(suitable_fname)
        suitable_img_inverse = 1 - suitable_img
        masks = np.concatenate((suitable_img, suitable_img_inverse), axis=2)
        return original_img, masks

    @staticmethod
    def _chunk(seq: Sequence, size: int):
        """
        Returns a certain 'size' amount of elements from sequence 'seq' at once.
        """
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def _fname_to_rgb_img_array(self, fname: str) -> np.ndarray:
        full_fname = os.path.join(self._basedir, fname)
        img = Image.open(full_fname).convert("RGB")
        array = np.array(img)
        assert len(array.shape) == 3  # shape: (_, _, 3)
        assert array.shape[2] == 3
        return array

    def _fname_to_binary_img_array(self, fname: str) -> np.ndarray:
        full_fname = os.path.join(self._basedir, fname)
        img = Image.open(full_fname).convert("1")  # black & white
        np_array = np.array(img).astype("float")
        return np.expand_dims(np_array, axis=2)  # shape: (x_dim, y_dim, 1)

    def _process_labels(self, label):
        """No further processing needed"""
        return label
