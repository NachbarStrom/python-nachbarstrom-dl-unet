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
        original, suitable, unusable = next(self._fnames_gen)
        data = self._fname_to_3_channel_array(original)
        label_suitable = self._fname_to_binary_array(suitable)
        label_unusable = self._fname_to_binary_array(unusable)
        return data, np.concatenate((label_suitable, label_unusable), axis=2)

    @staticmethod
    def _chunk(seq: Sequence, size: int):
        """
        Returns a certain 'size' amount of elements from sequence 'seq' at once.
        """
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def _fname_to_3_channel_array(self, fname: str) -> np.ndarray:
        full_fname = os.path.join(self._basedir, fname)
        img = Image.open(full_fname).convert("RGB")
        array = np.array(img)
        assert len(array.shape) == 3  # shape: (_, _, 3)
        assert array.shape[2] == 3
        return array

    def _fname_to_binary_array(self, fname: str) -> np.ndarray:
        full_fname = os.path.join(self._basedir, fname)
        img = Image.open(full_fname).convert("1")  # black & white
        np_array = np.array(img).astype("float")
        return np.expand_dims(np_array, axis=2)  # make 1 channel

    def _process_labels(self, label):
        """No further processing needed"""
        return label


if __name__ == '__main__':
    basedir = "/home/tomas/Desktop/labelbox-download"
    provider = LocalImgDataProvider(basedir=basedir)
    X, y = provider(2)
