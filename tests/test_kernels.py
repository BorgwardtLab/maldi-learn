"""Tests for the kernel functions."""

import unittest

from maldi_learn.data import MaldiTofSpectrum
from maldi_learn.kernels import DiffusionKernel

from sklearn.gaussian_process import GaussianProcessClassifier

import numpy as np


MOCK_DATA = [
    MaldiTofSpectrum(
        [[0.0,   5.0],
         [10.7,  8.0],
         [150.4, 10.],
         [1000,  3.0]
         ]
    ),
    MaldiTofSpectrum(
        [[0.0,   15.0],
         [10.7,  0.0],
         [150.4, 10.],
         [1000,  3.0]
         ]
    ),
    MaldiTofSpectrum(
        [[333,  47.0],
         [666,  23.0],
         [888,  42.],
         [999,  5.0]
         ]
    ),
]

MOCK_LABELS = [0, 0, 1]


class TestPIKE(unittest.TestCase):
    def test_fit(self):
        clf = GaussianProcessClassifier(kernel=DiffusionKernel())

        clf.fit(MOCK_DATA, MOCK_LABELS)

        # These are just a 'smoke test' in the sense that we want to
        # figure out whether these operations can be performed.
        clf.predict(MOCK_DATA)
        clf.predict_proba(MOCK_DATA)

        self.assertAlmostEqual(
            clf.kernel_.sigma,
            2.1993352,
            places=6
        )
