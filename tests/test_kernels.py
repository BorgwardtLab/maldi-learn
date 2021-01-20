"""Tests for the kernel functions."""

import unittest

from maldi_learn.data import MaldiTofSpectrum
from maldi_learn.kernels import DiffusionKernel

from sklearn.gaussian_process import GaussianProcessClassifier


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

MOCK_DATA_DIFFERENT_LENGTHS = [
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
        [[0.0, 0.0],
         [333,  47.0],
         [666,  23.0],
         [888,  42.],
         [999,  5.0]
         ]
    ),
]

MOCK_LABELS = [0, 0, 1]


class TestPIKE(unittest.TestCase):

    def _clf_smoke_test(self, X, y, expected_sigma):
        clf = GaussianProcessClassifier(kernel=DiffusionKernel())
        clf.fit(X, y)

        # These are just a 'smoke test' in the sense that we want to
        # figure out whether these operations can be performed.
        clf.predict(X)
        clf.predict_proba(X)

        self.assertAlmostEqual(
            clf.kernel_.sigma,
            expected_sigma,
            places=6
        )

    def test_equal_lengths(self):
        self._clf_smoke_test(
            MOCK_DATA,
            MOCK_LABELS,
            2.1993352
        )

    def test_different_lengths(self):
        self._clf_smoke_test(
            MOCK_DATA_DIFFERENT_LENGTHS,
            MOCK_LABELS,
            2.1993352
        )
