"""Test BinningVectorizer."""
import unittest

import numpy as np

from maldi_learn.data import MaldiTofSpectrum
from maldi_learn.vectorization import BinningVectorizer


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
         [10.7,  5.0],
         [150.4, 10.],
         [1000,  3.0]
         ]
    )
]


class TestBinningVectorizer(unittest.TestCase):
    def test_simple_binning(self):
        vectorizer = BinningVectorizer(2, min_bin=-0.1, max_bin=999)
        vectorized = vectorizer.fit_transform(MOCK_DATA)
        self.assertEqual(vectorized.ndim, 2)
        self.assertEqual(vectorized.shape[0], len(MOCK_DATA))
        self.assertEqual(vectorized.shape[1], 2)

        self.assertTrue(np.all(vectorized[0] == np.array([23., 3.])))
        self.assertTrue(np.all(vectorized[1] == np.array([30., 3.])))




