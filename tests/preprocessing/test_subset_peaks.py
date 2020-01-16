"""Test SubsetPeaksTransformer."""
import unittest

import numpy as np

from maldi_learn.preprocessing import SubsetPeaksTransformer
from maldi_learn.data import MaldiTofSpectrum


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


class TestSubsetPeakTransformer(unittest.TestCase):
    def test_transformer(self, n_peaks=2):
        transf = SubsetPeaksTransformer(n_peaks)
        transformed = transf.fit_transform(MOCK_DATA)
        print(transformed)
        # First example
        self.assertTrue(np.all(transformed[0][0] == np.array([10.7, 8.0])))
        self.assertTrue(np.all(transformed[0][1] == np.array([150.4, 10.0])))

        # Second example
        self.assertTrue(np.all(transformed[1][0] == np.array([0.0, 15.0])))
        self.assertTrue(np.all(transformed[1][1] == np.array([150.4, 10.0])))



