"""Test normalizers."""
import unittest

import numpy as np

from maldi_learn.data import MaldiTofSpectrum
from maldi_learn.preprocessing import TotalIonCurrentNormalizer


MOCK_DATA = [
    MaldiTofSpectrum(
        [[0.0,   5.0],
         [10.7,  8.0],
         [150.4, 10.],
         [1000,  3.0]
         ]
    ),  # Mean intensity 6.5
    MaldiTofSpectrum(
        [[0.0,   15.0],
         [10.7,  0.0],
         [150.4, 10.],
         [1000,  3.0]
         ]
    ),  # Mean intensity 7 or 9.3333 (with ignore zero intensity)
]

# Total mean intensity: 6.75 or 7.7142857143 (with ignore zero intensity)


class TestTotalIonCurrentNormalizer(unittest.TestCase):
    def test_dont_ignore_zero_intensity(self):
        transf = TotalIonCurrentNormalizer(ignore_zero_intensity=False)
        transformed = transf.fit_transform(MOCK_DATA)

        # Normalization factor first example: 6.5 / 6.75 = 0.9629
        transformed_intesities = transformed[0].intensities
        expected_intensities = MOCK_DATA[0].intensities * (6.5 / 6.75)
        self.assertTrue(np.allclose(
            transformed_intesities,
            expected_intensities
        ))

        # Normalization factor second example: 7 / 6.75 = 1.0370
        transformed_intesities = transformed[1].intensities
        expected_intensities = MOCK_DATA[1].intensities * (7 / 6.75)
        self.assertTrue(np.allclose(
            transformed_intesities,
            expected_intensities
        ))

    def test_ignore_zero_intensity(self):
        transf = TotalIonCurrentNormalizer(ignore_zero_intensity=True)
        transformed = transf.fit_transform(MOCK_DATA)

        # Normalization factor first example: 6.5 / 7.71428 = 0.9629
        transformed_intesities = transformed[0].intensities
        expected_intensities = MOCK_DATA[0].intensities * (6.5 / 7.71428)
        self.assertTrue(np.allclose(
            transformed_intesities,
            expected_intensities
        ))

        # Normalization factor second example: 9.3333 / 7.71428 = 1.0370
        transformed_intesities = transformed[1].intensities
        expected_intensities = MOCK_DATA[1].intensities * (9.3333 / 7.71428)
        self.assertTrue(np.allclose(
            transformed_intesities,
            expected_intensities
        ))
