"""Tests for topological preprocessing routines."""
import unittest

from maldi_learn.preprocessing import TopologicalPeakFiltering

from tests.mock import generate_mock_data


class TestToplogicalPreprocessing(unittest.TestCase):
    def test_correct_n_peaks(self, n_examples=10, n_peaks=100):
        mock_data = generate_mock_data(n_examples)
        transformer = TopologicalPeakFiltering(n_peaks=n_peaks)
        transformed_data = transformer.fit_transform(mock_data)
        print(transformed_data[0].shape)
        self.assertTrue(
            all([spectrum.n_peaks == n_peaks for spectrum in transformed_data])
        )
