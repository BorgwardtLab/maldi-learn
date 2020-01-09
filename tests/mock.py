"""Module for generating mock data of MALDI-TOF spectra."""
import numpy as np

from maldi_learn.data import MaldiTofSpectrum


def generate_mock_data(n_examples):
    """Generate random data with correct shape."""
    n_peaks = np.random.normal(1000, 100, size=n_examples).astype(int)
    print(n_peaks)
    return [
        MaldiTofSpectrum(
            np.random.uniform(0, 10000, size=(peaks, 2)))
        for peaks in n_peaks
    ]
