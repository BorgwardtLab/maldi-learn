"""Transformers for binning spectra."""

import joblib

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class BinningVectorizer(BaseEstimator, TransformerMixin):
    """Vectorizer based on binning MALDI-TOF spectra.

    Attributes:
        bin_edges_: Edges of the bins derived after fitting the transformer.

    """

    _required_parameters = ['n_bins']

    def __init__(
        self,
        n_bins,
        min_bin=float('inf'),
        max_bin=float('-inf'),
        n_jobs=None
    ):
        """Initialize BinningVectorizer.

        Args:
            n_bins: Number of bins to bin the inputs spectra into.
            min_bin: Smallest possible bin edge.
            max_bin: Largest possible bin edge.
            n_jobs: If set, uses parallel processing with `n_jobs` jobs
        """
        self.n_bins = n_bins
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.bin_edges_ = None
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit transformer, derives bins used to bin spectra."""
        # Find the smallest and largest time values in the dataset
        # It should be that the first/last time value is the smallest/biggest
        # but we call min/max to be safe.
        min_range = min(spectrum[:, 0].min() for spectrum in X)
        min_range = min(min_range, self.min_bin)
        max_range = max(spectrum[:, 0].max() for spectrum in X)
        max_range = max(max_range, self.max_bin)
        self.bin_edges_ = np.linspace(min_range, max_range, self.n_bins + 1)
        return self

    def transform(self, X):
        """Transform list of spectra into vector using bins.

        Args:
            X: List of MALDI-TOF spectra

        Returns:
            2D numpy array with shape [n_instances x n_bins]

        """
        if self.n_jobs is None:
            output = [self._transform(spectrum) for spectrum in X]
        else:
            output = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(self._transform)(s) for s in X
            )

        return np.stack(output, axis=0)

    def _transform(self, spectrum):
        times = spectrum[:, 0]

        valid = (times > self.bin_edges_[0]) & (times <= self.bin_edges_[-1])
        spectrum = spectrum[valid]

        vec = np.histogram(spectrum[:, 0], bins=self.bin_edges_, weights=spectrum[:, 1])[0]

        return vec
