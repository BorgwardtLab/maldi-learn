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
        combined_times = np.concatenate(
            [spectrum[:, 0] for spectrum in X], axis=0)
        min_range = min(self.min_bin, np.min(combined_times))
        max_range = max(self.max_bin, np.max(combined_times))

        _, self.bin_edges_ = np.histogram(
            combined_times, self.n_bins, range=(min_range, max_range))
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
        indices = np.digitize(times, self.bin_edges_, right=True)

        # Drops all instances which are outside the defined bin
        # range.
        valid = (indices >= 1) & (indices <= self.n_bins)
        spectrum = spectrum[valid]

        # Need to update indices to ensure that the first bin is at
        # position zero.
        indices = indices[valid] - 1
        identity = np.eye(self.n_bins)

        vec = np.sum(
            identity[indices] * spectrum[:, 1][:, np.newaxis], axis=0)

        return vec
