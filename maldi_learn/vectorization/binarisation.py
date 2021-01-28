"""Binarises binned spectra."""

import joblib

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class VectorBinariser(BaseEstimator, TransformerMixin):
    """Binarisation of binned spectra vector according to cut-off."""

    _required_parameters = ['cut_off']

    def __init__(
        self,
        cut_off,
        n_jobs=None
    ):
        """Initialize BinningVectorizer.

        Args:
            cut_off: Intensity cut-off value to use for binarisation.
            n_jobs: If set, uses parallel processing with `n_jobs` jobs
        """
        self.cut_off = cut_off
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit transformer, does nothing."""
        return self

    def transform(self, X):
        """Binarise intensity values according to cut-off value.

        Args:
            2D numpy array with shape [n_instances x n_bins]

        Returns:
            2D numpy array with shape [n_instances x n_bins]

        """
        idx = X >= self.cut_off
        
        X[idx] = 1
        X[~idx] = 0
        return X
