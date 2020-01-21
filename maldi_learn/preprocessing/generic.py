"""Generic preprocessing transformers for spectra."""
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator


class SubsetPeaksTransformer(BaseEstimator, TransformerMixin):
    """Transform to extract subset of peaks from spectrum."""

    def __init__(self, n_peaks=None):
        """Initialize transformer for subsetting peaks.

        Args:
            n_peaks: Number of peaks to extract from spectrum. If set to
            `None`, will just pass through input data without changing
            anything.
            on_less: Behaviour when one of the spectra has less than n_peaks
                peaks.

        """
        self.n_peaks = n_peaks

    def fit(self, X, y=None):
        """Fit transformer, does nothing."""
        return self

    def transform(self, X):
        """Get the n_peaks peaks with the highest intensity."""

        # Bail out early because there is nothing to do
        if self.n_peaks is None:
            return X

        output = []
        for spectrum in X:
            intensity = spectrum[:, 1]
            peak_indices = np.argsort(intensity, kind='stable')[::-1]
            # We want to sort back the indices to perserve the original order
            output.append(spectrum[sorted(peak_indices[:self.n_peaks])])
        return output
