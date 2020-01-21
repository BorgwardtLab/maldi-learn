"""Normalization strategies for MALDI-TOF spectra."""
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator


class TotalIonCurrentNormalizer(BaseEstimator, TransformerMixin):
    """Normalize spectra based on total ion content."""

    def __init__(self, ignore_zero_intensity=True):
        """Initialize total ion content based normalizer.

        Args:
            ignore_zero_intensity: Ignore peaks with zero intensity when
                computing the average used for normalization.

        """
        self.ignore_zero_intensity = ignore_zero_intensity
        self.mean_intensity = None

    def _normalize_spectrum(self, spectrum):
        if self.ignore_zero_intensity:
            intensities = spectrum.intensities[spectrum.intensities != 0.]
        else:
            intensities = spectrum.intensities
        mean_instance_intensity = np.mean(intensities)
        scaling = mean_instance_intensity / self.mean_intensity
        return spectrum * np.array([1, scaling])[np.newaxis, :]

    def _compute_mean_intensity_spectra(self, spectra):
        if self.ignore_zero_intensity:
            intensities = np.concatenate(
                [
                    spectrum.intensities[spectrum.intensities != 0.]
                    for spectrum in spectra
                ],
                axis=0
            )
        else:
            intensities = np.concatenate(
                [spectrum.intensities for spectrum in spectra], axis=0)
        return np.mean(intensities)

    def fit(self, X, y=None):
        """Fit transformer, computes average statistics of spectra."""
        self.mean_intensity = self._compute_mean_intensity_spectra(X)
        return self

    def transform(self, X):
        """Normalize spectra using total ion content."""
        return [
            self._normalize_spectrum(spectrum)
            for spectrum in X
        ]
