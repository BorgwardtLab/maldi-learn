"""Normalization strategies for MALDI-TOF spectra."""
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator


class TotalIonCurrentNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalize spectra based on total ion content. The normalizer
    supports different normalization strategies.
    """

    def __init__(self, ignore_zero_intensity=True, method='mean'):
        """Initialize total ion content based normalizer.

        Args:
            ignore_zero_intensity: Ignore peaks with zero intensity when
                computing the average used for normalization.

            method: Determines the method that is used to perform the
            normalization. If set to 'mean', computes averages over the
            spectra to normalize. If set to 'sum', normalizes each
            spectrum individually such that its intensities sum to one.
        """
        self.ignore_zero_intensity = ignore_zero_intensity
        self.mean_intensity = None
        self.method = method

    def _normalize_spectrum(self, spectrum, method):
        if method == 'mean':
            if self.ignore_zero_intensity:
                intensities = spectrum.intensities[spectrum.intensities != 0.]
            else:
                intensities = spectrum.intensities
            mean_instance_intensity = np.mean(intensities)
            scaling = mean_instance_intensity / self.mean_intensity
            return spectrum * np.array([1, scaling])[np.newaxis, :]
        elif method == 'sum':
            scaling = 1.0 / np.sum(spectrum.intensities)
            return spectrum * np.array([1, scaling])[np.newaxis, :]
        else:
            raise RuntimeError(
                    f'Unexpected normalization method "{method}"')

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
            self._normalize_spectrum(spectrum, method=self.method)
            for spectrum in X
        ]


class ScaleNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalizes a set of spectra such that their scales are not too
    small.
    """

    def _calculate_min_nonzero_intensity(self, spectra):
        intensities = np.concatenate(
            [
                s.intensities[s.intensities != 0] for s in spectra
            ],
            axis=0
        )
        return np.min(intensities)

    def _normalize_spectrum(self, spectrum):
        scaling = 1.0 / self.min_nonzero_intensity
        return spectrum * np.array([1, scaling])[np.newaxis, :]

    def fit(self, X, y=None):
        self.min_nonzero_intensity = self._calculate_min_nonzero_intensity(X)
        return self

    def transform(self, X):
        return [
            self._normalize_spectrum(spectrum) for spectrum in X
        ]
