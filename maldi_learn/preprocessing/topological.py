"""Preprocessing using TOpological Peak Filtering (TOPF)."""
from sklearn.base import TransformerMixin
from topf import PersistenceTransformer
from typing import List

from ..data import MaldiTofSpectrum


class TopologicalPeakFiltering(TransformerMixin):
    """Topological peak filtering using TOPF."""

    _required_arguments = ['n_peaks']

    def __init__(self, n_peaks):
        """Topological peak filtrering (TOPF) for MALDI-TOF spectra.

        Args:
            n_peaks: Number of peaks to retain. Peaks will be eliminated in
                top-down order starting from the one with the lowest
                persistence. Thus, if the var is 1, only the highest peak will
                be kept.

        """
        self.n_peaks = n_peaks

    def fit(self, X, y=None):
        """Do nothing."""
        return self

    @staticmethod
    def _remove_non_peaks(spectrum):
        return spectrum[spectrum[:, 1] != 0.]

    def transform(self, X: List[MaldiTofSpectrum]) -> List[MaldiTofSpectrum]:
        """Apply topological peak filtering to the data array X.

        Args:
            X: List of MALDI-TOF spectra.

        Returns:
            Sparse spectra containing only n_peaks peaks.

        """
        pers_transformer = PersistenceTransformer(
            calculate_persistence_diagram=False, n_peaks=self.n_peaks)

        return [
            MaldiTofSpectrum(
                self._remove_non_peaks(
                    pers_transformer.fit_transform(spectrum)
                )
            )
            for spectrum in X
        ]
