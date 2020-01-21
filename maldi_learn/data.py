"""Classes to standardize handling of Spectra."""

import numpy as np


class MaldiTofSpectrum(np.ndarray):
    """Numpy NDArray subclass representing a MALDI-TOF Spectrum."""

    def __new__(cls, peaks):
        """Create a MaldiTofSpectrum.

        Args:
            peaks: 2d array or list of tuples or list of list containing pairs
                of mass/charge to intensity.

        Raises:
            ValueError: If the input data is not in the correct format.

        """
        peaks = np.asarray(peaks).view(cls)
        if peaks.ndim != 2 or peaks.shape[1] != 2:
            raise ValueError(
                f'Input shape of {peaks.shape} does not match expected shape '
                'for spectrum [n_peaks, 2].'
            )
        return peaks

    @property
    def n_peaks(self):
        """Get number of peaks of the spectrum."""
        return self.shape[0]

    @property
    def intensities(self):
        """Get the intensities of the spectrum."""
        return self[:, 1]

    @property
    def mass_to_charge_ratios(self):
        """Get mass-t0-charge ratios of spectrum."""
        return self[:, 0]
