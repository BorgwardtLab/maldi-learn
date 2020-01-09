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
        peaks.n_peaks = peaks.shape[0]
        return peaks
