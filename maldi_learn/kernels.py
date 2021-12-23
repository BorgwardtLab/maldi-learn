"""Kernels for assessing the similarity between MALDI-TOF spectra."""

from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.gaussian_process.kernels import StationaryKernelMixin
from sklearn.gaussian_process.kernels import Kernel

from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels

import numpy as np
import sys


class PIKE(StationaryKernelMixin, Kernel):
    """Peak Information Kernel (PIKE).

    Implements a diffusion kernel that performs iterative smoothing of
    a MALDI-TOF spectrum.
    """

    def __init__(self,
                 sigma=1.0,
                 sigma_bounds=(1e-5, 1e5),
                 hotpatch_sklearn=True):
        """Initialise a new instance of the kernel.

        Parameters
        ----------
        sigma : float
            Smoothing parameter. Higher values imply that the distance
            between peaks is progressively being adjusted.

        sigma_bounds : tuple of floats
            Tuple specifying the minimum and maximum bound of the sigma
            scale parameter.

        hotpatch_sklearn : bool
            If set, ensures that `scikit-learn` is patched to ensure
            that the kernel can be applied. This involves *removing*
            a few consistency checks, so it should be treated with a
            little bit of care.
        """
        self.sigma = sigma
        self.sigma_bounds = sigma_bounds
        self.hotpatch_sklearn = hotpatch_sklearn

        if self.hotpatch_sklearn:
            def passthrough(*args, **kwargs):
                return args

            # TODO: this is not the best way of handling the calculation,
            # but `pairwise_distances` is making this so much easier.
            module = sys.modules['sklearn.metrics.pairwise']
            module.check_pairwise_arrays = passthrough

            sys.modules['sklearn.metrics.pairwise'] = module

            # Disable some additional validation checks. This is
            # required because the classifier behaves in various
            # ways when being subjected to inputs of *different*
            # lengths versus *same-length* inputs.
            module = sys.modules['sklearn.utils.validation']

            # We now also disable the `_assert_all_finite` routine
            # because it currently does not handle arrays of dtype
            # object correctly.
            module._assert_all_finite = passthrough

            sys.modules['sklearn.utils.validation'] = module

    @property
    def hyperparameter_sigma(self):
        """Return current value of smoothing parameter."""
        return Hyperparameter('sigma', 'numeric', self.sigma_bounds)

    @property
    def requires_vector_input(self):
        """Describe kernel properties.

        Returns
        -------
        False to indicate that the kernel does not use
        feature fixed-length features.
        """
        return False

    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate kernel to return its value and, optionally, a gradient.

        Returns the kernel value k(X, Y) and, if desired, its gradient
        as well. This is the main evaluation function. Its design uses
        the same API as in the Gaussian Process module of `sklearn`.

        Parameters
        ----------
        X : array of spectra
            Left argument of the returned kernel k(X, Y)

        Y : array of spectra
            Right argument of the returned kernel k(X, Y). If None, k(X,
            X), i.e. the diagonal, is evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        def evaluate_kernel(x, y):

            # Get the positions (masses) of the two spectra. This could
            # be rewritten more compactly following the new interface.
            #
            # TODO: simplify / refactor
            x_positions = np.array(x[:, 0]).reshape(-1, 1)
            y_positions = np.array(y[:, 0]).reshape(-1, 1)

            distances = pairwise_distances(
                x_positions,
                y_positions,
                metric='sqeuclidean'
            )

            # Calculate scale factors as the outer product of the peak
            # heights of the input data.
            x_peaks = np.array(x[:, 1])
            y_peaks = np.array(y[:, 1])

            P = np.outer(x_peaks, y_peaks)
            K = np.multiply(P, np.exp(-distances / (4 * self.sigma)))

            return np.sum(K) / (4 * self.sigma * np.pi)

        def evaluate_gradient(x, y):

            # TODO: simplify / refactor
            x_positions = np.array(x[:, 0]).reshape(-1, 1)
            y_positions = np.array(y[:, 0]).reshape(-1, 1)

            distances = pairwise_distances(
                x_positions,
                y_positions,
                metric='sqeuclidean'
            )

            # Calculate scale factors as the outer product of the peak
            # heights of the input data.
            x_peaks = np.array(x[:, 1])
            y_peaks = np.array(y[:, 1])

            P = np.outer(x_peaks, y_peaks)
            K = np.multiply(P, np.exp(-distances / (4 * self.sigma)))

            # Thanks to the simple form of the kernel, the gradient only
            # requires an additional multiplication, followed by scaling
            # it appropriately.
            K_gradient = np.multiply(K, (distances - 4 * self.sigma))

            # Sum over all pairwise kernel values to get the full
            # gradient between the two entries.
            return np.sum(K_gradient) / (8 * self.sigma**2)

        if Y is None:
            if eval_gradient:
                K = pairwise_kernels(X, metric=evaluate_kernel)
                K_gradient = pairwise_kernels(X, metric=evaluate_gradient)

                return K, K_gradient[:, :, np.newaxis]

            else:
                return pairwise_kernels(X, metric=evaluate_kernel)
        else:

            # Following the original API here, which prohibits gradient
            # evaluation for this case.
            if eval_gradient:
                raise ValueError(
                    'Gradient can only be evaluated when Y is None.')

            return pairwise_kernels(X, Y, metric=evaluate_kernel)

    def diag(self, X):
        """Return diagonal value of the kernel.

        Returns the diagonal of the kernel k(X, X). The result of this
        method is identical to calling `np.diag(self(X))`; however, it
        can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        diag_values = np.zeros(len(X))

        for i, x in enumerate(X):
            x_positions = np.array(x[:, 0]).reshape(-1, 1)

            distances = pairwise_distances(
                x_positions,
                x_positions,
                metric='sqeuclidean'
            )

            x_peaks = np.array(x[:, 1])

            P = np.outer(x_peaks, x_peaks)
            K = np.multiply(P, np.exp(-distances / (4 * self.sigma)))

            # Diagonal value for $x_i$
            diag_values[i] = np.sum(K)

        return diag_values / (4 * self.sigma * np.pi)

    def __repr__(self):
        """Return string representation of kernel."""
        return f'{self.__class__.__name__}({self.sigma:.8f})'


# Kept in order to be compatible with older versions of the library.
# This ensures that code does not have to be changed.
DiffusionKernel = PIKE
