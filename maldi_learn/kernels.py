'''
Kernels for assessing the similarity between MALDI-TOF spectra.
'''

from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.gaussian_process.kernels import StationaryKernelMixin
from sklearn.gaussian_process.kernels import Kernel

from sklearn.metrics import pairwise_distances

from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist

import numpy as np


class DiffusionKernel(StationaryKernelMixin, Kernel):
    '''
    Implements a diffusion kernel that performs iterative smoothing of
    a MALDI-TOF spectrum.
    '''

    def __init__(self, sigma=1.0):
        '''
        Initialises a new instance of the kernel.

        Parameters:
            sigma: Smoothing parameter
        '''

        self.sigma = sigma

    @property
    def hyperparameter_sigma(self):
        return Hyperparameter('sigma', 'numeric', self.sigma, 1)

    def __call__(self, X, Y=None, eval_gradient=False):
        '''
        Returns the kernel value k(X, Y) and, if desired, its gradient
        as well.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
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
        '''

        if Y is None:
            distances = pdist(X / (4 * self.sigma), metric='sqeuclidean')
            pass
        else:

            # Following the original API here, which prohibits gradient
            # evaluation for this case.
            if eval_gradient:
                raise ValueError(
                    'Gradient can only be evaluated when Y is None.')

            x_positions = np.array([x[0][0] for x in X]).reshape(-1, 1)
            y_positions = np.array([y[0][0] for y in Y]).reshape(-1, 1)

            distances = pairwise_distances(
                x_positions,
                y_positions,
                metric='sqeuclidean'
            )

            K = np.exp(distances / (8 * self.sigma))

        return K

        #X = np.atleast_2d(X)
        #length_scale = _check_length_scale(X, self.length_scale)
        #if Y is None:
        #    dists = pdist(X / length_scale, metric='sqeuclidean')
        #    K = np.exp(-.5 * dists)
        #    # convert from upper-triangular matrix to square matrix
        #    K = squareform(K)
        #    np.fill_diagonal(K, 1)
        #else:
        #    if eval_gradient:
        #        raise ValueError(
        #            "Gradient can only be evaluated when Y is None.")
        #    dists = cdist(X / length_scale, Y / length_scale,
        #                  metric='sqeuclidean')
        #    K = np.exp(-.5 * dists)

        #if eval_gradient:
        #    if self.hyperparameter_length_scale.fixed:
        #        # Hyperparameter l kept fixed
        #        return K, np.empty((X.shape[0], X.shape[0], 0))
        #    elif not self.anisotropic or length_scale.shape[0] == 1:
        #        K_gradient = \
        #            (K * squareform(dists))[:, :, np.newaxis]
        #        return K, K_gradient
        #    elif self.anisotropic:
        #        # We need to recompute the pairwise dimension-wise distances
        #        K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \
        #            / (length_scale ** 2)
        #        K_gradient *= K[..., np.newaxis]
        #        return K, K_gradient
        #else:
        #    return K

    def diag(self, X):
        '''
        Returns the diagonal of the kernel k(X, X). The result of this
        method is identical to np.diag(self(X)); however, it can be
        evaluated more efficiently since only the diagonal is evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        '''

        pass

    def __repr__(self):
        return f'{self.__class__.__name__}({self.sigma:.2f})'