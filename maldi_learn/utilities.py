"""Utilities functions for maldi-learn package.

It contains several functions used the main scripts.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def _check_y(y):
    """Check whether a given label data frame is valid."""
    assert type(y) is pd.DataFrame

    assert 'species' in y.columns
    assert 'code' in y.columns

    # TODO: check whether other checks are required to make this a valid
    # data frame.


def stratify_by_species_and_label(
    y,
    antibiotic,
    test_size=0.2,
    remove_invalid=True,
    random_state=123
):
    """Stratification by species and antibiotic label.

    This function performs a stratified train--test split, taking into
    account species *and* label information.

    Parameters
    ----------
    y : pandas.DataFrame
        Label data frame containing information about the species, the
        antibiotics, and other (optional) information, which is ignored
        by this function.

    antibiotic : str
        Specifies the antibiotic for the stratification. This must be
        a valid column in `y`.

    test_size: float
        Specifies the size of the test data set returned by the split.
        This function cannot guarantee that a specific test size will
        lead to a valid split. In this case, it will fail.

    remove_invalid : bool
        If set, removes invalid species--antibiotic combinations from
        the reported indices. A combination is invalid if the number
        of samples is insufficient. Such combinations cannot be used
        in stratified train--test split anyway.

    random_state:
        Specifies the random state to use for the split.

    Returns
    -------
    Tuple of train and test indices, each one of them being an
    `np.ndarray`. If `remove_invalid` has been set, the totality
    of all train and test indices does not necessarily add up to
    the whole data set.
    """
    _check_y(y)
    n_samples = y.shape[0]

    # Encode species information to simplify the stratification. Every
    # combination of antibiotic and species will be encoded as an
    # individual class.
    le = LabelEncoder()

    species_transform = le.fit_transform(y.species)
    labels = y[antibiotic].values

    # Creates the *combined* label required for the stratification. The
    # first dimension of the vector is the encoded species, while the
    # second dimension is the (binary) label calculated from information
    # about resistance & susceptibility.
    stratify = np.vstack((species_transform, labels)).T

    if remove_invalid:
        _, indices, counts = np.unique(
            stratify,
            axis=0,
            return_index=True,
            return_counts=True
        )

        # Get indices of all elements that appear an insufficient number of
        # times to be used in the stratification.
        invalid_indices = indices[counts < 2]

        # Replace all of them by a 'fake' class whose numbers are guaranteed
        # *not* to occur in the data set (because labels are encoded from 0,
        # and the binary label is either 0 or 1).
        stratify[invalid_indices, :] = [-1, -1]

    train_index, test_index = train_test_split(
        range(n_samples),
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )

    # Ensures that the reported indices can be easily used for subset
    # creation later on.
    train_index = np.asarray(train_index)
    test_index = np.asarray(test_index)

    # Remove all indices of the virtual class afterwards. Thus, the
    # reported train and test indices do not correspond to the whole
    # data set necessarily.
    if remove_invalid:
        train_index = train_index[
                        np.isin(train_index,
                                invalid_indices,
                                assume_unique=True,
                                invert=True)
                    ]

        test_index = test_index[
                        np.isin(test_index,
                                invalid_indices,
                                assume_unique=True,
                                invert=True)
                   ]

    return train_index, test_index
