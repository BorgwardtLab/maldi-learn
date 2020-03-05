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

    random_state:
        Specifies the random state to use for the split.

    Returns
    -------
    Tuple of train and test indices.
    """
    _check_y(y)
    n_samples = y.shape[0]

    # Encode species information to simplify the stratification. Every
    # combination of antibiotic and species will be encoded as an
    # individual class.
    le = LabelEncoder()

    species_transform = le.fit_transform(y.species)
    labels = y[antibiotic].values

    # Creates the *combined* label required for the stratification.
    stratify = np.vstack((species_transform, labels)).T

    train_index, test_index = train_test_split(
        range(n_samples),
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )

    return train_index, test_index
