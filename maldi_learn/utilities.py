"""Utilities functions for maldi-learn package.

It contains several functions used the main scripts.
"""

import itertools

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
    implementation='pandas',
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

    implementation : str
        Can be either `numpy` or `pandas` to indicate which implementation
        shall be used. Functionally, both of are equivalent. The `numpy`
        one is faster in case a lot of samples have to be thrown *away*,
        while the `pandas` one is faster in case of lot of samples have
        to be *kept*.

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

    if implementation == 'numpy':
        return _stratify_by_species_and_label_numpy(
                    y,
                    antibiotic,
                    test_size,
                    remove_invalid,
                    random_state
                )
    elif implementation == 'pandas':
        return _stratify_by_species_and_label_pandas(
                    y,
                    antibiotic,
                    test_size,
                    random_state
                )


def _stratify_by_species_and_label_numpy(
    y,
    antibiotic,
    test_size=0.2,
    remove_invalid=True,
    random_state=123
):
    # First, get the valid indices: valid indices are indices that
    # correspond to a finite label in the data. Since infinite, or
    # NaN values, cannot be handled, we have to remove them.
    labels = y[antibiotic].values
    labels = labels.astype('float')

    # These are the valid indices for the subsequent split. Set the
    # labels vector accordingly.
    valid_indices = np.nonzero(np.isfinite(labels))[0]

    # Valid labels that we will subsequently use. The additional
    # underscore is used because we might have to modify this vector
    # prior to using it for the split.
    labels_ = labels[valid_indices]

    # Encode species information to simplify the stratification. Every
    # combination of antibiotic and species will be encoded as an
    # individual class.
    species_encoded = LabelEncoder().fit_transform(y.species)

    # Again, only use *valid indices* here and employ an underscore
    # because the resulting vector might yet have to be modified.
    species_encoded_ = species_encoded[valid_indices]

    # Creates the *combined* label required for the stratification. The
    # first dimension of the vector is the encoded species, while the
    # second dimension is the (binary) label calculated from information
    # about resistance & susceptibility.
    stratify = np.vstack((species_encoded_, labels_)).T
    stratify = stratify.astype('int')

    if remove_invalid:
        unique, counts = np.unique(
            stratify,
            axis=0,
            return_counts=True
        )

        # Subset the valid indices by only keeping those elements that
        # occur at least twice. Repeat the subsetting from above, such
        # that the subsequent operations only use valid stratification
        # indices, and build a new stratification vector.

        unique = unique[counts >= 2]

        # Collect indices corresponding to all valid combinations. This
        # is relative to the stratification vector, though, so to get a
        # global lookup, we have to subset `valid_indices`.
        idx_ = [
            (stratify == u).all(axis=1).nonzero()[0].tolist() for u in unique
        ]

        idx_ = sorted(itertools.chain.from_iterable(idx_))
        valid_indices = valid_indices[idx_]

        labels_ = labels[valid_indices].astype('int')
        species_encoded_ = species_encoded[valid_indices].astype('int')
        stratify = np.vstack((species_encoded_, labels_)).T

    train_index, test_index = train_test_split(
        valid_indices,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )

    # Ensures that the reported indices can be easily used for subset
    # creation later on.
    train_index = np.asarray(train_index)
    test_index = np.asarray(test_index)

    return train_index, test_index


def _stratify_by_species_and_label_pandas(
    y,
    antibiotic,
    test_size,
    random_state,
):
    # Ensures that we always get an integer-based index for the data
    # frame, regardless of the existence of a code-based index. Also
    # create a simplified copy of the data frame to speed things up.
    df = y.reset_index()[['species', antibiotic]]

    # Slightly speeds up searching for duplicates when modifying the
    # data frame later on.
    df['species'] = LabelEncoder().fit_transform(df['species'])

    # Drop all NaN values in the current antibiotic column and change
    # the `dtype` to prepare for the subsequent stratification.
    df = df[df[antibiotic].notna()].astype(
        {
            'species': 'int',
            antibiotic: 'int',
        }
    )

    # Only keep duplicate rows, i.e. rows that occur multiple times.
    # These are the rows we can use for stratification below.
    df = df[df.duplicated(keep=False)]

    valid_indices = df.index  # Remaining indices that are valid for split
    stratify = df.values      # Joint class labels

    train_index, test_index = train_test_split(
        valid_indices,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )

    return np.asarray(train_index), np.asarray(test_index)
