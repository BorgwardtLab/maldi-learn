"""Utilities functions for maldi-learn package.

It contains several functions used the main scripts.
"""

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def stratify_by_species_and_label(y, antibiotic, test_size=0.2, random_state=123):
    
    n_samples = y.shape[0]
    # construct class and species vector for stratification
    le = LabelEncoder()

    # TODO change to species eventually
    species_transform = le.fit_transform(y.species)
    labels = y[antibiotic].values
    
    stratify = np.vstack((species_transform, labels)).T
    index_train, index_test = train_test_split(range(n_samples),
            test_size=test_size, stratify=stratify, random_state=random_state)
    return index_train, index_test 
