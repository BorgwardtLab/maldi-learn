#!/usr/bin/env python3
#
# Compares the AMR profiles between two sites and prints out the
# differences. This assumes that the same spectra have been measured for
# both sites.

import argparse
import dotenv
import os

import numpy as np
import pandas as pd

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-l', '--left',
            default='DRIAMS-E',
            help='First site to compare'
    )
    parser.add_argument(
            '-r', '--right',
            default='DRIAMS-F',
            help='Second site to compare'
    )

    args = parser.parse_args()

    # TODO: make years configurable
    filename_left = os.path.join(
        DRIAMS_ROOT, args.left, 'id', '2019', f'2019_clean.csv'
    )

    metadata_left = pd.read_csv(
                filename_left,
                low_memory=False,
                na_values=['-'],        # additional way to encode `Nan`
                keep_default_na=True,   # keep default `NaN` encodings
    )

    # TODO: make years configurable
    filename_right = os.path.join(
        DRIAMS_ROOT, args.right, 'id', '2019', f'2019_clean.csv'
    )

    metadata_right = pd.read_csv(
                filename_right,
                low_memory=False,
                na_values=['-'],        # additional way to encode `Nan`
                keep_default_na=True,   # keep default `NaN` encodings
    )

    metadata_left = metadata_left.sort_values(by=['id'])
    metadata_right = metadata_right.sort_values(by=['id'])

    # Check that the same spectra have been measured here and the same
    # species have been assigned.
    assert (metadata_left.id == metadata_right.id).all()
    assert (metadata_left.species == metadata_right.species).all()

    antibiotics_left = [
        c for c in metadata_left.columns if c[0].isupper()
    ]

    antibiotics_right = [
        c for c in metadata_right.columns if c[0].isupper()
    ]

    # Ensures that the assay that was employed is the same.
    assert antibiotics_left == antibiotics_right

    antibiotics = antibiotics_left
    n_differences = 0

    for (i1, row1), (i2, row2) in zip(metadata_left.iterrows(),
                                      metadata_right.iterrows()):

        print(f'Processing ID = {row1.id}...')

        row1 = row1[antibiotics]
        row2 = row2[antibiotics]

        detected_difference = False

        for a, left, right in zip(antibiotics, row1.values, row2.values):
            # Since 'NaN' != 'NaN' in all cases, let's handle this case
            # separately; we are not really interested in it.
            if type(left) is float and type(right) is float:
                if np.isnan(left) and np.isnan(right):
                    continue

            if left != right:
                print(f'\t{a}: left = {left}, right = {right}')
                detected_difference = True

        n_differences += detected_difference

    print(f'{n_differences}/{len(metadata_left)} rows differ')
