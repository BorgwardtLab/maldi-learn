"""Verify DRIAMS data set integrity.

The purpose of this script is to verify the integrity of the DRIAMS data
set. It is not expected that any of the checks performed by this script
should fail.
"""

import argparse
import os

import numpy as np

from maldi_learn.driams import load_spectrum

from tqdm import tqdm


def _has_no_nan(spectrum):
    # The spectrum, which is a data frame, must *not* contain any NaN
    # values.
    return not np.isnan(spectrum).any()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'ROOT',
        type=str,
        help='Root folder for the DRIAMS data set'
    )

    args = parser.parse_args()

    for root, dirs, files in os.walk(args.ROOT):
        filenames = [
            os.path.join(root, filename) for filename in files
        ]

        filenames = [
            fn for fn in filenames if os.path.splitext(fn)[1] == '.txt'
        ]

        if not filenames:
            continue

        for filename in tqdm(filenames, desc='Spectrum'):
            spectrum = load_spectrum(filename)

            code = os.path.basename(filename)
            code = os.path.splitext(code)[0]

            # TODO: update this once more checks are present
            if not _has_no_nan(spectrum):
                tqdm.write(code)
