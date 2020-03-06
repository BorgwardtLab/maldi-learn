"""Verify DRIAMS data set integrity.

The purpose of this script is to verify the integrity of the DRIAMS data
set. It is not expected that any of the checks performed by this script
should fail.
"""

import argparse
import os

from driams import load_spectrum


def _has_no_nan(spectrum):
    # The spectrum, which is a data frame, must *not* contain any NaN
    # values.
    return not spectrum.isna().values.any()


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

        spectra = [
            load_spectrum(fn) for fn in filenames
        ]

        for spectrum, filename in zip(spectra, filenames):
            code = os.path.basename(filename)
            code = os.path.splitext(code)[0]

            # TODO: update this once more checks are present
            if not _has_no_nan(spectrum):
                print(code)
