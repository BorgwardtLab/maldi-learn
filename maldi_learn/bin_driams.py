"""Bin the DRIAMS data set and save it.

The purpose of this script is to perform binning of the DRIAMS data set
and store the resulting spectra as preprocessed files. This speeds up a
large number of downstream classification tasks.
"""

import argparse
import dotenv
import os

from driams import DRIAMSDatasetExplorer
from driams import DRIAMSLabelEncoder
from driams import load_driams_dataset

from maldi_learn.vectorization import BinningVectorizer

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-s', '--site',
        default='DRIAMS-A',
        type=str,
        help='Site to pre-process')

    parser.add_argument(
        '-y', '--years',
        default=['2015', '2016', '2017', '2018'],
        type=str,
        nargs='+',
        help='Years to pre-process'
    )

    parser.add_args(
        '-b', '--bins',
        type=int,
        required=True,
        help='Number of bins to use for binning transformation'
    )

    args = parser.parse_args()

    # Get all available antibiotics for the selected site. We will
    # pre-process *all* the spectra.
    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)
    antibiotics = explorer.available_antibiotics(args.site)

    driams_dataset = load_driams_dataset(
            explorer.root,
            args.site,
            args.years,
            '*',  # Load all species; we do *not* want to filter anything
            antibiotics,
            encoder=DRIAMSLabelEncoder(),
            handle_missing_resistance_measurements='remove_if_all_missing'
    )

    bv = BinningVectorizer(
            args.bins,
            min_bin=2000,
            max_bin=20000,
            n_jobs=-1  # Use all available cores to perform the processing
    )

    X = bv.fit_transform(driams_dataset.X)
