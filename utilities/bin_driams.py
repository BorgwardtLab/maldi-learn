"""Bin the DRIAMS data set and save it.

The purpose of this script is to perform binning of the DRIAMS data set
and store the resulting spectra as preprocessed files. This speeds up a
large number of downstream classification tasks.
"""

import argparse
import dotenv
import os

import pandas as pd

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder
from maldi_learn.driams import load_driams_dataset

from maldi_learn.vectorization import BinningVectorizer

from tqdm import tqdm

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

    parser.add_argument(
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

    # Process each year separately, because that simplifies assigning
    # the output files.
    for year in tqdm(args.years, desc='Year'):
        driams_dataset = load_driams_dataset(
                explorer.root,
                args.site,
                year,
                '*',  # Load all species; we do *not* want to filter anything
                antibiotics[year],
                handle_missing_resistance_measurements='keep',  # Keep all
        )

        # Follows the same hierarchy as the other data sets. For
        # example, if site DRIAMS-A is being pre-processed, each
        # file will be stored in
        #
        #       $ROOT/DRIAMS-A/binned_$BINS/$YEAR
        #
        # for $BINS bins in the histogram. This makes re-loading
        # pre-processed spectra ridiculously easy.
        output_directory = os.path.join(
            explorer.root,
            args.site,
            f'binned_{args.bins}',
            year
        )

        os.makedirs(output_directory, exist_ok=True)

        bv = BinningVectorizer(
                args.bins,
                min_bin=2000,
                max_bin=20000,
                n_jobs=-1  # Use all available cores to perform the processing
        )

        codes = driams_dataset.y['code'].values

        for spectrum, code in tqdm(zip(driams_dataset.X, codes),
                                   total=len(codes),
                                   desc='Spectrum'):
            output_file = os.path.join(
                output_directory,
                f'{code}.txt'
            )

            # Might change this behaviour in the future, but for now,
            # let's play it safe and not overwrite anything.
            if os.path.exists(output_file):
                continue

            # This has the added advantage that we now *see* whenever
            # a new spectrum is being stored.
            tqdm.write(code)

            X = bv.fit_transform([spectrum])[0]

            # Turn the spectrum vector into a data frame that tries to
            # at least partially maintain a description. This also has
            # the advantage of automatically generating an index.
            df = pd.DataFrame({'binned_intensity': X})
            df.index.name = 'bin_index'

            # Use a proper separator to be compatible with our reader.
            df.to_csv(output_file, sep=' ')
