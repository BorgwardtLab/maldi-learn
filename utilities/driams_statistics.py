"""Print summary statistics of DRIAMS spectra.

The purpose of this script is to print some summary statistics about DRIAMS
data sets, stratified by site. This is just a debug script with no usage in
real-world analysis scenarios.
"""

import argparse
import dotenv
import os

import numpy as np

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import load_driams_dataset

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
                spectra_type='binned_6000',
        )

        codes = driams_dataset.y['code'].values

        for spectrum, code in tqdm(zip(driams_dataset.X, codes),
                                   total=len(codes),
                                   desc='Spectrum'):

            # Use intensity values only
            if len(spectrum.shape) == 2:
                spectrum = spectrum[:, 1]

            min_value, max_value = np.min(spectrum), np.max(spectrum)
            tic = np.sum(spectrum)

            print(f'*** {code} ***')
            print(f'Min: {min_value:.08f}')
            print(f'Max: {max_value:.08f}')
            print(f'TIC: {tic:.2f}')
