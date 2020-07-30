"""Example script for generating a summary of the DRIAMS data set.

This script demonstrates how to explore the DRIAMS data set and create
a useful summary of its available information.
"""


import dotenv
import os

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder
from maldi_learn.driams import load_driams_dataset


if __name__ == '__main__':
    dotenv.load_dotenv()
    DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)
    sites = explorer.available_sites

    for site in sites:
        antibiotics = explorer.available_antibiotics(site)
        years = sorted(antibiotics.keys())

        for year in years:
            print(year)

            driams_dataset = load_driams_dataset(
                    explorer.root,
                    site,
                    years=year,
                    species='*',
                    antibiotics=antibiotics[year],
                    encoder=DRIAMSLabelEncoder(),
            )

            y = driams_dataset.y

            for antibiotic in sorted(antibiotics[year]):
                counts = y[antibiotic].value_counts().to_dict()
                print(antibiotic, counts)
