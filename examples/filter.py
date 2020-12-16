"""Simple filter example.

This script demonstrates how to perform filtering using
`maldi-learn` and the DRIAMS data set.
"""

import dotenv
import os

from maldi_learn.driams import load_driams_dataset

from maldi_learn.filters import DRIAMSSpeciesFilter
from maldi_learn.filters import filter_by_machine_type


dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

driams_dataset = load_driams_dataset(
            DRIAMS_ROOT,
            'DRIAMS-A',
            ['2015', '2018'],
            '*',
            'Ciprofloxacin',
            handle_missing_resistance_measurements='keep',
            nrows=200,
            extra_filters=[
                filter_by_machine_type,
                DRIAMSSpeciesFilter(['Pseudomonas', 'coli'])
            ],
)

print(driams_dataset.y)
