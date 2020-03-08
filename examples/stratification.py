"""Demo for stratification options."""

import dotenv
import os

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder
from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import stratify_by_species_and_label

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)

driams_dataset = load_driams_dataset(
            explorer.root,
            'DRIAMS-A',
            ['2015', '2017'],
            '*',
            ['5-Fluorocytosin', 'Ciprofloxacin'],
            encoder=DRIAMSLabelEncoder(),
            handle_missing_resistance_measurements='remove_if_all_missing',
            nrows=4000,
)

# train-test split
train_index, test_index = stratify_by_species_and_label(
    driams_dataset.y, antibiotic='5-Fluorocytosin',
    remove_invalid=True,
)

print(train_index)
print(test_index)
