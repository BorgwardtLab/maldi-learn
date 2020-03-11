import dotenv
import os

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder
from maldi_learn.driams import load_driams_dataset

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)

driams_dataset = load_driams_dataset(
            explorer.root,
            'DRIAMS-A',
            ['2015', '2016', '2017', '2018'],
            '*',
            ['Trimethprim-Sulfamethoxazole'],
            encoder=DRIAMSLabelEncoder(),
            handle_missing_resistance_measurements='remove_if_all_missing',
)

print(driams_dataset.y)
