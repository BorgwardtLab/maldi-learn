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

antibiotics = [
    '5-Fluorocytosin',
    'Amikacin',
    'Amoxicillin',
    'Amoxicillin-Clavulanic acid',
    'Ampicillin-Amoxicillin',
    'Anidulafungin',
    'Aztreonam',
    'Caspofungin',
    'Cefazolin',
    'Cefepime',
    'Cefpodoxime',
    'Ceftazidim',
    'Cefuroxime',
    'Ceftriaxone',
    'Ciprofloxacin',
    'Clindamycin',
    'Colistin',
    'Daptomycin',
    'Ertapenem',
    'Erythromycin',
    'Fluconazole',
    'Fosfomycin-Trometamol',
    'Fusidic acid',
    'Gentamicin',
    'Imipenem',
    'Itraconazole',
    'Levofloxacin',
    'Meropenem',
    'Micafungin',
    'Nitrofurantoin',
    'Norfloxacin',
    'Oxacillin',
    'Penicillin',
    'Piperacillin-Tazobactam',
    'Rifampicin',
    'Teicoplanin',
    'Tetracycline',
    'Tobramycin',
    'Trimethoprim-Sulfamethoxazole',
    'Tigecycline',
    'Vancomycin',
    'Voriconazole',
]

driams_dataset = load_driams_dataset(
            explorer.root,
            'DRIAMS-A',
            ['2015', '2016', '2017', '2018'],
            '*',
            antibiotics=antibiotics,
            encoder=DRIAMSLabelEncoder(),
            handle_missing_resistance_measurements='remove_if_all_missing',
)

# train-test split
train_index, test_index = stratify_by_species_and_label(
    driams_dataset.y, antibiotic='5-Fluorocytosin',
    remove_invalid=True,
)

print(train_index)
print(test_index)
