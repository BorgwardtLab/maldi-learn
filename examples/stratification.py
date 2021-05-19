"""Demo for stratification options."""

import dotenv
import logging
import os

from maldi_learn.driams import DRIAMSDatasetExplorer
from maldi_learn.driams import DRIAMSLabelEncoder
from maldi_learn.driams import load_driams_dataset

from maldi_learn.utilities import case_based_stratification
from maldi_learn.utilities import stratify_by_species_and_label


dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)

antibiotics = [
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
    'Tigecycline',
    'Vancomycin',
    'Voriconazole',
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

logging.info('Loading data set')

driams_dataset = load_driams_dataset(
            explorer.root,
            'DRIAMS-A',
            '*',
            '*',
            antibiotics=antibiotics,
            handle_missing_resistance_measurements='remove_if_all_missing',
            nrows=500,
            id_suffix='strat',
            spectra_type='binned_6000',

)

logging.info('Finished loading data set')

for antibiotic in antibiotics:

    logging.info(f'Splitting for {antibiotic}...')

    try:
        train_index_1, test_index_1, train_labels_1, test_labels_1 = \
            stratify_by_species_and_label(
                driams_dataset.y, antibiotic=antibiotic,
                implementation='numpy',
                return_stratification=True,
            )

    except ValueError:
        continue

    logging.info('Finished first stratification')

    train_index_2, test_index_2, train_labels_2, test_labels_2 = \
        stratify_by_species_and_label(
            driams_dataset.y, antibiotic=antibiotic,
            implementation='pandas',
            return_stratification=True,
        )

    assert (train_index_1 == train_index_2).all()
    assert (test_index_1 == test_index_2).all()
    assert (train_labels_1 == train_labels_2).all()
    assert (test_labels_1 == test_labels_2).all()

    logging.info('Finished second stratification')

    train_index_3, test_index_3, train_labels_3, test_labels_3 = \
        case_based_stratification(
            driams_dataset.y, antibiotic=antibiotic,
            test_size=0.20,
            return_stratification=True,
        )

    assert (len(train_index_3) == len(train_labels_3))
    assert (len(test_index_3) == len(test_labels_3))
