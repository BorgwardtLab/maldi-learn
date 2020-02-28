"""
"""

from driams import DRIAMSDatasetExplorer
from driams import DRIAMSDataset
from driams import DRIAMSLabelEncoder

from driams import load_driams_dataset
from maldi_learn.vectorization import BinningVectorizer

explorer = DRIAMSDatasetExplorer('/Volumes/borgwardt/Data/DRIAMS')


driams_dataset = load_driams_dataset(
            explorer.root,
            'DRIAMS-A',
            ['2015', '2017'],
            'Staphylococcus aureus',
            ['Ciprofloxacin', 'Penicillin.ohne.Meningitis'],
            encoder=DRIAMSLabelEncoder(),
            handle_missing_resistance_measurements='remove_if_all_missing',
)



bv = BinningVectorizer(1000, min_bin=2000, max_bin=20000)

X = bv.fit_transform(driams_dataset.X)
print(X.shape)

