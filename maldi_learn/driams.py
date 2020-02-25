'''
Main module for the DRIAMS dataset. Contains general exploration classes
and loaders.
'''

import dotenv
import os

import pandas as pd

from maldi_learn.data import MaldiTofSpectrum

# Pulls in the environment variables in order to simplify the access to
# the root directory.
dotenv.load_dotenv()

DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

_metadata_columns = ['code', 'bruker_organism_best_match', 'species']


class DRIAMSDatasetExplorer:
    def __init__(self, root=DRIAMS_ROOT):
        self.root = root

    def _get_available_sites(self):
        for _, dirs, _ in os.walk(self.root):
            sites = sorted(dirs)
            break
        return sites

    def _is_site_valid(self, site):
        '''
        Checks whether a specified site is valid. A site is considered
        valid if there is at least one ID file and at least one
        spectrum, either pre-processed or raw.

        Parameters
        ----------

            site:
                Name of the site to query. The function will build the
                necessary path to access the site automatically.

        Returns
        -------

        True if the site is valid according to the criterion specified
        above.
        '''

        path = os.path.join(self.root, site)

        for _, dirs, _ in os.walk(path):

            # Check whether ID directory exists
            if 'id' not in dirs:
                return False

            # Invalid if neither `preprocessed` nor `raw` exists as
            # a directory.`
            if 'preprocessed' not in dirs and 'raw' not in dirs:
                return False

            break

        # ID directory exists and at least one of `preprocessed` or
        # `raw` exists as well. Check all available IDs next.
        if not self._check_id_files(os.path.join(path, 'id')):
            return False

        return True

    def _check_id_files(self, id_directory):

        n_dirs = 0
        filenames = []

        for root, dirs, files in os.walk(id_directory):
            n_dirs += len(dirs)

            filenames.extend([os.path.join(root, f)
                for f in files if not f.startswith('.')]
            )

        # TODO: raise warning; each directory must contain a single file
        # only
        if n_dirs != len(filenames):
            return False

        # If we only get valid ID files, there must not be a `False`
        # entry in the list.
        valid = [self._is_id_valid(f) for f in filenames]
        return False not in valid

    def _is_id_valid(self, id_file):
        if not os.path.exists(id_file):
            return False

        try:
            df = pd.read_csv(id_file, low_memory=False)

            if 'code' not in df.columns:
                return False

        # Any error will make sure that this ID file is *not* valid
        except:
            return False

        # If we made it this far, the file is sufficiently well-formed
        # to not destroy everything.
        return True

    def _get_available_years(self, site):

        path = os.path.join(self.root, site)
        for _, dirs, files in os.walk(path):
            years = sorted(dirs)
            break

        # TODO: check whether spectrum information is available and
        # if each year has at least a single spectrum associated to
        # it.
        return years

    def _get_available_antibiotics(self, site, year):
        '''
        Queries a given site for the antibiotics that are available in
        it and returns them.

        Parameters
        ----------

        site:
            Identifier of the site that is to be queried. The function
            will build the paths accordingly.

        year:
            Year for which the given site should be queried. The
            function will build the paths accordingly.

        Returns
        -------

        List of antibiotic names, sorted in alphabetical order.
        '''

        path = os.path.join(
                self.root,
                site,
                'id',
                year,
                f'{year}_clean.csv'
        )

        df = pd.read_csv(path)
        antibiotics = [c for c in df.columns if c[0].isupper()]
        antibiotics = [a for a in antibiotics if 'Unnamed' not in a]

        return sorted(antibiotics)

    def available_antibiotics(self, site):
        """Return all available antibiotics for a given site.

        Returns
        -------
        All available antibiotics for the given site, in a `dict` whose
        keys represent the available years, and whose values represent
        the antibiotics.
        """
        return {
            year: self._get_available_antibiotics(site, year)
            for year in self.available_years(site)
        }

    def available_years(self, site):
        return self._get_available_years(site)

    @property
    def available_sites(self):
        return self._get_available_sites()



class DRIAMSDataset:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        # TODO include checks if input is valid


    @property
    def is_multitask(self):
        n_cols = [c for c in self.y.columns if c not in _metadata_columns]
        return n_cols != 1
    
    # TODO implement
    @property
    def n_samples(self):
        return 

    # TODO implement
    @property
    def class_ratio(self):
        # return dict with label as key, and class fraction as value
        return fraq_dict


def load_driams_dataset(
    root,
    site,
    year,
    species,
    antibiotic,
    handle_missing_values='remove_all_missing'
):
    """Load DRIAMS data set for a specific site and specific year.

    This is the main loading function for interacting with DRIAMS
    datasets. Given required information about a site, a year, and
    a list of antibiotics, this function loads a dataset, handles
    missing values, and returns a `DRIAMSDataset` class instance.

    Notice that no additional post-processing will be performed. The
    spectra might thus have different lengths are not directly suitable
    for downstream processing in, say, a `scikit-learn` pipeline.

    Parameters
    ----------

    root:
        Root path to the DRIAMS dataset folder.

    site:
        Identifier of a site, such as `DRIAMS-A`.

    year:
        Identifier for the year, such as `2015`.

    species:
        Identifier for the species, such as *Staphylococcus aureus*.

    antibiotic:
        Identifier for the antibiotic to use, such as *Ciprofloxacin*.

    handle_missing_resistance_measurements:
        Strategy for handling missing resistance measurements. Can be
        one of the following:

            'remove_all_missing'
            'remove_any_missing'
            'keep'

    Returns
    -------

    Instance of `DRIAMSDataset`, containing all loaded spectra.
    """
    # TODO make work for raw and preprocessed spectra
    path_X = os.path.join(root, site, 'preprocessed', year)
    id_file = os.path.join(root, site, 'id', year, f'{year}_clean.csv')
    
    # read in id 
    metadata = _load_metadata(id_file, species, antibiotic, handle_missing_values)    
    
    # extract spectra id
    codes = metadata.code
    
    # load spectra
    #spectra = [MaldiTofSpectrum(pd.read_csv(f'{code}.txt', sep=' ', comment='#', engine='c').values) for code in codes]
    
    spectra = []
    return spectra, metadata



def _load_metadata(filename, species, antibiotic, handle_missing_values):
    '''
    
    '''
    assert handle_missing_values in
    ['remove_all_missing', 'remove_any_missing', 'keep']
    metadata = pd.read_csv(filename)
    
    metadata = metadata.query('species == @species')
    print(metadata.species)

    # TODO cleaner
    metadata = metadata[_metadata_columns+[antibiotic]]
    
    # handle_missing_values
    if handle_missing_values=='remove_all_missing' or
    handle_missing_values=='remove_any_missing':
        metadata = metadata.iloc[~metadata[antibiotic].isna().values]
    else:
        pass
    return metadata    



# HERE BE DRAGONS

explorer = DRIAMSDatasetExplorer('/Volumes/borgwardt/Data/DRIAMS')

print(explorer.__dict__)
print(explorer.available_sites)
print(explorer.available_years)
print(explorer._is_site_valid('DRIAMS-A'))

_, df = load_driams_dataset(explorer.root, 'DRIAMS-A', '2017', 'Staphylococcus aureus', 'Ciprofloxacin')

print(df.to_numpy().shape)
print(df.to_numpy().dtype)
print(df.to_numpy()[0])

print(explorer._get_available_antibiotics('DRIAMS-A', '2017'))
