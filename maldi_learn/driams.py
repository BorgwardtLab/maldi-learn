'''
Main module for the DRIAMS dataset. Contains general exploration classes
and loaders.
'''

import os
import glob

import pandas as pd

from maldi_learn.data import MaldiTofSpectrum

_metadata_columns = ['code', 'bruker_organism_best_match', 'species']


class DRIAMSDatasetExplorer:
    def __init__(self, root):
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
        
        print(years)
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
    


def load_driams_dataset(root, site, year, species, antibiotic, handle_missing_values='remove_all_missing'):
   
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
    print(filename)
    metadata = pd.read_csv(filename)
    
    metadata = metadata.query('species == @species')
    print(metadata.species)

    # TODO cleaner
    metadata = metadata[_metadata_columns]
    # TODO include handle_missing_values

    return metadata    



# HERE BE DRAGONS

explorer = DRIAMSDatasetExplorer('/Volumes/bs-dfs/Data/DRIAMS')

print(explorer.__dict__)
print(explorer.available_sites)
print(explorer.available_years)
print(explorer._is_site_valid('DRIAMS-A'))

_, df = load_driams_dataset('/Volumes/bs-dfs/Data/DRIAMS', 'DRIAMS-A', '2015', 'Staphylococcus aureus', 'Ciprofloxacin')

print(df.to_numpy().shape)
print(df.to_numpy().dtype)
print(df.to_numpy()[0])

print(explorer._get_available_antibiotics('DRIAMS-A', '2015'))
