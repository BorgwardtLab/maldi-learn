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
    antibiotics,
    handle_missing_resistance_measurements='remove_if_all_missing',
    load_raw=False,
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

    antibiotics:
        Identifier for the antibiotics to use, such as *Ciprofloxacin*.
        Can be either a `list` of strings or a single `str`, in which
        case only a single antibiotic will be loaded.

    handle_missing_resistance_measurements:
        Strategy for handling missing resistance measurements. Can be
        one of the following:

            'remove_if_all_missing'
            'remove_if_any_missing'
            'keep'

    load_raw:
        If set, loads the *raw* spectra instead of the pre-processed
        one. This has no bearing whatsoever on the labels and metadata
        and merely changes the resulting spectra. If not set, loads
        the pre-processed spectra instead.

    Returns
    -------

    Instance of `DRIAMSDataset`, containing all loaded spectra.
    """
    if load_raw:
        spectra_type = 'raw'
    else:
        spectra_type = 'preprocessed'

    path_X = os.path.join(root, site, spectra_type, year)
    id_file = os.path.join(root, site, 'id', year, f'{year}_clean.csv')

    # Metadata contains all information that we have about the
    # individual spectra and the selected antibiotics.
    metadata = _load_metadata(
        id_file,
        species,
        antibiotics,
        handle_missing_resistance_measurements
    )

    # The codes are used to uniquely identify the spectra that we can
    # load. They are required for matching files and metadata.
    codes = metadata.code

    spectra_files = [
        os.path.join(path_X, f'{code}.txt') for code in codes
    ]

    spectra = [
        MaldiTofSpectrum(
            pd.read_csv(f, sep=' ', comment='#', engine='c').values
        ) for f in spectra_files
    ]

    return spectra, metadata


def _load_metadata(
    filename,
    species,
    antibiotics,
    handle_missing_resistance_measurements
):

    # Ensures that we always get a list of antibiotics for subsequent
    # processing.
    if type(antibiotics) is not list:
        antibiotics = [antibiotics]

    assert handle_missing_resistance_measurements in [
            'remove_if_all_missing',
            'remove_if_any_missing',
            'keep'
    ]

    metadata = pd.read_csv(
                    filename,
                    low_memory=False,
                    na_values=['-'],
                    keep_default_na=True,
                )

    metadata = metadata.query('species == @species')
    print(metadata.species)

    # TODO make cleaner
    metadata = metadata[_metadata_columns + antibiotics]
    n_antibiotics = len(antibiotics)

    # handle_missing_values
    if handle_missing_resistance_measurements == 'remove_if_all_missing':
        na_values = metadata[antibiotics].isna().sum(axis='columns')
        metadata = metadata[na_values != n_antibiotics]
    elif handle_missing_resistance_measurements == 'remove_if_any_missing':
        na_values = metadata[antibiotics].isna().sum(axis='columns')
        metadata = metadata[na_values == 0]
    else:
        pass

    return metadata


# HERE BE DRAGONS

explorer = DRIAMSDatasetExplorer('/Volumes/borgwardt/Data/DRIAMS')

print(explorer._get_available_antibiotics('DRIAMS-A', '2015'))

print(explorer.__dict__)
print(explorer.available_sites)
print(explorer.available_years)
print(explorer._is_site_valid('DRIAMS-A'))

_, df = load_driams_dataset(
            explorer.root,
            'DRIAMS-A',
            '2017',
            'Staphylococcus aureus',
            ['Ciprofloxacin', 'Penicillin'],
            'remove_if_all_missing'
)

print(df.to_numpy().shape)
print(df.to_numpy().dtype)
print(df.to_numpy()[0])

print(explorer._get_available_antibiotics('DRIAMS-A', '2017'))
