"""Main module for the DRIAMS data set.

This is the main module for the DRIAMS data set. It contains general
exploration classes and loaders.
"""

import dotenv
import os
import itertools
import warnings

import numpy as np
import pandas as pd

from maldi_learn.data import MaldiTofSpectrum
from maldi_learn.preprocessing.generic import LabelEncoder

# Pulls in the environment variables in order to simplify the access to
# the root directory.
dotenv.load_dotenv()

# Root directory for the DRIAMS data set. Should contain additional
# folders for each site and each year. The variable is just used as
# a fall-back measure. It can be client-specified.
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

# These are the columns that we consider to contain metadata for the
# DRIAMS data set.
_metadata_columns = ['code', 'species', 'laboratory_species']


class DRIAMSDatasetExplorer:
    """Explorer class for the DRIAMS data set."""

    def __init__(self, root=DRIAMS_ROOT):
        """Create new instance based on a root data directory."""
        self.root = root

    def _get_available_sites(self):
        for _, dirs, _ in os.walk(self.root):
            sites = sorted(dirs)
            break
        return sites

    def _is_site_valid(self, site):
        """Check whether a specified site is valid.

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
        """
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
        """Check ID files for consistency and correctness.

        Checks the ID files of the DRIAMS data set for consistency and
        correctness. Makes sure that all files are properly referenced.
        """
        n_dirs = 0
        filenames = []

        for root, dirs, files in os.walk(id_directory):
            n_dirs += len(dirs)

            filenames.extend([
                os.path.join(root, f)
                for f in files if not f.startswith('.')
            ])

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

        path = os.path.join(self.root, site, 'id')
        for _, dirs, files in os.walk(path):
            years = sorted(dirs)
            break

        # TODO: check whether spectrum information is available and
        # if each year has at least a single spectrum associated to
        # it.
        return years

    def _get_available_antibiotics(self, site, year):
        """Query a given site for the antibiotics available in a given year.

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
        """
        path = os.path.join(
                self.root,
                site,
                'id',
                year,
                f'{year}_clean.csv'
        )

        df = pd.read_csv(path, low_memory=False)
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
        """Return available years for a given site."""
        return self._get_available_years(site)

    @property
    def available_sites(self):
        """Return available sites in the data set."""
        return self._get_available_sites()


class DRIAMSDataset:
    """DRIAMS data set."""

    def __init__(self, X, y):
        """Create new DRIAMS data set.

        Parameters
        ----------
        X:
            List of `MaldiTofSpectra` objects.
        y:
            Metadata data frame (`pandas.DataFrame`). Columns with
            antimicrobial information are indicated by capitalized
            header.
        """
        # checks if input is valid
        assert len(X) == y.shape[0]

        self.X = X
        self.y = y

    @property
    def is_multitask(self):
        n_cols = [c for c in self.y.columns if c not in _metadata_columns]
        return n_cols != 1

    @property
    def n_samples(self):
        return self.y.shape[0]

    @property
    def n_label_avail(self):
        return self.y.loc[:, [c for c in self.y.columns if c not in
            _metadata_columns]].notna().sum(axis=0)

    # TODO implement
    @property
    def class_ratio(self, antibiotic):
        print(self.y.count())
        # return dict with label as key, and class fraction as value
        return fraq_dict

    def to_numpy(self, antibiotic): 
        # return y as numpy array as imput for classification
        y = self.y.loc[:, [c for c in self.y.columns if c not in
            _metadata_columns]]
        return y[antibiotic].to_numpy().astype(int)


def load_spectrum(filename):
    """Load DRIAMS MALDI-TOF spectrum.

    This function encapsulates loading a MALDI-TOF spectrum from the
    DRIAMS data set. It should be used in lieu of any raw calls that
    aim to load a spectrum.

    Parameters
    ----------
    filename : str
        Filename from which to load the spectrum.

    Returns
    -------
    Instance of `MaldiTofSpectrum` class, containing the spectrum.
    """
    return MaldiTofSpectrum(
                pd.read_csv(
                    filename,
                    sep=' ',
                    comment='#',
                    engine='c').values
            )


def load_driams_dataset(
    root,
    site,
    years,
    species,
    antibiotics,
    encoder=None,
    handle_missing_resistance_measurements='remove_if_all_missing',
    spectra_type='preprocessed',
    **kwargs,
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

    years:
        Identifier for the year, such as `2015`. Can be either a `list`
        of strings or a single `str`, in which case only one year will
        be loaded.

    species:
        Identifier for the species, such as *Staphylococcus aureus*. If
        set to `*`, returns all species, thus performing no filtering.

    antibiotics:
        Identifier for the antibiotics to use, such as *Ciprofloxacin*.
        Can be either a `list` of strings or a single `str`, in which
        case only a single antibiotic will be loaded.

    encoder:
        If set, provides a mechanism for encoding labels into numbers.
        This will be applied *prior* to the missing value handling, so
        it is a simple strategy to remove invalid values. If no encoder
        is set, only missing values in the original data will be
        handled.

        Suitable values for `encoder` are instances of the
        `DRIAMSLabelEncoder` class, which performs our preferred
        encoding of labels.

    handle_missing_resistance_measurements:
        Strategy for handling missing resistance measurements. Can be
        one of the following:

            'remove_if_all_missing'
            'remove_if_any_missing'
            'keep'

    spectra_type : str
        Sets the type of data to load. This must refer to a folder
        within the hierarchy of DRIAMS containing the same spectra
        that are listed in the corresponding ID files. Setting new
        types can be useful for loading pre-processed spectra such
        as spectra that have already been binned.

        Changing this option has no effect on the metadata or the
        labels. It only affects the spectra. The following values
        are always valid:

            - `raw`: loads raw spectra (no pre-processing)
            - `preprocessed`: loads pre-processed spectra, whose peaks
              have been aligned etc.

        By default, pre-processed spectra are loaded.

    kwargs:
        Optional keyword arguments for changing the downstream behaviour
        of some functions. At present, the following keys are supported:

            - `nrows`: specifies number of rows to read from the data
              frame; reducing this is useful for debugging

    Returns
    -------
    Instance of `DRIAMSDataset`, containing all loaded spectra.
    """
    if type(years) is not list:
        years = [years]

    all_spectra = {}
    all_metadata = {}

    for year in years:
        path_X = os.path.join(root, site, spectra_type, year)
        id_file = os.path.join(root, site, 'id', year, f'{year}_clean.csv')

        # Metadata contains all information that we have about the
        # individual spectra and the selected antibiotics.
        metadata = _load_metadata(
            id_file,
            species,
            antibiotics,
            encoder,
            handle_missing_resistance_measurements,
            **kwargs,
        )

        # The codes are used to uniquely identify the spectra that we can
        # load. They are required for matching files and metadata.
        codes = metadata.code

        spectra_files = [
            os.path.join(path_X, f'{code}.txt') for code in codes
        ]

        spectra = [
            load_spectrum(f) for f in spectra_files
        ]

        problematic_codes = [
            c for c, s in zip(codes, spectra) if np.isnan(s).any()
        ]

        if problematic_codes:
            warnings.warn(f'Found problematic codes: {problematic_codes}')

        all_spectra[year] = spectra
        all_metadata[year] = metadata

    spectra, metadata = _merge_years(all_spectra, all_metadata)
    return DRIAMSDataset(spectra, metadata)


def _load_metadata(
    filename,
    species,
    antibiotics,
    encoder,
    handle_missing_resistance_measurements,
    **kwargs,
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
                    nrows=kwargs.get('nrows', None),
                )

    # Perform no species filtering if *all* species are requested.
    if species != '*':
        metadata = metadata.query('species == @species')

    # ensures that all requested antibiotics are present in the
    # dataframe. might be filled with nans if not present
    metadata = metadata.reindex(columns=_metadata_columns + antibiotics)

    # TODO raise warning if absent antibiotics are requested
    # TODO make cleaner

    metadata = metadata[_metadata_columns + antibiotics]
    n_antibiotics = len(antibiotics)

    if encoder is not None:
        metadata = encoder.fit_transform(metadata)

    if handle_missing_resistance_measurements == 'remove_if_all_missing':
        na_values = metadata[antibiotics].isna().sum(axis='columns')
        metadata = metadata[na_values != n_antibiotics]
    elif handle_missing_resistance_measurements == 'remove_if_any_missing':
        na_values = metadata[antibiotics].isna().sum(axis='columns')
        metadata = metadata[na_values == 0]
    else:
        pass

    return metadata


def _merge_years(all_spectra, all_metadata):

    all_columns = set()
    for df in all_metadata.values():
        all_columns.update(df.columns)

    for year in all_metadata.keys():
        all_metadata[year] = all_metadata[year].reindex(columns=all_columns)

    metadata = pd.concat([df for df in all_metadata.values()])
    spectra = [s for s in itertools.chain.from_iterable(all_spectra.values())]

    assert sum(metadata.duplicated(subset=['code'])) == 0, \
        'Duplicated codes in different years.'

    return spectra, metadata


class DRIAMSLabelEncoder(LabelEncoder):
    """Encoder for DRIAMS labels.

    Encodes antibiotic resistance measurements in a standardised manner.
    Specifically, *resistant* or *intermediate* measurements are will be
    converted to `1`, while *suspectible* measurements will be converted
    to `0`.
    """

    def __init__(self):
        """Create new instance of the encoder."""
        # These are the default encodings for the DRIAMS dataset. If
        # other values show up, they will not be handled; this is by
        # design.
        encodings = {
            'R': 1,
            'I': 1,
            'S': 0,
            'S(2)': np.nan,
            'R(1)': np.nan,
            'R(2)': np.nan,
            'L(1)': np.nan,
            'I(1)': np.nan,
            'I(1), S(1)': np.nan,
            'R(1), I(1)': np.nan,
            'R(1), S(1)': np.nan,
            'R(1), I(1), S(1)': np.nan
        }

        # Ignore the metadata columns to ensure that these values will
        # not be replaced anywhere else.
        super().__init__(encodings, _metadata_columns)
