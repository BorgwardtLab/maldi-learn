"""Main module for the DRIAMS data set.

This is the main module for the DRIAMS data set. It contains general
exploration classes and loaders.
"""

import collections
import dotenv
import hashlib
import os
import itertools
import warnings

import numpy as np
import pandas as pd

from maldi_learn.data import MaldiTofSpectrum
from maldi_learn.preprocessing.generic import LabelEncoder

from maldi_learn.exceptions import AntibioticNotFoundException
from maldi_learn.exceptions import AntibioticNotFoundWarning
from maldi_learn.exceptions import SpeciesNotFoundException
from maldi_learn.exceptions import SpeciesNotFoundWarning
from maldi_learn.exceptions import SpectraNotFoundException
from maldi_learn.exceptions import SpectraNotFoundWarning
from maldi_learn.exceptions import _raise_or_warn

from maldi_learn.filters import DRIAMSFilter

# Pulls in the environment variables in order to simplify the access to
# the root directory.
dotenv.load_dotenv()

# Root directory for the DRIAMS data set. Should contain additional
# folders for each site and each year. The variable is just used as
# a fall-back measure. It can be client-specified.
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')

# These are the columns that we consider to contain metadata for the
# DRIAMS data set. Note that they will only be used if *present*. It
# is not an error if one of them is missing.
_metadata_columns = [
    'id',
    'code',
    'species',
    'laboratory_species',
    'case_no',
    'acquisition_date',
    'workstation',
]


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


def _check_id_file(id_file):
    """Check whether an ID file is valid.

    This is an internal function for checking the consistency of
    a DRIAMS id file. The file is consistent if it the following
    conditions are met:

        - the file must contain a 'species' column
        - the file must contain a 'code' column
        - the 'code' column must not contain `NaN` values
        - neither one of these columns must be empty

    Parameters
    ----------
    id_file : str
        Full path to the ID file that should be checked.

    Returns
    -------
    `True` if the ID file is valid, else `False`.
    """
    if not os.path.exists(id_file):
        warnings.warn(f'File {id_file} not found. This will cause an error.')
        return False

    try:
        df = pd.read_csv(id_file, low_memory=False)

        if 'code' not in df.columns or 'species' not in df.columns:
            warnings.warn('Either "code" column or "species" column '
                          'is missing.')
            return False

        if df['code'].empty or df['code'].isna().sum() != 0:
            warnings.warn('Either "code" column is empty or it contains '
                          'NaN values.')
            return False

        # Species information is allowed to be missing; while some of
        # these samples cannot be readily used for classification, it
        # it possible that another type of analysis might use them.
        if df['species'].empty:
            warnings.warn('"Species" column is empty.')
            return False

    # Any exception will make sure that this ID file is *not* valid; we
    # are not sure which exception to expect here.
    except Exception as e:
        warnings.warn(f'Caught the following exception: {e}')
        return False

    # If we made it this far, the file is sufficiently well-formed
    # to not destroy everything.
    return True


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
        valid = [_check_id_file(f) for f in filenames]
        return False not in valid

    def _get_available_years(self, site):

        path = os.path.join(self.root, site, 'id')
        years = []

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
        antibiotics = [c for c in df.columns if not c[0].islower()]
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

    def metadata_fingerprints(self, site, id_suffix='clean'):
        """Return available metadata filenames and their fingerprints.

        This function is a purely descriptive function whose goal is to
        provide more information about the metadata files. For each one
        of these files, it will calculate the SHA-1 hash and return it.

        Parameters
        ----------
        site : str
            Specifies which site should be used for the fingerprint
            information.

        id_suffix : str
            Optional suffix for specifying that different versions of
            the ID files shall be used.
        """
        hashes = {}
        for year in self.available_years(site):
            path = os.path.join(
                        self.root,
                        site,
                        'id',
                        year,
                        f'{year}_{id_suffix}.csv'
                )

            df = pd.read_csv(path, low_memory=False)

            hash_ = hashlib.sha1(
                    pd.util.hash_pandas_object(df, index=True).values
                ).hexdigest()

            hashes[os.path.basename(path)] = hash_

        return hashes


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

    @property
    def class_ratio(self, antibiotic):
        # extract copy of series
        ab_series = self.y[antibiotic].dropna()
        # return dict with label as key, and class fraction as value
        return ab_series.value_counts(normalize=True).to_dict()

    def to_numpy(self, antibiotic, dtype=int, y=None):
        """Convert label feature vector to `numpy` array.

        Given a data set and an antibiotic, this function creates
        `numpy` array for use in downstream analysis tasks.

        Parameters
        ----------
        antibiotic : str
            Name of the antibiotic for which labels are supposed to be
            returned.

        dtype : type
            Sets type of the created array. Normally, this should not
            have to be changed.

        y : `pandas.DataFrame`
            Optional data frame whose labels should be converted to an
            array. If set, applies all information to a copy of `y`
            instead of applying it to the current data set.

        Returns
        -------
        `numpy.ndarray` of shape (n, 1), where n is the number of
        samples in the data set.
        """
        if y is None:
            y = self.y

        # TODO: is it necessary to ignore all metadata columns here?
        y = y.loc[:, [c for c in self.y.columns if c not in _metadata_columns]]

        return y[antibiotic].to_numpy().astype(dtype)


def load_spectrum(filename, on_error):
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
    if os.path.isfile(filename):
        return MaldiTofSpectrum(
                    pd.read_csv(
                        filename,
                        sep=' ',
                        comment='#',
                        engine='c').values
                )
    else:
        _raise_or_warn(
                SpectraNotFoundException,
                SpectraNotFoundWarning,
                f'Spectra filename does not exist: {filename}',
                on_error
        )
        return None


def load_driams_dataset(
    root,
    site,
    years,
    species,
    antibiotics,
    encoder=DRIAMSLabelEncoder(),
    handle_missing_resistance_measurements='remove_if_all_missing',
    spectra_type='preprocessed',
    on_error='raise',
    id_suffix='clean',
    extra_filters=[],
    **kwargs,
):
    """Load DRIAMS data set for a specific site and specific year.

    This is the main loading function for interacting with DRIAMS
    datasets. Given required information about a site, a year, and
    a list of antibiotics, this function loads a dataset, handles
    missing values, and returns a `DRIAMSDataset` class instance.

    Note that no additional post-processing will be performed. The
    spectra might have different lengths that cannot be used for a
    downstream processing or analysis task, or in a `scikit-learn`
    pipeline.

    To change this behaviour, load a certain type of spectra, such
    as `binned_6000`.

    Parameters
    ----------
    root : str
        Root path to the DRIAMS dataset folder.

    site : str
        Identifier of a site, such as `DRIAMS-A`.

    years : str or list of str
        Identifier for the year, such as `2015`. Can be either a `list`
        of strings or a single `str`, in which case only one year will
        be loaded. If set to `*`, returns all available years.

    species : str
        Identifier for the species, such as *Staphylococcus aureus*. If
        set to `*`, returns all species, thus performing no filtering.

    antibiotics : str or list of str
        Identifier for the antibiotics to use, such as *Ciprofloxacin*.
        Can be either a `list` of strings or a single `str`, in which
        case only a single antibiotic will be loaded.

    encoder : `LabelEncoder` instance or `None`
        If set, provides a mechanism for encoding labels into numbers.
        This will be applied *prior* to the missing value handling, so
        it is a simple strategy to remove invalid values. If no encoder
        is set (i.e. the parameter is `None`), only missing values in
        the original data will be handled. By default, an encoder that
        should be suitable for most tasks is used; `DRIAMSLabelEncoder`
        implements our preferred encoding of labels.

    handle_missing_resistance_measurements : str
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

    on_error : str
        Sets the behaviour in case of an error. If set to 'raise', the
        code will raise an exception for every error it encounters. If
        set to 'warn' or 'warning', only a warning will be shown.

    id_suffix : str
        An optional suffix that is applied when searching for ID files.
        This parameter does not have to be changed during normal
        operations and is only useful when debugging.

    extra_filters : list of callable
        Optional filter functions that will be applied to the data set
        before returning it to the user. Filters will be applied in the
        exact ordering in which they are supplied to this function.

    kwargs:
        Optional keyword arguments for changing the downstream behaviour
        of some functions. At present, the following keys are supported:

            - `nrows`: specifies number of rows to read from the data
              frame; reducing this is useful for debugging

    Returns
    -------
    Instance of `DRIAMSDataset`, containing all loaded spectra.
    """
    # Get all available years
    if years == '*':
        years = DRIAMSDatasetExplorer(root).available_years(site)
    # Pretend that we always have a list of years
    elif type(years) is not list:
        years = [years]

    all_spectra = {}
    all_metadata = {}

    for year in years:
        path_X = os.path.join(root, site, spectra_type, year)

        # Determine filename portion for loading the ID file
        if id_suffix is not None:
            filename = f'{year}_{id_suffix}.csv'
        else:
            filename = f'{year}.csv'

        id_file = os.path.join(
            root,
            site,
            'id',
            year,
            filename
        )

        # Metadata contains all information that we have about the
        # individual spectra and the selected antibiotics.
        metadata = _load_metadata(
            id_file,
            species,
            antibiotics,
            encoder,
            handle_missing_resistance_measurements,
            on_error,
            **kwargs,
        )

        if extra_filters:
            driams_filter = DRIAMSFilter(extra_filters)
            mask = metadata.apply(driams_filter, axis=1)
            metadata = metadata[mask]

        # The codes are used to uniquely identify the spectra that we can
        # load. They are required for matching files and metadata.
        codes = metadata.code

        spectra_files = [
            os.path.join(path_X, f'{code}.txt') for code in codes
        ]

        spectra = [
            load_spectrum(f, on_error) for f in spectra_files
        ]

        # Remove missing spectra from metadata and spectra
        missing_codes = [
            c for c, s in zip(codes, spectra) if s is None
        ]

        metadata = metadata[~metadata['code'].isin(missing_codes)]
        spectra = [
            s for s in spectra if s is not None
        ]
        codes = metadata.code

        # Indentify codes with NaNs in spectrum
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
    on_error,
    **kwargs,
):
    """Load metadata file.

    This function does the 'heavy lifting' for loading the metadata
    files. It ensures that all desired species and antibiotics are
    loaded correctly and encoded for subsequent processing.

    Please refer to `load_driams_dataset()` for a description of all
    parameters.
    """
    # Ensures that we always get a list of antibiotics for subsequent
    # processing.
    if (not isinstance(antibiotics, collections.abc.Sequence) or
            isinstance(antibiotics, str)):
        antibiotics = [antibiotics]

    assert handle_missing_resistance_measurements in [
            'remove_if_all_missing',
            'remove_if_any_missing',
            'keep'
    ]

    if not _check_id_file(filename):
        raise RuntimeError(f'ID file {filename} is invalid. Please check '
                           f'whether it contains all required columns.')

    metadata = pd.read_csv(
                    filename,
                    low_memory=False,
                    na_values=['-'],        # additional way to encode `Nan`
                    keep_default_na=True,   # keep default `NaN` encodings
                    nrows=kwargs.get('nrows', None),
                )

    # Perform no species filtering if *all* species are requested.
    if species != '*':

        if species not in metadata['species'].values:
            _raise_or_warn(
                    SpeciesNotFoundException,
                    SpeciesNotFoundWarning,
                    f'Species {species} was not found',
                    on_error
            )

        metadata = metadata.query('species == @species')

    # Check existence of each antibiotic prior to re-indexing the data
    # frame. We do not want to return invalid data frames, even though
    # it is possible that a data frame is empty, depending on how this
    # function handles missing values.
    for antibiotic in antibiotics:

        if antibiotic not in metadata.columns:
            _raise_or_warn(
                AntibioticNotFoundException,
                AntibioticNotFoundWarning,
                f'Antibiotic {antibiotic} was not found',
                on_error
            )

    # Not all label files might have the same meta columns available, so
    # we only use the ones that *are* available.
    metadata_columns_available = [
        c for c in _metadata_columns if c in metadata.columns
    ]

    # Type-cast all columns into `object`. This ensures that the label
    # encoding works correctly in all cases because `object` makes it
    # possible to handle `nan` and arbitrary strings.
    metadata = metadata.astype({a: 'object' for a in antibiotics}, copy=False)

    # Ensures that all requested antibiotics are present in the
    # data frame. Afterwards, we restrict the data frame to the
    # relevant columns.
    metadata = metadata.reindex(
        columns=metadata_columns_available + antibiotics
    )
    metadata = metadata[metadata_columns_available + antibiotics]

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
