'''
Main module for the DRIAMS dataset. Contains general exploration classes
and loaders.
'''

import os
import glob

import pandas as pd


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


    @property
    def available_sites(self):
        return self._get_available_sites()

# HERE BE DRAGONS

explorer = DRIAMSDatasetExplorer('/Volumes/borgwardt/Data/DRIAMS')

print(explorer.available_sites)
print(explorer._is_site_valid('DRIAMS-A'))
