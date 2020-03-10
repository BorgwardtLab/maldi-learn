"""Demonstrate hash capabilities."""

import dotenv
import json
import os

from maldi_learn.driams import DRIAMSDatasetExplorer

dotenv.load_dotenv()
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT')


if __name__ == '__main__':
    explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)

    print(json.dumps(
            explorer.metadata_fingerprints('DRIAMS-A'),
            indent=4
          ))
