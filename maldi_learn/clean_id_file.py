#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd


def clean_data(filename):
    df = pd.read_csv(filename, low_memory=False, index_col='code')

    print(np.argwhere(df.index.isna()))

    # Select columns for deletion. We want to remove columns that could
    # potentially leak patient information.
    columns_to_delete = [
        'strain',
        'TAGESNUMMER',
        'Value',
        'A',
        'Organism(best match)',
        'Score1',
        'Organism(second best match)',
        'Score2',
        'SPEZIES_MALDI',
        'GENUS',
        'SPEZIES_MLAB',
        'MATERIAL',
        'AUFTRAGSNUMMER',
        'STATION',
        'PATIENTENNUMMER',
        'GEBURTSDATUM',
        'GESCHLECHT',
        'EINGANGSDATUM',
        'LOKALISATION'
    ]

    df = df.drop(columns=columns_to_delete)
    df = df.rename(columns={'KEIM': 'species'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')

    args = parser.parse_args()

    clean_data(args.INPUT)
