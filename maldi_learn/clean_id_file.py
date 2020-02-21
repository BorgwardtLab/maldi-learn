#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd


def clean_data(filename):
    df = pd.read_csv(filename, low_memory=False, encoding='utf8')

    # Select columns for deletion. We want to remove columns that could
    # potentially leak patient information.
    columns_to_delete = [
        'strain',
        'TAGESNUMMER',
        'Value',
        'A',
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

    df = df.drop(columns=columns_to_delete)     # remove obsolete columns
    df = df.dropna(subset=['code'])             # remove missing codes
    df = df.drop_duplicates()                   # drop full duplicates
    df = df.drop_duplicates(subset=['code'], keep=False) # remove entries with duplicated ids    


    df = df.rename(columns={
        'KEIM': 'species',
        'Organism(best match)': 'bruker_organism_best_match',
    })

    duplicate_codes = df[df.duplicated('code')]['code'].values
    
    for code in duplicate_codes:
        rows = df.loc[df.code == code]
        
        #print(set(rows['species'].values))

    #print(df[df.duplicated('code', keep=False)])

    #print(np.argwhere(df.index.isna()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')

    args = parser.parse_args()

    clean_data(args.INPUT)
