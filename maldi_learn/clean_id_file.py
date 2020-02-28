#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd


def clean_data(filename, outfile):
    df = pd.read_csv(filename, low_memory=False, encoding='utf8')
    print(f'\nInput file: {filename}')
    print(f'ID file starting shape: {df.shape}')

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
    print(f'Remove columns: {columns_to_delete}')    
    
    df = df.drop(columns=columns_to_delete)     # remove obsolete columns
    df = df.dropna(subset=['code'])             # remove missing codes
    df = df.drop_duplicates()                   # drop full duplicates
    print(f'ID file shape after basic clean-up: {df.shape}')

    duplicate_codes = df[df.duplicated('code')]['code'].values
    df = df.drop_duplicates(subset=['code'], keep=False) # remove entries with duplicated ids    
    print(f'Number of non-unique codes: {len(duplicate_codes)}')
    print(f'ID file final shape: {df.shape}')

    df = df.rename(columns={
        'KEIM': 'species',
        'Organism(best match)': 'bruker_organism_best_match',
    })

    
    ab_name_map = {
        'Amoxicillin...Clavulansaeure.bei.unkompliziertem.HWI': 'Amoxicillin-Clavulans.unkompl.HWI',
        'Ampicillin...Amoxicillin': 'Ampicillin-Amoxicillin',
        'Piperacillin...Tazobactam': 'Piperacillin-Tazobactam',
        'Amoxicillin...Clavulansaeure': 'Amoxicillin-Clavulanic acid',
        'Fusidinsaeure': 'Fusidic acid',
        'Ceftazidim.1': 'Ceftazidime',
        'Ceftazidim.Avibactam': 'Ceftazidime-Avibactam',
        'X5.Fluorocytosin': '5-Fluorocytosin',
        'Fosfomycin.Trometamol': 'Fosfomycin-Trometamol',
        'Ceftolozan...Tazobactam': 'Ceftolozane-Tazobactam',
        'Cefepim': 'Cefepime',
        'Posaconazol': 'Posaconazole',
        'Tigecyclin': 'Tigecycline',
        'Cefpodoxim': 'Cefpodoxime',
        'Ceftobiprol': 'Ceftobiprole',
        'Fluconazol': 'Fluconazole',
        'Cefuroxim': 'Cefuroxime',
        'Tetracyclin': 'Tetracycline',
        'Ceftriaxon': 'Ceftriaxone',
        'Itraconazol': 'Itraconazole',
        'Cotrimoxazol': 'Trimethoprim-Sulfamethoxazole',
        'Minocyclin': 'Minocycline',
        'Voriconazol': 'Voriconazole',
        'Metronidazol': 'Metronidazole',
        'Aminoglykoside': 'Aminoglycosides',
        'Chinolone': 'Quinolones',
        'Doxycyclin': 'Doxycycline',
        'Cefixim': 'Cefixime',
    }


    # TODO assert no duplicates in code

    # TODO rename columns to standard antibiotic names
    df = df.rename(columns=ab_name_map)

    df.to_csv(outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')
    parser.add_argument('OUTPUT', type=str, help='Output file')

    args = parser.parse_args()

    clean_data(args.INPUT, args.OUTPUT)
