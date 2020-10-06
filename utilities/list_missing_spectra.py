#!/usr/bin/env python3
#
# Given an ID file of the DRIAMS data set and a root directory, checks
# the directory for missing spectra and lists their IDs.

import argparse
import os

import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')
    parser.add_argument('ROOT', type=str, help='Root directory')

    args = parser.parse_args()

    df = pd.read_csv(args.INPUT, low_memory=False)

    # Will contain the codes of all spectra. For simplicity, we assume
    # that everything that ends with '.txt' is a spectrum. This is not
    # a problem, though, because we are just checking whether the code
    # that we are looking for is part of that list; additional entries
    # do not matter.
    all_spectra = []

    for root, dirs, files in os.walk(args.ROOT):
        for filename in files:
            if os.path.splitext(filename)[1] == '.txt':
                all_spectra.append(
                    os.path.splitext(filename)[0]
                )

    # This will significantly speed up the subsequent queries because
    # we are not dealing with a list any more. 
    all_spectra = set(all_spectra)

    for code in df['code']:
        if code not in all_spectra:
            print(code)
