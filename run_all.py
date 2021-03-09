#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Script to run all the steps in the project workflow
"""

import importlib
import argparse

# Importing our modules
# Since we have digits at the start of the modules we
# use dynamic import
prepare_data = importlib.import_module('.01_prepare_data', 'src')
build_features = importlib.import_module('.03_build_features', 'src')


#-------------------------------------------------------------------
# Variables 
# ------------------------------------------------------------------
RAW_DATA_PATH = 'data/raw/flights_raw.csv'
INTERIM_DATA_PATH = 'data/interim/flights_interim.csv'
PROCESSED_DATA_PATH = 'data/processed/flights_processed.csv'

# TODO: Route Selection? Options?

# ------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='bla bla bla')
    parser.add_argument('-path', type=str, help='path to the raw data file', default=RAW_DATA_PATH)
    parser.add_argument('--pf',
                        action='store_const',
                        const=True,
                        default=False,
                        #type=bool,
                        help='indicates if using only performed flights')
    args = parser.parse_args()


    print('Starting data workflow...')
    data_prepared = prepare_data.prepare_data(args.path)
    print('STEP 1 - Data Preparation: done!', data_prepared) 
    data_preprocessed = build_features.build_features(INTERIM_DATA_PATH, pf=args.pf)
    print('STEP 2 - Building Features: done!', data_preprocessed)
    print('Finished!')
