#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Script to run all the steps in the project workflow
"""

import importlib

# Importing our modules
# Since we have digits at the start of the modules we
# use dynamic import
prepare_data = importlib.import_module('.data.01_prepare_data', 'src')


#-------------------------------------------------------------------
# Variables 
# ------------------------------------------------------------------
RAW_DATA_PATH = 'data/raw/flights_raw.csv'

# TODO: Route Selection? Options?

# ------------------------------------------------------------------

if __name__ == '__main__':
    data_prepared = prepare_data.prepare_data(RAW_DATA_PATH)
    print('done!', data_prepared) 