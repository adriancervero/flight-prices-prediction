#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
      Configuration variables
"""

RANDOM_STATE = 42

# Data paths
RAW_DATA_PATH = '../data/raw/flights_raw.csv.gz'
INTERIM_DATA_PATH = '../data/interim/flights_interim.csv'
TRAIN_PATH = "../data/processed/train.csv"
TEST_PATH = "../data/processed/test.csv"
VALID_PATH = "../data/processed/valid.csv"
FIGURES_PATH = "../figures/"
PRICE_BINS_PATH = '../data/processed/price_bins.csv'
BINS_DAYS_PATH = '../data/processed/bins_days.csv'

### Preprocessing variables ####
# Day Range to use in test set when splitting
SPLIT_TEST_DAYS = 30
# feature to use to combine price
COMBINE_PRICE_FEATURE = 'customPrice'
# last days trend to compute customPrice and weight of this trend
LAST_DAYS = 20
W = .8
# percentage of the minimum price drop to take into account to consider it worthwhile to wait
MIN_DROP_PER_TRAIN = 0.05
MIN_DROP_PER_TEST = 0.00
# feature used for agg. flights
AGG_COLS = ['orig-dest', 'airline', 'session', 'days_until_dep']

### Model Training variables ###

# Feature selection
# categorical
CATEGORICAL = ['orig-dest', 'airline', 'session']

# numerical
NUMERICAL = ['days_until_dep', 'competition', 'prob']

# target
TARGET = 'wait'

# MODEL OUTPUT
MODEL_OUTPUT = '../models/'