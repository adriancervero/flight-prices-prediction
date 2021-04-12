#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
      Configuration variables
"""


# Data paths
RAW_DATA = '../data/raw/flights_raw.csv'
INTERIM_DATA_PATH = '../data/interim/flights_interim.csv'
TRAIN_PROCESSED = "../data/processed/train.csv"
TEST_PROCESSED = "../data/processed/test.csv"
TRAIN_FOLDS = "../data/processed/train_folds.csv"

# Day Range to use in test set when splitting
TEST_DAYS = 14

# Number of folds for cross validation
N_FOLDS = 5

# Feature selection
NUMERICAL = ['days_until_dep', 'fly_duration', 'day_of_month', 'log_price', 'hops', 'competition']
CATEGORICAL = ['flyFrom', 'flyTo', 'day_of_week', 'session']
TARGET = ['waiting_days']

# MODEL OUTPUT
MODEL_OUTPUT = '../models/'

