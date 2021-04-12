#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Create stratified folds for cross validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os, sys

import config as cfg

def create_folds(data, n):
    # new column called kfold and fill it with -1
    data['kfold'] = -1

    # shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    
    # get number of bins using Sturge's rule
    #num_bins = int(np.floor(1+np.log2(len(data))))
    num_bins = 10

    # bin targets
    data.loc[:, 'bins'] = pd.cut(
        data['waiting_days'], bins=num_bins, labels=False
    )

    # init kfold class
    skf = StratifiedKFold(n_splits=n)

    # fill the kfold column 
    for n, (train_idx, valid_idx) in enumerate(skf.split(X=data, y=data['bins'].values)):
        data.loc[valid_idx, 'kfold'] = n
    
    data = data.drop('bins', axis=1)
    return data


if __name__ == "__main__":    
    # Load training data
    os.chdir(sys.path[0])
    train = pd.read_csv(cfg.TRAIN_PROCESSED)
    
    # create folds
    train_folds = create_folds(train, cfg.N_FOLDS)

    # store data with folds column
    train_folds.to_csv(cfg.TRAIN_FOLDS, index=False)

    

