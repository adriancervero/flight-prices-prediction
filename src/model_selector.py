#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Model Selector: dictionary with models for training module
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import numpy as np

import config as cfg

models = {
    'log_reg': LogisticRegression(max_iter=500),
    "rf": RandomForestClassifier(max_depth=6, random_state=cfg.RANDOM_STATE),
    'svm': SVC(probability=True), # no converge
    'knn':KNeighborsClassifier(n_neighbors=4),
    'xgb':XGBClassifier(),
}


param_grid = {
    "log_reg":{
        "C":[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        "solver": ['newton-cg', 'lbfgs'],
    },
    "rf":{
        "n_estimators": np.arange(10, 500, 10),
        "max_depth": np.arange(1, 9),
        "criterion": ["entropy", "gini"],
        "max_features": ['auto','sqrt'],
        "min_samples_split": [2, 5, 8, 10, 20],
        "min_samples_leaf":[1, 2, 4, 5, 6],
        
    },
    "xgb":{
        "model__eta": [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
        "model__gamma": np.arange(0, 5, 0.1),
        "model__max_depth": np.arange(1, 8),
        #"model__delta_step": np.arange(1, 10),
        "model__subsample": np.arange(0.5, 1, 0.1),
        

    }
    
}