#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    
"""

import pandas as pd
import numpy as np
import argparse
import pickle
import logging
import json

# Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
# My imports
import config as cfg
import model_selector

def feature_selection(df):
    num_attribs = cfg.NUMERICAL
    cat_attribs = cfg.CATEGORICAL
    target = cfg.TARGET
    return df[num_attribs + cat_attribs + target + ['kfold']]

def create_pipeline(model):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, cfg.NUMERICAL),
        ('cat', cat_pipeline, cfg.CATEGORICAL),
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline

def run(model):
    # load training data 
    df = pd.read_csv(cfg.TRAIN_FOLDS)

    # selecting features for training
    df = feature_selection(df)

    # Init model and get params for hypertuning
    m = model_selector.models[model]
    param_grid = model_selector.param_grid[model]

    # Create pipeline
    pipeline = create_pipeline(m)


    X = df.drop(['waiting_days','kfold'], axis=1)
    y = df['waiting_days'].values

    # Training using Cross Validation
    """     
    scores = cross_val_score(
        pipeline, X, y, cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
    )
    print('Score:', -scores.mean())
    """
    # Training with RandomSearch
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=1,
        scoring='neg_root_mean_squared_error',
        verbose=2,
        n_jobs=1,
        cv=5,
        random_state=42,
    )
    random_search.fit(X, y)

    # print results

    """
    results = f"\nModel: {m.__class__.__name__}\n" \
        + f"Best score: {-random_search.best_score_}\n" \
        + "Best parameters:\n"

    for param_name in sorted(param_grid.keys()):
        results += f"\t{param_name}: {best_parameters[param_name]}\n"

    print(results)
    """
    best_parameters = random_search.best_estimator_.get_params()
    results = {
        'Model': m.__class__.__name__,
        'Best score':-random_search.best_score_,
        'Best parameters': [f"{param_name}: {best_parameters[param_name]}" for param_name in param_grid.keys()],
        'Features': list(df.columns)
    }
    results_json = json.dumps(results, indent=4)
    print(results_json)
    logging.info(results_json)

    # Save best model and pipeline
    best_model = random_search.best_estimator_
    pickle.dump(best_model, open(f'../models/pipeline_{model}.pkl', 'wb'))
    
    

if __name__ == '__main__':
    # ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)

    # read arguments from command line
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, filename="logs", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")


    run(
        model=args.model
    )