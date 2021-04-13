#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    
"""

import pandas as pd
import numpy as np
import argparse
import pickle

# Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.model_selection import cross_val_score
# My imports
import config as cfg
import model_selector

def feature_selection(df):
    num_attribs = cfg.NUMERICAL
    cat_attribs = cfg.CATEGORICAL
    target = cfg.TARGET
    return df[num_attribs + cat_attribs + target + ['kfold']]

def create_pipeline_old():
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, cfg.NUMERICAL),
        ('cat', cat_pipeline, cfg.CATEGORICAL),
    ])

    return full_pipeline

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

    # Init model
    m = model_selector.models[model]

    # Create pipeline
    pipeline = create_pipeline(m)


    X = df.drop(['waiting_days','kfold'], axis=1)
    y = df['waiting_days'].values

    # Training using Cross Validation     
    scores = cross_val_score(
        pipeline, X, y, cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
    )

    print('Score:', -scores.mean())

    # Save model and pipeline

    pipeline.fit(X, y)

    pickle.dump(m, open(f'../models/model_{model}.pkl', 'wb'))
    pickle.dump(pipeline, open(f'../models/pipeline_{model}.pkl', 'wb'))



if __name__ == '__main__':
    # ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)

    # read arguments from command line
    args = parser.parse_args()

    run(
        model=args.model
    )