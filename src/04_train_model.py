#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Train a model using my PriceEstimatorTrainer class
    then evaluates on valid or test. Return results by route.

    Args:
        - model : {'rf', 'log_reg', 'knn', 'svm', 'xgb'}, default='rf' 
                Model to train the data with (It must be 
                in the model selector dictionary)

        - n_iter : int, default=0
                Number of iterations for random search params

        - -m, comment : str 
                Explanatory notes about current experiment, doesn't affect
                training process

        - testset : {'valid', 'test'}, default='valid' 
                Whether to use the validation set or the test set
                for evaluate the trained model.

        - -o, output : str
                Name to store the model. If no passed, model will not be
                stored.

    @author: Adrián Cervero - May 2021
    @github: https://github.com/adriancervero/flight-prices-prediction 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import logging
import os, sys

# My imports
import config as cfg
import model_selector
from PriceEstimatorTrainer import PriceEstimatorTrainer

def generate_model(param_grid):
    """ Returns given model with random params from param_grid parameter """
    n_estimators = np.random.choice(param_grid['n_estimators'])
    max_depth = np.random.choice(param_grid['max_depth'])
    criterion = np.random.choice(param_grid['criterion'])
    max_features = np.random.choice(param_grid['max_features'])
    min_samples_split = np.random.choice(param_grid['min_samples_split'])
    min_samples_leaf = np.random.choice(param_grid['min_samples_leaf'])

    model = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    criterion=criterion,
                                    max_features=max_features,
                                    min_samples_split=min_samples_split,
                                    random_state=cfg.RANDOM_STATE)
    return model




def run(model, n_iter, comment, testset, output):
    """ Performs training process """

    print("\n----- 04 - Training Model -----\n\n")

    # load training data 
    os.chdir(sys.path[0])
    df = pd.read_csv(cfg.TRAIN_PATH)

    # Init model and get params for hypertuning
    m = model_selector.models[model]
    if model == 'rf':
        param_grid = model_selector.param_grid[model]

    #threshold_values = [0.4, 0.42, 0.44,0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6]
    #for threshold in threshold_values:

    if n_iter == 0: # no hypertunning
        trainer = PriceEstimatorTrainer(model=m, pred_threshold=.5, n=10000)
    
        # fit model
        trainer.fit()

        # print results on validation
        acc, mean_savings = trainer.evaluate(on=testset, plot=True)
    
        logging.info(f"Precision: {acc}, Mean Savings %: {mean_savings}, Notes:{comment}, Model:{m}")
        logging.info("---------------------------------------------------------------------------------------")

        if output != '':
            output_path = cfg.MODEL_OUTPUT + output
            pickle.dump(trainer.model, open(output_path, 'wb'))
            print('\n\nmodel stored: ', output_path)
            
    else: # hypertunning

        for _ in range(0, n_iter):
            m = generate_model(param_grid)
            trainer = PriceEstimatorTrainer(model=m)
    
            # fit model
            trainer.fit()

            # print results on validation
            acc, mean_savings = trainer.evaluate(on=testset)
        
            logging.info(f"Precision: {acc}, Mean Savings %: {mean_savings}, Notes:{comment}, Model:{m}")
            logging.info("---------------------------------------------------------------------------------------")
        

    

if __name__ == '__main__':
    # ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rf') # model to use
    parser.add_argument('-t', type=int, default=0) # random search hypertuning iterations
    parser.add_argument('-m', type=str, default='') # notes
    parser.add_argument('--on', type=str, default='valid') # evaluate on validation or test
    parser.add_argument('-o', type=str, default='') # model save name

    # read arguments from command line
    args = parser.parse_args()
    # logger
    logging.basicConfig(level=logging.DEBUG, filename="training_logs", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    run(
        model=args.model,
        n_iter = args.t,
        comment=args.m,
        testset=args.on,
        output=args.o,
    )