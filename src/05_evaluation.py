#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import pickle
import argparse
import os, sys

import config as cfg
from Simulator import Simulator

def evaluator(model):
    os.chdir(sys.path[0])

    pipeline_path = cfg.MODEL_OUTPUT + f'pipeline_{model}.pkl'
    pipeline = pickle.load(open(pipeline_path, 'rb'))
    test_data = pd.read_csv(cfg.TEST_PROCESSED)
    sim = Simulator(5000, test_data, pipeline)
    sim.run()

if __name__ == '__main__':
    # ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)

    # read arguments from command line
    args = parser.parse_args()

    evaluator(
        model=args.model
    )