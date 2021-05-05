#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import pickle
import argparse
import os, sys

import config as cfg
from Simulator import Simulator


def evaluator(model, n_travelers):
    os.chdir(sys.path[0])
    # od
    pipeline_path = cfg.MODEL_OUTPUT + model
    pipeline = pickle.load(open(pipeline_path, 'rb'))
    test_data = pd.read_csv(cfg.TEST_PROCESSED)
    
    sim = Simulator(n_travelers, test_data, pipeline)
    sim.run()

if __name__ == '__main__':
    # ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('-n', type=int, default=5000)

    # read arguments from command line
    args = parser.parse_args()

    evaluator(
        model=args.model,
        n_travelers=args.n
    )