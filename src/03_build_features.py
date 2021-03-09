#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Build new features into the dataset
    @author: Adrián Cerveró
"""

INTERIM_DATA_PATH = '../data/interim/flights_interim.csv'
PROCESSED_DATA_PATH = '/processed/flights_processed.csv'

import pandas as pd
import numpy as np
from pathlib import Path
import os, sys

def get_performed_flights(flights):
    departure_dates = pd.to_datetime(flights['dDate'])
    max_collection_day = pd.to_datetime(flights['collectionDate']).max()                    
    pf = flights[departure_dates <= max_collection_day]
    return pf 

def build_features(filename, pf=False):

    os.chdir(sys.path[0])
    df = pd.read_csv(filename)

     # log transformation on target
    df['log_price'] = np.log(df['price'])
    # Day of month
    df['day_of_month'] = df['dDate'].apply(lambda x: int(x.split('-')[2]))

    # Day of the week
    df['day_of_week'] = pd.to_datetime(df['dDate']).apply(lambda x: x.day_of_week)
    days_of_week = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
    df['day_of_week'] = df['day_of_week'].map(days_of_week)

    # Session (morning, afternoon, night)
    df['session'] = pd.cut(pd.to_datetime(df['dTime']), bins=4, labels=['night', 'morning', 'afternoon', 'evening'])

    # Route
    df['orig-dest'] = df['flyFrom']+'-'+df['flyTo']
    
    # Airline
    df['airline'] = df['airlines'].apply(lambda x: x.split(',')[0])
    
    # Days until Departure
    collected = pd.to_datetime(df['collectionDate'])
    departure =  pd.to_datetime(df['dDate'])
    daysUntilDep = departure - collected
    df['days_until_dep'] = daysUntilDep.apply(lambda x: str(x).split()[0])
    
    # Hopping
    df['hops'] = df['route'].apply(lambda x: len(x.split('->')) - 2)
    df['direct'] = df['hops'] == 0
    
    # Competition Factor
    competition = df.groupby(['flyFrom','flyTo','dDate'])['airline'].nunique().reset_index()
    competition.columns = ['flyFrom','flyTo','dDate', 'competition']
    df = pd.merge(df, competition, on=['dDate', 'flyFrom', 'flyTo'])
    
    # Store data
    store_path = str(Path(filename).parent.parent) + PROCESSED_DATA_PATH
    df.to_csv(store_path, index=False)
    return store_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data raw >> Data preprocessed')
    parser.add_argument('-path', type=str, help='path to the raw data file', default=INTERIM_DATA_PATH)
    parser.add_argument('--pf',
                        action='store_const',
                        const=True,
                        default=False,
                        #type=bool,
                        help='indicates if using only performed flights')
    args = parser.parse_args()

    filename = args.path
    print(build_features(filename,pf=args.pf))
