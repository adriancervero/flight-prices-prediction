#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Build new features into the dataset
    @author: Adrián Cerveró
"""


import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from pathlib import Path
import os, sys

import config as cfg
from tqdm import tqdm
tqdm.pandas()

def fill_missing(df):
    """
        Fills in the missing days by duplicating the flights of the previous day.
        
        Args:
            - df: Dataframe with flights data
        Returns: 
            - Same dataframe with news added rows with no missing days.
    """
    collectionDates = df['collectionDate'].unique()
    dates_range = pd.date_range(collectionDates[0], collectionDates[-1]).tolist()
    dates_range = [str(date).split()[0] for date in dates_range]
    missing_dates = [date for date in dates_range if date not in collectionDates]
    missing_dates_dt = [datetime.strptime(date, '%Y-%m-%d') for date in missing_dates]
    previous_dates_dt = [date + timedelta(days=-1) for date in missing_dates_dt]
    previous_dates = [datetime.strftime(date, '%Y-%m-%d') for date in previous_dates_dt]
    
    for idx, date in enumerate(missing_dates):
        previous_date = previous_dates[idx]
        this_date = df[df['collectionDate'] == previous_date].copy()
        this_date['collectionDate'] = date
        df = df.append(this_date)
    return df

def add_dDate_features(df):
    """ New features extracted from departure date """
    # Day of month
    df['day_of_month'] = pd.to_datetime(df['dDate']).dt.day
    # Day of the week
    df['day_of_week'] = pd.to_datetime(df['dDate']).apply(lambda x: x.day_of_week)
    days_of_week = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
    df['day_of_week'] = df['day_of_week'].map(days_of_week)
    return df

def add_days_until_dep_feature(df):
    """ Remaining days until flight departure """
    collected = pd.to_datetime(df['collectionDate'])
    departure =  pd.to_datetime(df['dDate'])
    daysUntilDep = departure - collected
    df['days_until_dep'] = daysUntilDep.apply(lambda x: str(x).split()[0]).astype(int)
    return df

def add_competition_feature(df):
    """ Competition factor represents how many flights are for a given day 
        for the same itinerary """
    competition = df.groupby(['flyFrom','flyTo','dDate'])['airline'].nunique().reset_index()
    competition.columns = ['flyFrom','flyTo','dDate', 'competition']
    df = pd.merge(df, competition, on=['dDate', 'flyFrom', 'flyTo'])
    return df

def add_hist_prices(df, progress_bar=True):
    """ 
        New feature 'hist_prices' consisting of a historical price list for each flight.
        
        Args: 
            - df: Dataframe with flights data.
            - progress_bar: If enabled, it displays a progress bar during execution.
            
        Returns: 
            - Same dataframe with the new column 'hist_prices'
    """
    print("...Adding 'hist_prices' feature...", end='\n')
    sorted_by_date = df.sort_values(by='collectionDate')
    if progress_bar: 
        grouped = sorted_by_date.groupby(['id'])['price'].progress_apply(list)
    else:
        grouped = sorted_by_date.groupby(['id'])['price'].apply(list)
    grouped = grouped.reset_index(name='hist_prices')
    merged = pd.merge(df, grouped, on='id')
    return merged

def build_features(df):
    """ Add new features """

    #  new features from 'price' column
    df['log_price'] = np.log(df['price'])
    
    # new features from 'dDate' column
    df = add_dDate_features(df)
    # Day session
    df['session'] = pd.cut(pd.to_datetime(df['dTime']), bins=4, labels=['night', 'morning', 'afternoon', 'evening'])
    # Route
    df['orig-dest'] = df['flyFrom']+'-'+df['flyTo']
    # Airline
    df['airline'] = df['airlines'].apply(lambda x: x.split(',')[0])
    # Days until departure 
    df = add_days_until_dep_feature(df)
    # Hopping
    df['hops'] = df['route'].apply(lambda x: len(x.split('->')) - 2)
    df['direct'] = df['hops'] == 0

    # Competition factor
    df = add_competition_feature(df)

    # id flight
    df['id'] = df.groupby(['dDate', 'flyFrom', 'flyTo', 'dTime', 'aTime', 'airline', 'fly_duration']).ngroup()

    # Historical prices for each flight
    df = add_hist_prices(df)
    # TODO: features from hist_prices; min, max, mean, quartiles...
    return df

def filter_flights(df):
    """ Flight filtering to select only valid flights. This is, flights which 
        we have the price for each of the days until the departure of the flight
        
        Args:
            - df: Dataframe with flights data
        Returns
            - new_df: New dataframe with only valid flights.
    """
    max_days = df.groupby('id')['days_until_dep'].transform(max)
    hist_lengths = df['hist_prices'].apply(len)
    new_df = df[max_days == hist_lengths].copy()
    return new_df

def add_waiting_days(df):
    """
        Add new feature 'waiting_days' that indicates the days to wait to 
        get the best price among the remaining days until flight departure. 
        This will be the target variable.
        
        Args: 
            - df: Dataframe with flights data
        Returns
            - Same dataframe with the new target: 'waiting_days'
    """
    
    waiting_days_list = np.array([])
    for row in tqdm(df.itertuples(), total=df.shape[0]):
        current_price = row.price
        hist = row.hist_prices
        days_until_dep = int(row.days_until_dep)
        idx = len(hist)-days_until_dep
        if days_until_dep > 1:
            next_days_prices = hist[idx+1:]
            idx_min = np.argmin(next_days_prices)
            min_price = next_days_prices[idx_min]
            if min_price < current_price:
                waiting_days = idx_min+1
            else:
                waiting_days = 0
        else:
            waiting_days = 0
       # print(row.id, days_until_dep, row.price, next_days_prices, waiting_days)
        waiting_days_list = np.append(waiting_days_list, waiting_days)
    df['waiting_days'] = waiting_days_list.astype(int)
    return df

def split_data(df, test_days=14):
    flight_dates = pd.to_datetime(df['dDate'])
    split_date = flight_dates.max() - timedelta(days=test_days)
    train = df[flight_dates <= split_date].copy()
    test = df[flight_dates > split_date].copy()
    return train, test

def store_data():
    """ Store dataframe with new features in data processed folder """
    store_path = str(Path(filename).parent.parent) + PROCESSED_DATA_PATH
    df.to_csv(store_path, index=False)
    return store_path

def preprocessing(filename, verbose):
    """[summary]

    Args:
        filename (str): [description]
    """
    
    if verbose:
        print('Starting preprocessing...')
        print('...Loading data...', end='\r')

    os.chdir(sys.path[0])
    df = pd.read_csv(filename)

    # drop 'seats' because has many nan
    df.drop('seats', axis=True, inplace=True)
    
    # missing data
    df = fill_missing(df)

    if verbose:
        print('...Adding new features...')

    # adding new features
    df = build_features(df)

    # filter flights
    df = filter_flights(df)

    if verbose:
        print('...Adding target variable...')
    # add target 'waiting_days'
    df = add_waiting_days(df)

    print('\nPreprocessing done!')
    
    # split data in train and test sets
    train, test = split_data(df, test_days=cfg.TEST_DAYS)

    # Store data in processed folder
    store_path = str(Path(filename).parent.parent) + cfg.TRAIN_PROCESSED
    train.to_csv(store_path, index=False)
    print('\nTrain data stored successfully!:', store_path)
    store_path = str(Path(filename).parent.parent) + cfg.TEST_PROCESSED
    test.to_csv(store_path, index=False)
    print('Test data stored successfully!:', store_path)
    

if __name__ == '__main__':
    preprocessing(cfg.INTERIM_DATA_PATH, verbose=True)
    
    
