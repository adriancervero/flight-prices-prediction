#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Prepare data for model training:
        1. Missing values treatment
        2. Adding new features
        3. Split data in train and test
        4. Agg. flights data
        5. Label data (wait column)
        6. Create price_bins datafrom for estimate days to wait

        Input: data/interim
        Output: data/processed

        @author: Adrián Cervero - May 2021
        @github: https://github.com/adriancervero/flight-prices-prediction
"""

#-------------------------------------------------------------------
# Imports 
# ------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import datetime, timedelta
import os, sys
import argparse
from tqdm import tqdm
tqdm.pandas()

import config as cfg

import warnings
warnings.filterwarnings('ignore')


def load_data(path):
    """ Load data and return a Pandas dataframe. """
    print('...loading data from .csv...')
    
    os.chdir(sys.path[0]) # relative path to this file
    df = pd.read_csv(path)
    return df

def cleaning_df(df):
    """ Drop  irrelevant columns and outliers """

    # outliers
    Q1 = df['price'].quantile(.25)
    Q3 = df['price'].quantile(.75)
    IQR = Q3-Q1
    df = df[df['price'] < IQR*1.5 ]

    # columns to drop
    drop_cols = ['seats', 'dTimeUTC', 'aTimeUTC', 'flight_no',
            'cityFrom', 'cityCodeFrom', 'cityTo', 'cityCodeTo', 'countryFrom',
             'countryTo']
    return df.drop(drop_cols, axis=1, )


def add_days_until_dep_feature(df):
    """ Remaining days until flight departure """
    collected = pd.to_datetime(df['collectionDate'])
    departure =  pd.to_datetime(df['dDate'])
    daysUntilDep = departure - collected
    df['days_until_dep'] = daysUntilDep.dt.days
    return df

def build_features(df):
    """ Add new features """

    # orig and dest in same column
    df['orig-dest'] = df['flyFrom']+'-'+df['flyTo']
    # departure time in three categories: 'morning', 'evening', 'night'
    df['session'] = pd.cut(pd.to_datetime(df['dTime']), bins=3, labels=['night', 'morning', 'evening'])
    # Departure day of week
    days_of_week = {5:'Monday', 6:'Tuesday', 0:'Wednesday', 1:'Thursday', 2:'Friday', 3:'Saturday', 4:'Sunday'}
    df['day_of_week'] = pd.to_datetime(df['dTime']).dt.weekday.map(days_of_week)
    # airline
    df['airline'] = df['airlines'].str.split(',').apply(lambda x: x[0])
    # days until departure
    df = add_days_until_dep_feature(df)
    # price log-transform
    df['log_price'] = np.log(df['price'])
    # number of stops between origin and destinations
    df['hops'] = df['route'].str.split('->').apply(len)-2

    return df


def split_data(df, test_days=30):
    """ Split data in train/test set """

    collection_dates = pd.to_datetime(df['collectionDate'])
    departure_dates = pd.to_datetime(df['dDate'])
    
    split_date = collection_dates.max() - timedelta(days=test_days)
    test_idx = (collection_dates >= split_date) & (departure_dates <= collection_dates.max())
    test = df[test_idx]
    train = df[~test_idx]

    return train, test

def q25(x):
    """ Return first quantile of x """
    return x.quantile(0.25)

def get_labels(row):
    """ 
    Assign wait or buy label. If price decrease in next days
    label 1 otherwise label 0. The decrease has to exceed a threshold 
    (MIN_DROP_PER variable)
    """
    current_d = row['days_until_dep']
    current_price = row[cfg.COMBINE_PRICE_FEATURE]
    list_prices = np.array(row['list_prices'])
    next_days = list_prices[:current_d-1]
    if len(next_days) == 0:
        return 0
    else:
        min_price = np.min(next_days)

        if min_price < current_price and 1-(min_price/current_price) > cfg.MIN_DROP_PER_TRAIN:
            return 1
        else:
            return 0

def get_last_days_q25(row):
    """ Compute first quantile of last x days for each group """
    list_prices = row['list_prices']
    start_idx = row['days_until_dep']
    end_idx = start_idx + cfg.LAST_DAYS

    last_days_prices = list_prices[start_idx:end_idx]
    if last_days_prices == []:
        last_days_prices = list_prices[-1]
    return np.quantile(last_days_prices, 0.25)

def create_price_bins(train):
    """ Return dataframe with estimate prices for each days until dep value """
    # Creation of bins
    lower = train['days_until_dep'].min()
    higher = train['days_until_dep'].max()

    n_bins = int(higher/5)
    edges = range(lower, higher+5, 5)

    lbs = ['(%d, %d]'%(edges[i], edges[i+1]) for i in range(len(edges)-1)]
    train.loc[:, 'days_bins'] = pd.cut(train['days_until_dep'],bins=n_bins, labels=lbs)

    price_bins = train.groupby(['orig-dest', 'airline', 'days_bins'])['price'] \
                    .quantile(.25).rename('price_est').reset_index().dropna()
    
    # prices
    # plotting bins prices
    plt.figure(figsize=(8,4))
    fig, axes = plt.subplots(1, 1, figsize=(8,4))
    sns.boxplot(x='days_bins', y='price_est', data=price_bins, palette="Blues_r");
    plt.xticks(rotation=45);
    plt.xlabel('Days until departure')
    plt.ylabel('Price Estimated')

    fig.savefig(cfg.FIGURES_PATH+'est_prices.png', pad_inches=0.5, bbox_inches='tight')

    bins_days = train[['days_until_dep', 'days_bins']].drop_duplicates()
    bins_days['days_bins'] = bins_days['days_bins'].cat.codes

    # save dataframes
    price_bins.to_csv('../data/processed/price_bins.csv', index=False)
    bins_days.to_csv('../data/processed/bins_days.csv', index=False)

def agg_flights(train):
    """ 
    Grouping fligths by previously defined agg. cols. 
    (orig-dest, airline, session, days_until_dep), adding
    new features and label data with wait/buy

    """
    agg_cols = cfg.AGG_COLS

    grouped = train \
                .groupby(agg_cols)[['price', 'fly_duration']] \
                .agg({'price':['min', 'median', q25, 'count'],
                    'fly_duration':'mean'}).dropna().reset_index()

    # remove multilevel columns
    grouped.columns = agg_cols + ['min', 'median', 'q25', 'count', 'fly_duration']

    # count will act like some kind of competition factor
    grouped.rename(columns={'count':'competition'}, inplace=True)

    # Feature 'total_q25': First quantile of each group during all time period
    total_q25 = train.groupby(agg_cols)['price'].quantile(0.25) ######################### Revisar
    total_q25 = total_q25.rename('total_q25').reset_index()
    grouped['total_q25'] = pd.merge(grouped, total_q25, on=agg_cols, how='left')['total_q25']

    # lastdays_q25: First quantile of last n days 
    list_prices = grouped.groupby(['orig-dest','airline', 'session'])['q25'].agg(list)
    list_prices = list_prices.reset_index()
    list_prices.rename(columns={'q25':'list_prices'}, inplace=True)
    grouped['list_prices'] = pd.merge(grouped, list_prices, on=['orig-dest','airline','session'], how='left')['list_prices']

    grouped['lastdays_q25'] = grouped.progress_apply(get_last_days_q25, axis=1)

    # CustomPrice = ticket price weightened considering last days trend
    # We use this feature for estimate labels
    grouped['customPrice'] =  cfg.W * grouped['total_q25'] + (1-cfg.W) * grouped['lastdays_q25']
    
    # target
    grouped['wait'] = grouped.progress_apply(get_labels, axis=1)
    print(grouped['wait'].value_counts())

    # remove no needed columns
    grouped.drop(['total_q25', 'list_prices', 'lastdays_q25'], axis=1, inplace=True, errors='ignore')

    grouped.rename(columns={'q25':'price'}, inplace=True)

    # prob feature: percentage of wait flights in a group
    probs = grouped.groupby(['orig-dest','session' ,'days_until_dep'])['wait'].mean().reset_index()
    probs.rename(columns={'wait':'prob'}, inplace=True)
    grouped = pd.merge(grouped, probs, on=['orig-dest', 'session' ,'days_until_dep'], how='left')

    # plot prob and save it
    wait_grouped = grouped.groupby(['orig-dest','days_until_dep'])['wait'].mean().reset_index()
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    sns.scatterplot(x='days_until_dep', y='wait', hue='wait', data=wait_grouped)
    plt.xlabel('Days until departure');
    plt.ylabel('Wait %');

    fig.savefig(cfg.FIGURES_PATH + "prob_feature.png", pad_inches=0.5, bbox_inches='tight')

    return grouped

def filter_flights(df):
    routes = ['MAD-EZE', 'MAD-MEX', 'MAD-JFK']
    df = df[(df['orig-dest'].isin(routes))]
    return df


def preprocessing(filename, min_drop_per):
    """
        Prepare data for model training:
            1. Missing values treatment
            2. Adding new features
            3. Split data in train, valid and test
            4. Aggregate train flights
            5. Label data (wait column)
            6. Create price_bins datafrom for estimate days to wait


        Args:
            filename (str): input data path
    """
    print("\n----- 03 - Feature Engineering -----")
    
    cfg.MIN_DROP_PER_TRAIN = min_drop_per

    # load data
    df = load_data(cfg.INTERIM_DATA_PATH)

    # missing values and remove no needed cols
    df = cleaning_df(df);
    
    
    print('...Adding new features...')

    # adding new features
    df = build_features(df)

    #df = filter_flights(df)

    # split data in train, valid and test
    train, test = split_data(df, test_days=cfg.SPLIT_TEST_DAYS)

    valid_idx = test.sample(frac=.5, random_state=cfg.RANDOM_STATE).index

    valid = test.loc[valid_idx].copy()
    test = test.drop(valid_idx)

    # aggregate flights 
    print('...Grouping flight data...')
    
    grouped = agg_flights(train)

    # estimated prices dataframe
    create_price_bins(train)

    # Store data in processed folder
    os.chdir(sys.path[0])

    grouped.to_csv(cfg.TRAIN_PATH, index=False)
    print('\nTraining data stored successfully!:', cfg.TRAIN_PATH)

    valid.to_csv(cfg.VALID_PATH, index=False)
    print('Validation data stored successfully!:', cfg.VALID_PATH)
    
    test.to_csv(cfg.TEST_PATH, index=False)
    print('Testing data stored successfully!:', cfg.TEST_PATH)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_drop', type=float, default=0.05)
    args = parser.parse_args()

    preprocessing(cfg.INTERIM_DATA_PATH, args.min_drop)
    
    
