#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Prepare raw data for the exploratory analysis
    @author: Adrián Cerveró
"""

import pandas as pd
import numpy as np
from datetime import datetime

RAW_DATA_PATH = '../../data/raw/flights_raw.csv'
INTERIM_DATA_PATH = '../../data/interim/flights_interim.csv'

def load_data(path):
    """
    Load the raw data and return a Pandas dataframe.

    Args:
        path (str): raw data filename path
    Returns:
        df (DataFrame)
    """
    # raw data has no column name so we need to provide it to the dataframe
    columns = ['dTime', 'dTimeUTC', 'aTime', 'aTimeUTC', 'airlines',
               'fly_duration', 'flyFrom', 'cityFrom', 'cityCodeFrom','flyTo',
               'cityTo','cityCodeTo', 'distance', 'price', 'route', 'countryFrom',
          'countryTo', 'flight_no', 'seats', 'collectionDate']

    df = pd.read_csv(path, names=columns)
    return df

def process_dates_cols(df):
    """
    - Split dates in date and time columns. 
    - Convert from UTC Timestapms into datestimes.
    - Remove time from Collection Date columns.

    Args:
        df (DataFrame): Flights dataframe.

    Returns:
        df (DataFrame): The same dataframe with date columns 
                        preprocessed.
    """
    # local dates
    df['dDate'] = df['dTime'].apply(lambda x: x.split(' ')[0])
    df['dTime'] = df['dTime'].apply(lambda x: x.split(' ')[1][:5])
    df['aDate'] = df['aTime'].apply(lambda x: x.split(' ')[0])
    df['aTime'] = df['aTime'].apply(lambda x: x.split(' ')[1][:5])

    # utc dates
    df['dTimeUTC'] = df['dTimeUTC'].apply(lambda x: datetime.utcfromtimestamp(x))
    df['aTimeUTC'] = df['aTimeUTC'].apply(lambda x: datetime.utcfromtimestamp(x))
    
    # collection date
    df['collectionDate'] = df['collectionDate'].apply(lambda x: x.split(' ')[0])
    return df

def prepare_data(filename):
    """
    Load dataset, preprocess some columns and reordering them.
    Store the result in data/interim for the next step in the workflow.

    Args:
        filename (str): raw data filename path
    """
    flights = load_data(filename)
    flights = process_dates_cols(flights)

    columns = ['collectionDate','dDate', 'dTime', 'aDate', 'aTime', 'dTimeUTC', 'aTimeUTC',
           'flyFrom', 'flyTo', 'airlines', 'flight_no', 'fly_duration', 'distance', 'route',
           'price','seats', 'cityFrom', 'cityCodeFrom', 'cityTo', 'cityCodeTo', 'countryFrom', 
           'countryTo']

    flights = flights[columns]
    flights.to_csv(INTERIM_DATA_PATH, index=False)
    return '01 data prepared!: ' + INTERIM_DATA_PATH

if __name__ == '__main__':
    filename = RAW_DATA_PATH
    print(prepare_data(filename))
    

    