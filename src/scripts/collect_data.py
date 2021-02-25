#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is a script for extract flights data from Kiwi API (https://docs.kiwi.com)
and store it in a PostgreSQL

@author: Adrián Cerveró Orero
@date: 17/01/2021

"""
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine



DAYS_RANGE = 90
ROUTES = [
    # Regional
    ('MAD', 'BCN'), # Madrid-Barcelona
    ('MAD', 'TFN'), # Madrid-Tenerife
    ('BCN', 'PMI'), # Barcelona-Palma de Mallorca
    # International
    ('MAD', 'LHR'), # Madrid-London
    ('MAD', 'JFK'), # Madrid-New York
    ('MAD', 'EZE'), # Madrid-Buenos Aires
    ('MAD', 'MEX'), # Madrid-Mexico City
    ('BCN', 'LGW'), # Barcelona-London
    ('BCN', 'AMS'), # Barcelona-Amsterdam
    ('BCN', 'FCO'), # Barcelona-Roma
]

HOST = '********************'
PORT = '****'
DB  = '******'
USER = '******''
PASSW = '********************'

def send_request(origin, dest, date_from, date_to, limit=1000):
    """ 
    Connect to the API and request flight data 
    
        Parameters: 
            origin (str): IATA code of the origin airport
            dest (str): IATA code of the destination
            date_from (str): search flights from this date (dd/mm/YYYY)
            date_to (str): search flights upto this date (dd/mm/YYYY)
            limit (int): limit number of results; default value is 1000
        
        Returns:
            response_json (dict): dict with flight data response
    """
    URL = "https://api.skypicker.com/flights?partner=picky&v=3"
    PARAMS = {
        'flyFrom':origin,
        'to':dest,
        'date_from': date_from,
        'date_to': date_to,
        'limit':limit
    }
    
    attemps = 0
    while attemps < 5:
        try:
            response = requests.get(url=URL, params=PARAMS)
            if 'message' in list(response.json().keys()):
                print(response.text)
                return
            break
        except Exception as e:
            if attemps > 5:
                print('Tries number wasted')
                return -1
            else:
                print('ERROR:', e)
                attemps += 1
                time.sleep(3)    

    
    return response.json()

def routes_to_string(routes):
    """ Converts route list into string"""
    steps = [route[0] for route in routes] + [routes[-1][1]]
    return ' -> '.join(steps)

def process_response(data):
    """ 
    Process the JSON object; selecting relevant columns, make some preprocessing 
    and transform into a DataFrame.
    
        Parameters: 
                data (dict): dict with flight data from 'send_request' method
        Returns:
                final_df (Pandas DataFrame) : DataFrame with relevant data 
                extracted from JSON response.
    """
    columns = ['dTime', 'dTimeUTC', 'aTime', 'aTimeUTC', 'airlines',
               'fly_duration', 'flyFrom', 'cityFrom', 'cityCodeFrom','flyTo',
               'cityTo','cityCodeTo', 'distance', 'price']

    df = pd.DataFrame(data)

    try:
        final_df = df[columns].copy()
    except Exception as e:
        print('ERROR:', e)
        for col in columns:
            if col in df.columns:
                print(col, 'OK')
            else:
                print(col, 'column not found')
    # bit of preprocessing before store it in database
    final_df['route'] = df['routes'].apply(routes_to_string)
    final_df['countryFrom'] = df['countryFrom'].apply(lambda x: x['name'])
    final_df['countryTo'] = df['countryTo'].apply(lambda x: x['name'])
    final_df['flight_no'] = df['route'].apply(lambda x: x[0]['flight_no'])
    final_df['seats'] = df['availability'].apply(lambda x: x['seats'])
    final_df['airlines'] = final_df['airlines'].apply(lambda x: ', '.join(x))
    final_df['dTime'] = final_df['dTime'].apply(lambda x: datetime.fromtimestamp(x))
    final_df['aTime'] = final_df['aTime'].apply(lambda x: datetime.fromtimestamp(x))
    final_df['collectionDate'] = pd.Series([datetime.now()]*final_df.shape[0])

    return final_df

def request_data(days_range, routes, verbose=False):
    """ Function that manage all the requests to the API in order to gather 
        all needed data.
        
            Parameters:
                days_range (int) : Range of days in which to check flights prices.
                verbose (bool) : Print data requested in real time for debugging.
            Returns:
                fligths (Pandas DataFrame) : DataFrame with all the data collected.
    """
    print('Requesting data...')
    df = pd.DataFrame([])
    for route in routes:
        # Initialize date to tomorrow
        i_date = datetime.now().date() + timedelta(days=1)
        for i in range(days_range):

            i_date_str = i_date.strftime('%d/%m/%Y')
            orig = route[0]
            dest = route[1]

            response = send_request(orig, dest, i_date_str, i_date_str)
            if response == -1:
                continue
            data = process_response(response['data'])
            
            if verbose:
                print(i_date_str,'from:', orig, 'to:',dest, 'flighs:',data.shape[0])
            df = df.append(data)

            # update date to the next day
            i_date += timedelta(days=1)
        
    with open('flights.log', 'a') as f:
        now = datetime.now()
        now_str = now.strftime("%d-%b-%Y (%H:%M:%S)")
        
        f.write('%s - Flights added: %d \n' % (now_str, df.shape[0]))
    
    df.to_csv(now.strftime("data/flights_%d-%m.csv"), index=False)
    print('Data requesting done!')
    return df

def store_db(df):
    """ Connects to SQL database and insert fligths collected """
    
    url = 'postgres+psycopg2://%s:%s@%s:%s/%s' % (USER, PASSW, HOST, PORT, DB)
    print('Connecting to... %s:%s/%s' % (HOST, PORT, DB))
    
    engine = create_engine(url)
    
    df.to_sql('flights', engine, if_exists='append', index=False)
    print('Data stored sucessfully!')

def main():
    data = request_data(DAYS_RANGE, ROUTES, verbose=True)
    store_db(data)

if __name__ == "__main__":
    main()
