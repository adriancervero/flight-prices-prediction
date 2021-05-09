#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    This module contains the class PriceEstimator
    used by the front-end application.

"""

import pandas as pd
import numpy as np

import config as cfg
from src.PriceEstimatorTrainer import get_estimated_price, get_wait_prices

class PriceEstimator:
    """ 
        PriceEstimator for the front-end application. Output function
        returns predictions and days to wait for a price drop in a 
        pandas dataframe.

        Attributes:
            - model: trained model by PriceEstimatorTrainer
            - flights: Dataframe with flight data for searches 
                    and predictions.
    """
    def __init__(self, model, flights):
        self.model = model
        self.flights = flights
        self.train = pd.read_csv('data/processed/train.csv')
        self.price_bins = pd.read_csv('data/processed/price_bins.csv')
        self.bins_days = pd.read_csv('data/processed/bins_days.csv')
        
    def _requested_flights(self, orig, dest, collectionDate, dDate, limit=5):
        """ Return cheapest flights with requested itinerary """
        orig_dest = f'{orig}-{dest}'
        flights = self.flights
        
        req = flights[(flights['orig-dest'] == orig_dest) & \
                      (flights['dDate'] == dDate) & \
                     (flights['collectionDate'] == collectionDate)]
        
        return req.sort_values(by='price')[:limit]
    
    def _prepare_data(self, df):
        """ Prepare data for prediction """
        CATEGORICAL = cfg.CATEGORICAL
        NUMERICAL = cfg.NUMERICAL

        agg_cols = cfg.AGG_COLS
        df = pd.merge(df, self.train[['competition', 'prob']+agg_cols], on=agg_cols, how='left') 
        X = df[CATEGORICAL+NUMERICAL]
        return X

    
    def _predict(self, X):
        """ Use the trained model for make predictions """
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]
        return preds, probs
    
    def _get_wait_estimate(self, df):
        """ Returns days to wait for price drop """ 

        price_bins = self.price_bins
        bins_days = self.bins_days
        
        wait = df
        
        list_prices = price_bins.groupby(['orig-dest','airline'])['price_est'].agg(list).rename('list_price_est').reset_index()
        wait = pd.merge(wait, list_prices, on=['orig-dest','airline'], how='left')
        wait = pd.merge(wait, bins_days, on='days_until_dep', how='left')
        
        wait['price_min_bin'] = wait.apply(get_estimated_price, axis=1)
        wait['min_bin'] = wait['price_min_bin'].apply(lambda x: x[1])
        wait['price_est'] = wait['price_min_bin'].apply(lambda x: x[0])
        wait['current_price_est'] = wait['price_min_bin'].apply(lambda x: x[2])
        wait.drop('price_min_bin', axis=1, inplace=True)

        # compute drop percentage
        drop_per = ((1-wait['price_est']/wait['current_price_est'])*100).round().abs().astype(int)

        # days to wait
        wait['min_wait'] = wait['days_until_dep'] - (wait['min_bin']*5+6)
        wait['max_wait'] = wait['days_until_dep'] - (wait['min_bin']*5+2)

        # wait_price
        wait = pd.merge(wait, self.min_prices[['orig-dest', 'dDate', 'list_prices', 'days_until_dep']], on=['orig-dest', 'dDate', 'days_until_dep'])
        wait_price = wait.apply(get_wait_prices, axis=1)
        
        return drop_per, wait['min_wait'], wait['max_wait'], wait_price

    def _get_min_prices(self, flights):
        """ 
        Return dataframe with min prices for each pair departure date/days until 
        departure and labeled according to the number of days remaining 
        until flight departure.
        """
        # getting minimum prices
        min_prices = flights.groupby(['orig-dest','dDate','days_until_dep'])['price'].min().reset_index()

        # list daily prices for each itinerary 
        list_prices = min_prices.groupby(['orig-dest', 'dDate'])['price'].agg(list).reset_index()
        list_prices.rename(columns={"price":"list_prices"}, inplace=True)
        min_prices['list_prices'] = pd.merge(min_prices, list_prices, on=['orig-dest','dDate']).reset_index()['list_prices']

        
        
        self.min_prices = min_prices
    
    def output(self, orig, dest, collectionDate, dDate):
        """ 
            Function for front-end app. Given a flight predict wait or buy 
            and give an estimate of the days to wait, estimated price drop 
            percentage and probability.
        """
        req = self._requested_flights(orig, dest, collectionDate, dDate)
        
        X = self._prepare_data(req)
        self._get_min_prices(self.flights)

        preds, probs = self._predict(X)
        drop_per, min_wait, max_wait, wait_price = self._get_wait_estimate(req)
        

        output = pd.DataFrame({
            'wait':preds,
            'probs':probs,
            'drop_per':drop_per,
            'min_wait': min_wait,
            'max_wait': max_wait,
            'wait_price': wait_price,
        })
        
        return req, output