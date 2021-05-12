#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
    This module contains the class PriceEstimatorTrainer
    used by the training module.

"""

#-------------------------------------------------------------------
# Imports 
# ------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg
import sys, os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score

#-------------------------------------------------------------------
# Aux. Functions 
# ------------------------------------------------------------------

def create_pipeline(model, scaler=StandardScaler(), encoder=OneHotEncoder(handle_unknown='ignore')):
    """ 
        Return a preprocessing pipeline that scale numerical variables
        and encode categorical ones .        
    """
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler),     
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', encoder),
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, cfg.NUMERICAL),
        ('cat', cat_pipeline, cfg.CATEGORICAL),
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    return pipeline

def get_actual_labels(row):
    """ Assign wait or buy label using actual data """
    current_d = row['days_until_dep']
    current_price = row['price']
    list_prices = np.array(row['list_prices'])
    next_days = list_prices[:current_d-1]
    if len(next_days) == 0:
        return 0
    else:
        min_price = np.min(next_days)

        if min_price < current_price and 1-(min_price/current_price) > cfg.MIN_DROP_PER_TEST:
            return 1
        else:
            return 0

def get_estimated_price(row):
    """ 
        Compute for each flight estimated price and days to wait
        until the price will be minimum in next days until departure.

        Returns:
            - min_price: minimum price that the flight will reach
            - min_bin: days bin belonging to that price
            - current_price: estimate price for current day
                (this will be used for compute drop percentage)
    """
    list_prices = row['list_price_est']
    current_bin = row['days_bins']
    current_price = list_prices[current_bin]
    next_prices = list_prices[:current_bin]
    if next_prices == []:
        print(row['id'])
    min_bin = np.argmin(next_prices)
    min_price = next_prices[min_bin]
    return min_price, min_bin, current_price

def get_wait_prices(row):
    """ Returns price after estimated waiting days.
        This will be used for compute money saved. """
    list_prices = row['list_prices']
    current_day = row['days_until_dep']
    min_wait = row['min_wait']
    max_wait = row['max_wait']
    i = current_day-max_wait
    j = current_day-min_wait
    wait_prices = list_prices[i:j]
    if wait_prices == []:
        return np.nan
    return np.min(wait_prices)

#-------------------------------------------------------------------
# Class definition
# ------------------------------------------------------------------

class PriceEstimatorTrainer:
    """
        PriceEstimatorTrainer trains a model that predicts price drops
        for flights. Evaluates on simulated passengers and compute 
        the savings using the trained model.

        Attributes:
            model (str): model to use
            pred_threshold (float): threshold for predictions probabilities
            n (int): number of passengers to simulates when evaluating

            data atributes (train, valid, test, price_bins, bins_days):
                all the data needed for the process get loaded during
                class initialization.

    """
    def __init__(self, model, pred_threshold=0.5, n=10000):
        self.model = create_pipeline(model)
        self.train = pd.read_csv(cfg.TRAIN_PATH)
        self.valid = pd.read_csv(cfg.VALID_PATH)
        self.test = pd.read_csv(cfg.TEST_PATH)
        self.price_bins = pd.read_csv(cfg.PRICE_BINS_PATH)
        self.bins_days = pd.read_csv(cfg.BINS_DAYS_PATH)
        self.n = n
        self.pred_threshold = pred_threshold
        
    def _prepare_train(self, df):
        """ Select features and split target """
        X = df[cfg.CATEGORICAL+cfg.NUMERICAL].copy()
        y = df[cfg.TARGET].values
        
        return X, y
    
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

        # label wait or buy according next days on list prices 
        min_prices['wait'] = min_prices.apply(get_actual_labels, axis=1)
        
        self.min_prices = min_prices
        
    def _generate_flights(self, flights):
        """ 
        Create dataframe with test flights for perform predictions.
        Steps:
            1. Generate n travelers
            2. Assign to each passenger the cheapest flight available
            3. Add needed features for model using training flights groups
            4. Add label from min_prices dataframe (target for our predictions)
        """
        agg_cols = cfg.AGG_COLS
        # generate travelers: (id, collectionDate, orig-dest, dDate)
        travelers = flights[flights['days_until_dep']>=7][['collectionDate']+['orig-dest', 'dDate']].sample(self.n, random_state=cfg.RANDOM_STATE)
        travelers['id'] = np.arange(self.n)

        # assinging cheapest flights to travelers
        travelers_flights = pd.merge(travelers, flights, on=['collectionDate']+['orig-dest', 'dDate'])
        cheapest_flights = travelers_flights.groupby('id')['price'].idxmin()
        test_flights = travelers_flights.iloc[cheapest_flights]

        # adding 'competition' and 'prob' columns from train
        test_flights = pd.merge(test_flights, self.train[['competition', 'prob']+agg_cols], on=agg_cols, how='left')

        # adding 'wait' col (actual labels) from min_prices df
        test_flights = pd.merge(test_flights, self.min_prices[['orig-dest', 'dDate','days_until_dep', 'wait']], on=['orig-dest', 'dDate','days_until_dep'], how='left')
        return test_flights

    def _make_predictions(self, flights):
        """ Use the trained model for make predictions and 
            compute probabilities. """
        test_flights = flights
        X_valid = test_flights[cfg.CATEGORICAL + cfg.NUMERICAL]
        y_valid = test_flights[cfg.TARGET].values

        #test_flights['predicted'] = self.model.predict(X_valid)
        probs = self.model.predict_proba(X_valid)[:, 1]
        predicted = (probs >= self.pred_threshold).astype(int)

        #predicted = np.random.choice(np.array([0,1]), len(predicted))


        test_flights['predicted'] = predicted
        test_flights['wait_prob'] = probs
        self.test_flights = test_flights
        
    def _compute_savings(self):
        """ Given wait predictions estimate days to wait and compute savings comparing prices. """
        
        test_flights = self.test_flights
        price_bins = self.price_bins
        bins_days = self.bins_days
        
        # get just with predicted wait rows
        wait = test_flights[(test_flights['predicted'] == 1) & (test_flights['days_until_dep'] >= 7)].copy()

        # adding estimated prices
        list_prices = price_bins.groupby(['orig-dest','airline'])['price_est'].agg(list).rename('list_price_est').reset_index()
        wait = pd.merge(wait, list_prices, on=['orig-dest','airline'], how='left')
        wait = pd.merge(wait, bins_days, on='days_until_dep', how='left')

        wait['price_min_bin'] = wait.apply(get_estimated_price, axis=1)
        wait['min_bin'] = wait['price_min_bin'].apply(lambda x: x[1])
        wait['price_est'] = wait['price_min_bin'].apply(lambda x: x[0])
        wait['current_price_est'] = wait['price_min_bin'].apply(lambda x: x[2])
        wait.drop('price_min_bin', axis=1, inplace=True)

        # compute drop percentage
        wait['drop_per'] = ((1-wait['price_est']/wait['current_price_est'])*100).round()

        # days to wait
        wait['min_wait'] = wait['days_until_dep'] - (wait['min_bin']*5+6)
        wait['max_wait'] = wait['days_until_dep'] - (wait['min_bin']*5+2)

        # adding wait price
        wait = pd.merge(wait, self.min_prices[['orig-dest', 'dDate', 'list_prices', 'days_until_dep']], on=['orig-dest', 'dDate', 'days_until_dep'])
        wait['wait_price'] = wait.apply(get_wait_prices, axis=1)

        # savings
        wait['savings'] = wait['price'] - wait['wait_price']

        # merge with full dataframe
        self.final_df = pd.merge(test_flights, wait, how='left')

        # if less than a week for departure we will predict 'buy'
        self.final_df.loc[self.final_df['days_until_dep']<7, 'predicted'] = 0
        self.final_df.loc[self.final_df['days_until_dep']<7, 'wait'] = 0
        
    def _report_results(self):
        """ Print results by all routes and by individual routes """
        
        df = self.final_df
        
        # TOTAL RESULTS
        savings = df[df['savings'] > 0]['savings'].sum()
        losses = df[df['savings'] < 0]['savings'].sum()
        accuracy = accuracy_score(df['wait'], df['predicted'])*100
        f1 = f1_score(df['wait'],  df['predicted'])*100
        precision = precision_score(df['wait'], df['predicted'])*100
        recall = recall_score(df['wait'], df['predicted'])*100


        df['saving_per'] = df['savings']/df['price']
        saving_per = (df.groupby('orig-dest')['saving_per'].mean()*100).round(2).astype('str') + '%'
        res_total = pd.DataFrame({
            'Model': self.model['model'].__class__.__name__,
            'Savings (k€)': round(savings/1000, 1),
            'Losses (k€)': round(losses/1000, 1),
            'Mean (%)': round(df['saving_per'].mean(), 2)*100,
            'Accuracy': str(round(accuracy, 2)) + '%',
            'f1-score': str(round(f1, 2)) + '%',
            'Precision': str(round(precision, 2)) + '%',
            'Recall': str(round(recall, 2)) + '%',
        }, index=['0'])

        acc = round(accuracy_score(df['wait'], df['predicted']), 2)*100
        savs = round(df['saving_per'].mean(), 2)*100

        print(res_total.to_string(index=False)+'\n\n')

        # RESULTS BY ROUTE
        savings = (df[df['savings'] > 0].groupby('orig-dest')[['savings']].sum()/1000).round(1)
        losses = (df[df['savings'] <= 0].groupby('orig-dest')[['savings']].sum()/1000).round(1)
        wait_per = (df.groupby('orig-dest')['predicted'].mean()*100).round(2).astype('str') + '%'
        mean_savings = df.groupby('orig-dest')['savings'].mean()
        df['correct'] = (df['predicted'] == df['wait']) & (df['wait']==1)


        res_byroute = pd.DataFrame({
            'Savings (k€)': savings['savings'],
            'Losses (k€)': losses['savings'],
            'Mean Savings (€)': round(mean_savings, 2),
            'Savings Percentage': saving_per,
            'Wait predicted': wait_per,
        }, index=savings.index)


        print(res_byroute.to_string())
        print(f'\nThreshold: {self.pred_threshold}')
        print("Confusion Matrix:")
        print(confusion_matrix(df['wait'], df['predicted']))
        return precision, savs
    
    def _plot_results(self):
        """ Plot money saved by simulated passengers by route """
        df = self.final_df
        savings = df[df['savings'] > 0].groupby('orig-dest')[['savings']].sum().astype(int).reset_index()
        losses = df[df['savings'] < 0].groupby('orig-dest')[['savings']].sum().astype(int).reset_index()
        fig, axs = plt.subplots(figsize=(8,6))
        g = sns.barplot(x='savings', y='orig-dest', data=savings, color='seagreen', );
        g2 = sns.barplot(x='savings', y='orig-dest', data=losses, color='darksalmon');
        #g.spines['left'].set_position(('axes', 0))

        # values text
        for index, row in savings.iterrows():
            g.text(row['savings']+600, index+0.10, str(row['savings'])+' €' , color='darkgreen', ha="center")
        for index, row in losses.iterrows():
            g2.text(row['savings']-600, index+0.10, str(row['savings'])+' €' , color='darkred', ha="center")

        sns.despine(left=False, bottom=True)
        # remove x axis
        plt.yticks(fontsize=14)
        plt.xticks([])
        plt.xlabel(None)
        plt.ylabel(None)
        lim = savings['savings'].max()
        plt.xlim(-lim,lim)

       
        fig.savefig('../figures/savings_by_route.png', pad_inches=0.2, bbox_inches='tight')
        
    def fit(self):
        """ Fit model with training data """
        X_train, y_train = self._prepare_train(self.train)
        self.model.fit(X_train, y_train)
    
    def evaluate(self, on='valid', plot=False):
        """ 
            Use test/validation data for evaluate the model printing the savings
            of the simulated passengers
        """
        if on == 'valid':
            test = self.valid
        elif on == 'test':
            test = self.test
            
        self._get_min_prices(test)
        test_flights = self._generate_flights(test)
        self._make_predictions(test_flights)
        self._compute_savings()
        precision, mean_savings = self._report_results()
        if plot:
            self._plot_results()
        return precision, mean_savings

