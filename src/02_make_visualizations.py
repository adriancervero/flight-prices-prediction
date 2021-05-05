#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Make some plots and store them in figures folder
    @author: Adrián Cerveró
"""
#-------------------------------------------------------------------
# Imports 
# ------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import os, sys
import matplotlib.ticker as ticker

import config as cfg

sns.set_palette('deep')
import warnings
warnings.filterwarnings('ignore')

#-------------------------------------------------------------------
# Variables 
# ------------------------------------------------------------------
INTERIM_DATA_PATH = cfg.INTERIM_DATA_PATH  # output
FIGURES_PATH = cfg.FIGURES_PATH
# ------------------------------------------------------------------
def load_data(path):
    """ Load data and return a Pandas dataframe. """
    print('...loading data from .csv...')
    
    os.chdir(sys.path[0]) # relative path to the .py file
    df = pd.read_csv(path)
    return df

def build_features(flights):
    """ Build featured needed for some plots """
    flights['orig-dest'] = flights['flyFrom'] +'-'+ flights['flyTo']
    flights['airline'] = flights['airlines'].str.split(',')[0][0]
    flights['flight_no'] = flights['airline']+''+flights['flight_no'].astype(str)
    days_of_week = {5:'Monday', 6:'Tuesday', 0:'Wednesday', 1:'Thursday', 2:'Friday', 3:'Saturday', 4:'Sunday'}
    flights['day_of_week'] = pd.to_datetime(flights['dDate']).dt.weekday.map(days_of_week)
    flights['session'] = pd.cut(pd.to_datetime(flights['dTime']), bins=3, labels=['night', 'morning', 'evening'])
    flights = add_days_until_dep_feature(flights)
    return flights

def add_days_until_dep_feature(df):
    """ Remaining days until flight departure """
    collected = pd.to_datetime(df['collectionDate'])
    departure =  pd.to_datetime(df['dDate'])
    daysUntilDep = departure - collected
    df['days_until_dep'] = daysUntilDep.dt.days
    return df

def plot_price_hist(flights):
    """ Plot price distributions """
    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    # plot price hist
    sns.histplot(x='price', data=flights, kde=False, bins=50, ax=axes[0]);
    axes[0].set_title('Price Histogram')

    # plot log transform price hist
    flights['log_price'] = np.log(flights['price'])
    sns.histplot(x='log_price', data=flights, kde=False, bins=50, ax=axes[1]);
    axes[1].set_title('Log-Price Histogram')

    sns.despine(offset=5)

    figure_path = FIGURES_PATH+'price_hist.png'
    fig.savefig(figure_path, pad_inches=0.5, bbox_inches='tight')
    print(figure_path)

def plot_price_over_time(flights):
    """ Plot mean price over three months of data collected """
    grouped = flights.groupby('collectionDate')['price'].mean().reset_index()

    
    fig, axes = plt.subplots(1,1,figsize=(20,5))
    ax = sns.lineplot(x='collectionDate', y='price', data=grouped);
    ax.xaxis.set_major_locator(ticker.LinearLocator(10))

    plt.xticks(rotation=0);
    sns.despine(offset=0)
    plt.xlabel('Day of Data Collection');
    plt.ylabel('Price')
    plt.title('Price Over Time', fontsize=16);

    figure_path = FIGURES_PATH+'price_over_time.png'
    fig.savefig(figure_path, pad_inches=0.5, bbox_inches='tight')
    print(figure_path)

def plot_day_of_week_mad_ny(flights):
    """ Plot weekday impact on Madrid-New York route """
    mad_ny = flights[flights['orig-dest'] == 'MAD-JFK']
    grouped = mad_ny.groupby(['orig-dest', 'day_of_week', 'days_until_dep'])['price'].quantile(.25).reset_index()

    plt.figure(figsize=(14,8));
    g = sns.FacetGrid(grouped, col="day_of_week", col_wrap=4);
    g.map(sns.lineplot, "days_until_dep", "price");
    g.fig.suptitle('Price trends by weekday on MAD-JFK', fontsize=18);
    plt.tight_layout()

    figure_path = FIGURES_PATH+'day_of_week_mad-jfk.png'
    g.savefig(figure_path, pad_inches=0.5, bbox_inches='tight')
    print(figure_path)

def plot_day_of_week_all(flights):
    
    fig, axes = plt.subplots(1, 1, figsize=(20,5))
    grouped = flights.groupby(['orig-dest', 'day_of_week', 'days_until_dep'])['log_price'].quantile(.25).reset_index()
    sns.boxplot(x='orig-dest', y='log_price', data=grouped, showfliers=False, hue='day_of_week',
                hue_order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']);
    sns.despine(offset=10);
    plt.ylabel('Price')
    plt.xlabel('Route');
    plt.title('Prices by Session on each route', fontsize=16);
    [plt.axvline(x, color = 'grey', linestyle='--') for x in [0.5,1.5,2.5,3.5, 4.5,5.5, 6.5, 7.5,8.5]];
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    figure_path = FIGURES_PATH+'day_of_week_all.png'
    fig.savefig(figure_path, pad_inches=0.5, bbox_inches='tight')
    print(figure_path)

def plot_session(flights):
    grouped = flights.groupby(['orig-dest', 'session', 'days_until_dep'])['log_price'].quantile(.25).reset_index()

    
    g = sns.FacetGrid(grouped, col="orig-dest", col_wrap=5, hue='session');
    g.map(sns.lineplot, "days_until_dep", "log_price");
    g.fig.suptitle('Price trends by Session', fontsize=18);
    plt.tight_layout()
    plt.legend();

    figure_path = FIGURES_PATH+'session.png'
    g.savefig(figure_path, pad_inches=0.5, bbox_inches='tight')
    print(figure_path)

def plot_days_until_dep(flights):
    
    fig, axes = plt.subplots(1, 1, figsize=(12,5))
    sns.lineplot(x='days_until_dep', y='log_price', data=flights);
    sns.despine(offset=0);
    plt.ylabel('Price')
    plt.xlabel('Days Until Departure');

    figure_path = FIGURES_PATH+'days_until_dep_vs_price.png'
    fig.savefig(figure_path, pad_inches=0.5, bbox_inches='tight')
    print(figure_path)

def make_visualizations():
    flights = load_data(INTERIM_DATA_PATH)
    print("...making plots...")
    flights = build_features(flights)
    plot_price_hist(flights)
    plot_price_over_time(flights)
    plot_day_of_week_mad_ny(flights)
    plot_day_of_week_all(flights)
    plot_session(flights)
    plot_days_until_dep(flights)
    
    print('\nDone!')
if __name__ == '__main__':
    make_visualizations()