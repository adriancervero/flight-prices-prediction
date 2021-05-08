#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Front-end App built with Streamlit """

import sys, os
import streamlit as st
import streamlit.components.v1 as components
import numpy as np 
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from src.PriceEstimator import PriceEstimator
from datetime import timedelta



def load_data():
    valid = pd.read_csv('data/processed/valid.csv')
    test = pd.read_csv('data/processed/test.csv')
    return valid.append(test)

def load_estimator(model_name, flights_data):
    """ Load estimator object with trained model """
    model = pickle.load(open('models/'+model_name, 'rb'))
    estimator = PriceEstimator(model, flights_data)
    return estimator

def build_pieChart(prob):
    """ Plot a pie chart with price drop probability """
    fig, ax = plt.subplots()

    # color
    if prob >= 0.6:
        main = 'seagreen'
    elif prob >= 0.25:
        main = 'goldenrod'
    else:
        main = 'darkred'
    pie_colors = [main, 'grey']

    plt.pie([prob, 1-prob], colors=pie_colors, startangle=90)

    # text prob
    font_dict ={
        'fontsize':40,
        'fontname':'Roboto'
    }
    prob_str = str(round(prob*100))+'%'
    plt.text(x=-0.35, y=-0.10, s=prob_str, fontdict=font_dict)


    # white circle for donut shape
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle)

    return fig

if __name__ == '__main__':
    header = st.beta_container()
    input_container = st.beta_container()
    results = st.beta_container()

    # load estimator
    model_name = 'rf_pre_69%_sav_11%.pkl'
    flights_data = load_data()
    estimator = load_estimator(model_name, flights_data)

    # HEADER ##

    with header:
        st.title("Flight Tickets: Buy or Wait")
        st.write("This predictor recommends buying a airline \
         ticket now or waiting for a price drop. The model is a classifier that has \
         been trained on past flights that travel 10 routes over 3 months.")
        st.write("The model \
         returns two values: the probability that the price will decrease from the \
         current day until the departure of the flight, and a text response estimating\
         how many days to wait for the price decrease.")

        st.write('Github: https://github.com/adriancervero/flight-prices-prediction')


    # SEARCH FLIGHT INPUT
    with input_container:
        st.subheader('Search flight tickets:')

        col1, col2, col3 = st.beta_columns(3)
        orig = col1.selectbox('From',options=['Madrid-MAD', 'Barcelona-BCN'])
        if orig == 'Madrid-MAD':
            dest_options = ['New York-JFK', 'Buenos Aires-EZE', 'Mexico City-MEX', 'Barcelona-BCN', 'London-LHR', 'Tenerife-TFN']
        else:
            dest_options = ['Amsterdam-AMS', 'Rome-FCO', 'London-LGW', 'Palma de Mallorca-PMI']
        dest = col2.selectbox('To', options=dest_options)
        max_date = pd.to_datetime(flights_data['dDate']).max()
        min_date = pd.to_datetime(flights_data['dDate']).min()
        dDate_input = col3.date_input('Departure Date',value=min_date+timedelta(days=7), max_value=max_date, min_value=min_date)
        search_button = st.button('Search')
        

        # After search button clicked
        if search_button:
            today = str(min_date).split()[0]
            dDate = str(dDate_input).split()[0]
            
            orig = orig.split('-')[1]
            dest = dest.split('-')[1]
            req, out = estimator.output(orig, dest, today, dDate)


            for ind, tup in enumerate(req.iterrows()):
                flight = tup[1]
                output = out.loc[ind]

                wait = bool(output['wait'])
                if wait == False:
                    res_str = 'Price drop is unlikely. Buy now!'
                else:
                    min_wait = int(output['min_wait'])
                    max_wait = int(output['max_wait'])
                    res_str = f'Price drop is likely. Wait {min_wait} to {max_wait} days'

                h1, h2 = st.beta_columns(2)

                price = flight['price']
                if ind == 0: # First flight
                    # Price
                    col1, col2 = st.beta_columns(2)
                    price_str = f'<p style="color:black; font-size:30px;">{price}€</p>'
                    col1.markdown(price_str, unsafe_allow_html=True)
                    
                    
                    if wait: # predicts wait
                        res_str_style = f'<p style="background-color:cornflowerblue; border-radius: 6px; color:white; display:inline; font-size:20px;">{res_str}</p>'
                        col2.text('Wait price:')
                        col2.write(output['wait_price'])

                    else: # predicts buy
                        res_str_style = f'<p style="background-color:seagreen; border-radius: 6px;color:white; display:inline; font-size:16px;">{res_str}</p>'

                    # text response
                    col1.markdown(res_str_style, unsafe_allow_html=True)
                    

                else: # alternatives flights
                    price_str = f'<p style="color:black; font-size:18px;">{price}€</p>'
                    st.markdown(price_str, unsafe_allow_html=True)

                # Flight info layout
                col1, col2, col3 = st.beta_columns(3)

                # COL 1
                col1.text('')
                col1.text('')
                col1.text(flight['orig-dest'])
                col1.text('Airline: '+ flight['airline'])

                # COL 2
                duration = round(flight['fly_duration'],1)
                col2.text('')
                col2.text('')
                col2.text(flight['dTime'] + ' ------> ' + flight['aTime'])
                col2.text('Route: '+flight['route'])

                # COL 3
                if ind == 0:
                    col3.subheader('Price drop prob.')
                    chart = build_pieChart(output['probs'])
                    col3.write(chart)
                    #col1.button('Book flight')
                    input_container.subheader('Alternative flights:')
                    input_container.markdown("""<div style="height:3px;border:none;color:#333;background-color:#333;" </div> """,unsafe_allow_html=True)
                else:
                    col3.button('Select', key=ind)
                    st.markdown('---')

