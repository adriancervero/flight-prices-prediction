import streamlit as st
import streamlit.components.v1 as components
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

with header:
    st.title("Welcome!")
    st.text("In this project I look into the transactions of taxis in NYC")

with dataset:
    st.header('Flights Dataset')
    data = pd.read_csv('data/processed/test.csv')
    st.write(data.head())
    

with features:
    a = st.header('The features I created')
    if st.button('Click'):
        a.header('re')
    st.date_input('Departure Date')
with model_training:
    st.header('Time to train the model!')
    st.markdown('---')
    col1, col2, col3 = st.beta_columns(3)
    #col1.header("col1")
    #col2.header("col2")
    
    col1.text('MAD-JFK')
    col1.text('Iberia')
    col2.text('200€')
    col3.bar_chart()
    #components.html("""<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """)
    st.markdown('---')
    col1, col2, col3 = st.beta_columns(3)
    #col1.header("col1")
    #col2.header("col2")
    
    col1.text('MAD-JFK')
    col1.text('Iberia')
    col2.text('200€')
    col3.button('Select')
    #components.html("""<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """)
    st.markdown('---')

