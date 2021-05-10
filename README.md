# Flight Prices Prediction to Determine the Right Time to Buy Airline Tickets

![Alt Text](demo.gif)

## Objective
The price of a flight varies greatly over the months leading up to the departure day. These fluctuations are due to many factors; supply/demand, 
airline offers, remaining days before the flight departs, etc... 
\
\
In this project the objective is to try to model these fluctuations in order to create a predictor that is able to indicate if the flight price 
will go down in the future and how many days the traveler should wait to get the best price. 

## Instructions 
First, clone the repository and go into the folder:
```
git clone https://github.com/adriancervero/flight-prices-prediction.git
cd flight-prices-prediction
```
Next, create the environment with the necessary dependencies:
```
conda env create --file environment.yml
```
Finally, execute make command for reproduce all the analysis
```
make all
```
Results are stored in /reports
\
Model is in /models
\
To launch the front-end, you can insert the following command:
```
streamlit run app.py
```
\
The whole process can also be replicated by following the jupyter notebooks available in /notebooks
## Data collection

