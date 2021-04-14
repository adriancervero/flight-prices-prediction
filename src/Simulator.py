import pandas as pd 
import numpy as np 
import random

def get_waiting_price(row):
    if row['predicted'] != 0:
        #idx = row['predicted'] - 1
        list_prices = row['hist_prices'].strip('][').split(', ')
        idx = len(list_prices)-row['days_until_dep']+row['predicted']
        waiting_price = float(list_prices[idx])
        if np.isnan(waiting_price):
            waiting_price = row['price']
    else:
        waiting_price = row['price']
        
    return waiting_price


def get_best_waiting_price(row):
    if row['waiting_days'] != 0:
        #idx = row['waiting_days'] - 1
        list_prices = row['hist_prices'].strip('][').split(', ')
        idx = len(list_prices)-row['days_until_dep']+row['waiting_days']
        waiting_price = float(list_prices[idx])
        if np.isnan(waiting_price):
            waiting_price = row['price']
    else:
        waiting_price = row['price']
        
    return waiting_price


class Simulator():
    def __init__(self, n, flights, pipeline):
        self.n = n
        self.flights = flights
        self.pipeline = pipeline
        
    def generate_travellers(self):
        """
        routes = self.flights['orig-dest'].unique()
        departures = self.flights['dDate'].unique()
        
        #requests = self.flights['days_until_dep'].unique()
        requests = np.arange(7, 46)
        travellers_routes = random.choices(routes, k=self.n)
        travellers_dep_date = random.choices(departures, k=self.n)
        travellers_req_date = random.choices(requests, k=self.n)
        id_traveler = np.arange(0, self.n)
        self.travellers = pd.DataFrame({'id_traveler': id_traveler,
                                        'route': travellers_routes,
                                        'departure': travellers_dep_date,
                                        'request':travellers_req_date})
        """
        self.travellers = self.flights[['orig-dest','dDate', 'days_until_dep']].sample(self.n)
        self.travellers['id_traveler'] = np.arange(0, self.n)
        
    def get_cheapest_flights(self):
        # select cheapest flight for each traveler
        #merged = pd.merge(self.travellers, self.flights, left_on=['departure', 'route', 'request'], right_on=['dDate', 'orig-dest', 'days_until_dep'])
        merged = pd.merge(self.travellers, self.flights, on=['orig-dest','dDate', 'days_until_dep'])
        #group_cols = list(self.travellers.columns)
        cheapest_indexes = merged.groupby('id_traveler')['price'].idxmin()
        self.cheapest_flights = merged.loc[cheapest_indexes]

    def prepare_data(self, data):
        num_attribs = ['days_until_dep', 'fly_duration', 'day_of_month', 'log_price', 'hops', 'competition']
        cat_attribs = ['flyFrom', 'flyTo', 'day_of_week', 'session']
    
        #data_prepared = self.pipeline.transform(data[num_attribs+cat_attribs])
        return data[num_attribs+cat_attribs]
    
    def make_predictions(self):
        data_prepared = self.prepare_data(self.cheapest_flights)
        
        predicted = self.pipeline.predict(data_prepared).round()
        self.cheapest_flights['predicted'] = predicted.round().astype(int)
        
    def compute_savings(self):
        df = self.cheapest_flights
        df['waiting_price'] = df.apply(get_waiting_price, axis=1)
        df['best_waiting_price'] = df.apply(get_best_waiting_price, axis=1)
        df['savings'] = df['price'] - df['waiting_price']
        
    def visualize_results(self):
        df = self.cheapest_flights[['price', 'predicted', 'waiting_price', 'savings']]
        
        current_sum = df['price'].sum()
        current_mean = df['price'].mean()
        
        model_sum = df['waiting_price'].sum()
        model_mean = df['waiting_price'].mean()
        
        savings_sum = df['savings'].sum()
        savings_mean = df['savings'].mean()
        
        
        
        percent = str(round(savings_sum/current_sum*100, 2)) + '%'
        
        results = {'Current': [current_sum, current_mean],
                   'Model': [model_sum, model_mean],
                   'Savings':[savings_sum, savings_mean],
                   'Percentage': [percent, percent]}
        
        df = pd.DataFrame(results, index=['Total (€)', 'Per traveler (€)'])
        df = df.round(2)
        print(df.to_string())
        
    def visualize_results_by_route(self):
        
        df = self.cheapest_flights[['orig-dest', 'price', 'waiting_price', 'savings', 'best_waiting_price']]
        by_route = df.groupby('orig-dest')[['price', 'waiting_price', 'savings', 'best_waiting_price']].agg(['sum', 'mean'])
        
        
        by_route['percentage'] = (by_route['savings']['sum']/by_route['price']['sum']*100).round(2).astype(str) + '%'
        by_route['max_percentage'] = ((by_route['price']['sum']-by_route['best_waiting_price']['sum'])/by_route['price']['sum']*100).round(2).astype(str) + '%'
        print(by_route.to_string())
        
        print('\nOnly waiting predictions:\n')
        wait = self.cheapest_flights['waiting_days'] != 0
        only_wait = self.cheapest_flights[wait]
        df = only_wait[['orig-dest', 'price', 'waiting_price', 'savings', 'best_waiting_price']]
        by_route = df.groupby('orig-dest')[['price', 'waiting_price', 'savings', 'best_waiting_price']].agg(['sum', 'mean'])
        by_route['percentage'] = (by_route['savings']['sum']/by_route['price']['sum']*100).round(2).astype(str) + '%'
        by_route['max_percentage'] = ((by_route['price']['sum']-by_route['best_waiting_price']['sum'])/by_route['price']['sum']*100).round(2).astype(str) + '%'
        print(by_route.to_string())
        
    def run(self):
        self.generate_travellers()
        self.get_cheapest_flights()
        self.make_predictions()
        self.compute_savings()
        self.visualize_results_by_route()