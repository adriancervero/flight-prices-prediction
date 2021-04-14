from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

models = {
    "decision_tree_mse": DecisionTreeRegressor(
        criterion="mse"
    ),
    "decision_tree_poisson": DecisionTreeRegressor(
        criterion="poisson"
    ),

    "rf": RandomForestRegressor(n_jobs=-1),
    'lr': LinearRegression(),
}


param_grid = {
    "rf":{
        "model__n_estimators": np.arange(50, 100, 10),
        "model__max_depth": np.arange(1, 31),
        "model__criterion": ["mse", "mae"]
    }
}