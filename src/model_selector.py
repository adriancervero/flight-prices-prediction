from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

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