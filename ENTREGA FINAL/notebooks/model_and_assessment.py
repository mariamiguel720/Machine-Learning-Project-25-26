# ---------------------LIBRARIES --------------------- #
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Linear Regression - OLS, ridge, lasso, elastic net regression
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.ensemble import RandomForestRegressor


#Model evaluation
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, mean_absolute_percentage_error
import statsmodels.api as sm

# ----------------- MODEL AND ASSESSMENT ----------------- #

def train_model(model, X_train, y_train):
    """ Train the given model on provided data."""
    model.fit(X_train, y_train) # Fit the model
    return model

def evaluate_model(model, X, y):
    """Evaluate a regression model using multiple metrics:
    R2, Adjusted R2, MAE, MSE, RMSE, MedAE, MAPE."""
    n_samples = X.shape[0] # number of observations
    n_features = X.shape[1] # number of features
    
    y_pred = model.predict(X) # predicted values
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = rmse = np.sqrt(mse)
    medae = median_absolute_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)

# Compile metrics into a dictionary
    metrics_dict = {
        'R2': r2,
        'Adjusted R2': adj_r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MedAE': medae,
        'MAPE (%)': mape
    }
    return metrics_dict

# Create comparison DataFrame
def comparison_metrics(model, X_train, y_train, X_val, y_val):
    """Create a comparison DataFrame for training and validation metrics."""

    metrics_train = evaluate_model(model, X_train, y_train)
    metrics_val = evaluate_model(model, X_val, y_val)
    
    comparison_df = pd.DataFrame({
        'Metric': list(metrics_train.keys()),
        'Train': list(metrics_train.values()),
        'Validation': list(metrics_val.values()),
        'Iteration': model.n_iter_
    })
    return comparison_df