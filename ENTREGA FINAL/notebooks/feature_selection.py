# ------------------------------------------------------ LIBRARIES -------------------------------------------- #
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import ceil
import math


#filter methods
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import spearmanr
# spearman 
from sklearn.feature_selection import SelectKBest, f_regression

# mutual information
from sklearn.feature_selection import mutual_info_classif

#wrapper methods
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE


# embedded methods
from sklearn.linear_model import Lasso


# ------------------------------------------------------ FILTER METHODS -------------------------------------------- #

# ------------------ Variance Threshold ------------------ #

# Remove constant or low-variance numerical features
def apply_variance_filter(df_fit, df_to_apply, threshold=0.0):
    """ Removes constant or low-variance numerical features based on training data only.
        Parameters:
            df_fit: DataFrame used to fit the VarianceThreshold selector (e.g., training set).
            df_to_apply: DataFrame to apply the feature selection (e.g., validation or test set).
            threshold: Variance threshold below which features will be removed.
        Returns:
            df_to_apply_f: DataFrame with low-variance features removed.
    """

    df_to_apply_f = df_to_apply.copy()

    # Select numerical columns
    num_cols = df_fit.select_dtypes(include='number').columns

    # Fit selector on training data
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df_fit[num_cols])

    # Get kept feature names
    kept_features = num_cols[selector.get_support()]
    removed_features = [c for c in num_cols if c not in kept_features]

    # Keep same columns in all datasets
    cols_to_keep = kept_features.tolist() + df_fit.select_dtypes(exclude='number').columns.tolist()
    df_to_apply = df_to_apply[cols_to_keep]

    # Display summary
    print(f"Variance threshold = {threshold}")
    print(f"Removed {len(removed_features)} feature(s): {removed_features}" if removed_features else "No features removed.")
    
    return df_to_apply


# ------------------ Highly Correlated Features ------------------ #

# Remove one feature from each pair of highly correlated numerical features
def remove_highly_correlated_features(df_fit, df_to_apply, threshold):
    """ Removes one feature from each pair of highly correlated numerical features based on training data only.
        Parameters:
            df_fit: DataFrame used to compute correlations (e.g., training set).
            df_to_apply: DataFrame to apply the feature selection (e.g., validation or test set).
            threshold: Correlation threshold above which one feature from the pair will be removed.
        Returns:
            df_to_apply: DataFrame with highly correlated features removed.
            to_drop: List of features that were removed."""
    
    # Select only numerical columns
    num_cols = df_fit.select_dtypes(include='number').columns
    
    # Compute absolute correlation matrix
    corr_matrix = df_fit[num_cols].corr(method='spearman').abs()
    
    # Create mask for the upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify features to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Correlation threshold = {threshold}")
    print(f"Features removed due to high correlation: {len(to_drop)}")
    if to_drop:
        print(f"Removed features: {to_drop}")

    # Drop the same features from all datasets
    df_to_apply = df_to_apply.drop(columns=to_drop)
    
    return df_to_apply, to_drop



# ------------------ Correlation with the target ------------------ #

