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

# ------------------ VARIANCE THRESHOLD ------------------ #

# Remove constant or low-variance numerical features
def apply_variance_filter(df_fit, df_to_apply, threshold=0.0, return_summary=False):
    """ Removes constant or low-variance numerical features based on training data only.
        Parameters:
            df_fit: DataFrame used to fit the VarianceThreshold selector (e.g., training set).
            df_to_apply: DataFrame to apply the feature selection (e.g., validation or test set).
            threshold: Variance threshold below which features will be removed.
            return_summary: If True, prints a summary of the features removed.
        Returns:
            df_to_apply_f: DataFrame with low-variance features removed.
    """

    df_to_apply = df_to_apply.copy()

    # Select numerical columns
    metric_cols = df_fit.select_dtypes(include='number').columns

    # Fit selector on training data
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df_fit[metric_cols])

    # Get kept feature names
    kept_features = metric_cols[selector.get_support()]

    # Apply to df_to_apply
    df_to_apply = df_to_apply[kept_features]

    # Print summary if requested
    if return_summary:
        print(f"Total features kept: {len(kept_features)}")
        print(f"Features selected: {kept_features.tolist()}")
        print(f"Nº Features eliminated: {len(metric_cols) - len(kept_features)}")

    return df_to_apply


# ------------------ HIGHLY CORRELATED FEATURES ------------------ #

# Remove one feature from each pair of highly correlated numerical features
def remove_highly_correlated_features(df_fit, df_to_apply, threshold = 0.9, return_summary=False):
    """ Removes one feature from each pair of highly correlated numerical features based on training data only.
        Parameters:
            df_fit: DataFrame used to compute correlations (e.g., training set).
            df_to_apply: DataFrame to apply the feature selection (e.g., validation or test set).
            threshold: Correlation threshold above which one feature from the pair will be removed.
            return_summary: If True, prints a summary of the features removed.
        Returns:
            df_to_apply: DataFrame with highly correlated features removed.
        """
    

    df_to_apply = df_to_apply.copy()

    # Select only numerical columns
    num_cols = df_fit.select_dtypes(include='number').columns
    
    # Compute absolute correlation matrix
    corr_matrix = df_fit[num_cols].corr(method='spearman').abs()
    
    # Create mask for the upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify features to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Identify features to keep
    cols_to_keep_2 = [col for col in df_to_apply.columns if col not in to_drop]

    # Apply to df_to_apply
    df_to_apply = df_to_apply[cols_to_keep_2]

    # Print summary if requested
    if return_summary:
        print(f"Total features kept: {len(num_cols) - len(to_drop)}")
        print(f"Features selected: {cols_to_keep_2}")
        print(f"Nº Features eliminated: {len(to_drop)}")
    
    return df_to_apply



# ------------------ CORRELATION WITH THE TARGET ------------------ #

# Select numerical features based on Spearman correlation with the target variable
def spearman_correlation_selection(df_fit, df_to_apply, target, threshold=0.3, return_summary=False):
    """ Selects numerical features based on Spearman correlation with the target variable.
        Parameters:
            df_fit: DataFrame used to compute correlations (e.g., training set).
            df_to_apply: DataFrame to apply the feature selection (e.g., validation or test set).
            target: Series or array-like target variable corresponding to df_fit.
            threshold: Minimum absolute Spearman correlation required to keep a feature.
            return_summary: If True, prints a summary of the selection process.
        Returns:
            DataFrame with selected features based on Spearman correlation.
    """

    df_to_apply = df_to_apply.copy()

    # Select numerical columns
    metric_cols = df_fit.select_dtypes(include='number').columns.tolist()

    # Compute Spearman correlation for all features with the target
    corr = df_fit[metric_cols].apply(lambda feat: spearmanr(feat, target)[0])
    # Absolute values and sort descending
    corr = corr.abs().sort_values(ascending=False)

    # Select features with Spearman correlation above the threshold
    cols_to_keep_3 = corr[corr > threshold].index.tolist()  

    df_to_apply = df_to_apply[cols_to_keep_3]

    # Print summary if requested
    if return_summary:
        print(f"Total features kept: {len(cols_to_keep_3)}")
        print(f"Features selected by Spearman method: {cols_to_keep_3}")
        print(f"Nº Features eliminated: {len(metric_cols) - len(cols_to_keep_3)}")

    return df_to_apply






# ------------------------------------------------------ WRAPPED METHODS -------------------------------------------- #

# ------------------ RFE - Recursive Feature Elimination ------------------ #

# Select features using Recursive Feature Elimination (RFE) method
def rfe_selection(df_fit, df_to_apply, target, rfe_model, n_features, return_summary = False):
    """ Selects features using Recursive Feature Elimination (RFE) method.
        Parameters:
            df_fit: DataFrame used to fit the RFE selector (e.g., training set).
            df_to_apply: DataFrame to apply the feature selection (e.g., validation or test set).
            target: Series or array-like target variable corresponding to df_fit.
            rfe_model: Estimator object to use for RFE.
            n_features: Number of features to select.
            return_summary: If True, prints a summary of the selection process.
        Returns:
            DataFrame with selected features based on RFE method.
    """
    # Initialize RFE with the specified model and number of features
    rfe = RFE(estimator = rfe_model, n_features_to_select = n_features)

    # Fit RFE on the training data
    fitted_rfe = rfe.fit(df_fit, target)

    # Get selected feature names
    cols_to_keep_4 = df_fit.columns[fitted_rfe.support_].tolist()

    # Print summary if requested
    if return_summary:
        print(f"Total features kept: {len(cols_to_keep_4)}")
        print(f"Features selected by RFE method: {cols_to_keep_4}")
        print(f"Nº Features eliminated: {df_fit.shape[1] - len(cols_to_keep_4)}")

    return cols_to_keep_4




# ------------------------------------------------------ EMBEDDED METHODS -------------------------------------------- #

# ------------------ LASSO ------------------ #

# Select features using Lasso regression method
def lasso_selection(df_fit, df_to_apply, target, threshold, return_summary=False):
    """ Selects features using Lasso regression method.
        Parameters:
            df_fit: DataFrame used to fit the Lasso model (e.g., training set).
            df_to_apply: DataFrame to apply the feature selection (e.g., validation or test set).
            target: Series or array-like target variable corresponding to df_fit.
            threshold: Coefficient threshold below which features will be removed.
            return_summary: If True, prints a summary of the selection process.
        Returns:
            DataFrame with selected features based on Lasso method.
    """
    # Fit Lasso model on the training data 
    fitted_lasso = Lasso().fit(df_fit, target)

    # Get selected feature names
    cols_to_keep_5 = df_fit.columns[np.abs(fitted_lasso.coef_) > threshold].tolist()

    # Print summary if requested
    if return_summary:
        print(f"Total features kept: {len(cols_to_keep_5)}")
        print(f"Features selected by Lasso method: {cols_to_keep_5}")
        print(f"Nº Features eliminated: {df_fit.shape[1] - len(cols_to_keep_5)}")
    
    return cols_to_keep_5



# --------------------------------------------- COMPARISON BETWEEN MODELS --------------------------------------- #

def comparison_feature_selection(df, filter_cols, wrapped_cols, embedded_cols, n_agreed, return_summary=False):
    # Create comparison table for feature selection methods
    comparison_df = pd.DataFrame({'Feature': df.columns})
    comparison_df['Lasso'] = comparison_df['Feature'].isin(embedded_cols).astype(int)
    comparison_df['RFE'] = comparison_df['Feature'].isin(wrapped_cols).astype(int)
    comparison_df['Spearman'] = comparison_df['Feature'].isin(filter_cols).astype(int)

    # Add 'Sum' column to count how many methods selected each feature
    comparison_df['Sum'] = comparison_df[['Lasso', 'RFE', 'Spearman']].sum(axis=1)

    # Select features agreed upon by all three methods
    final_features = comparison_df[comparison_df['Sum'] >= n_agreed]['Feature'].tolist()

    # Print summary if requested
    if return_summary:
        print("Total features before selection:", df.shape[1])
        print(f"Total features selected by at least {n_agreed} methods: {len(final_features)}")
        print(f"Selected features: {final_features}")

    return final_features

