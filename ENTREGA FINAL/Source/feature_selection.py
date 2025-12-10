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
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error


# embedded methods
from sklearn.linear_model import Lasso


# ------------------------------------------------------ FILTER METHODS -------------------------------------------- #

# ------------------ VARIANCE THRESHOLD ------------------ #

# Remove constant or low-variance numerical features
def apply_variance_filter(df, threshold=0.0, return_summary=False):
    """ Removes constant or low-variance numerical features.
        Parameters:
            df: DataFrame to apply the feature selection.
            threshold: Variance threshold below which features will be removed.
            return_summary: If True, prints a summary of the selection process.
        Returns:
            cols_to_keep_1: List of feature names kept after applying variance threshold.
    """

    # Select numerical columns
    metric_cols = df.select_dtypes(include='number').columns

    # Fit selector on training data
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[metric_cols])

    # Get kept feature names
    cols_to_keep_1 = metric_cols[selector.get_support()]

    # Print summary if requested
    if return_summary:
        print(f"Total features kept: {len(cols_to_keep_1)}")
        print(f"Features selected: {cols_to_keep_1.tolist()}")
        print(f"Nº Features eliminated: {len(metric_cols) - len(cols_to_keep_1)}")

    return cols_to_keep_1


# ------------------ HIGHLY CORRELATED FEATURES ------------------ #

# Remove one feature from each pair of highly correlated numerical features
def remove_highly_correlated_features(df, threshold = 0.9, return_summary=False):
    """ Removes one feature from each pair of highly correlated numerical features.
        Parameters:
            df: DataFrame to apply the feature selection.
            threshold: Correlation threshold above which one feature from the pair will be removed.
            return_summary: If True, prints a summary of the selection process.
        Returns:
            cols_to_keep_2: List of feature names kept after removing highly correlated features.
    """

    # Select only numerical columns
    num_cols = df.select_dtypes(include='number').columns
    
    # Compute absolute correlation matrix
    corr_matrix = df[num_cols].corr(method='spearman').abs()
    
    # Create mask for the upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identify features to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Identify features to keep
    cols_to_keep_2 = [col for col in df.columns if col not in to_drop]

    # Print summary if requested
    if return_summary:
        print(f"Total features kept: {len(num_cols) - len(to_drop)}")
        print(f"Features selected: {cols_to_keep_2}")
        print(f"Nº Features eliminated: {len(to_drop)}")
    
    return cols_to_keep_2



# ------------------ CORRELATION WITH THE TARGET ------------------ #

# Select numerical features based on Spearman correlation with the target variable
def spearman_correlation_selection(df, target, threshold=0.3, return_summary=False):
    """ Selects numerical features based on Spearman correlation with the target variable.
        Parameters:
            df: DataFrame to apply the feature selection.
            target: Series or array-like target variable corresponding to df.
            threshold: Correlation threshold above which features will be kept.
            return_summary: If True, prints a summary of the selection process.
        Returns:
            cols_to_keep_3: List of feature names kept after applying Spearman correlation selection.
    """

    # Select numerical columns
    metric_cols = df.select_dtypes(include='number').columns.tolist()

    # Compute Spearman correlation for all features with the target
    corr = df[metric_cols].apply(lambda feat: spearmanr(feat, target)[0])
    # Absolute values and sort descending
    corr = corr.abs().sort_values(ascending=False)

    # Select features with Spearman correlation above the threshold
    cols_to_keep_3 = corr[corr > threshold].index.tolist()  

    # Print summary if requested
    if return_summary:
        print(f"Total features kept: {len(cols_to_keep_3)}")
        print(f"Features selected by Spearman method: {cols_to_keep_3}")
        print(f"Nº Features eliminated: {len(metric_cols) - len(cols_to_keep_3)}")

    return cols_to_keep_3






# ------------------------------------------------------ WRAPPED METHODS -------------------------------------------- #

# ------------------ RFE - Recursive Feature Elimination ------------------ #

# # Select features using Recursive Feature Elimination (RFE) method
# def rfe_selection(df, target, rfe_model, n_features, return_summary = False):
#     """ Selects features using Recursive Feature Elimination (RFE) method.
#         Parameters:
#             df: DataFrame to apply the feature selection.
#             target: Series or array-like target variable corresponding to df.
#             rfe_model: Estimator object to use for RFE.
#             n_features: Number of features to select.
#             return_summary: If True, prints a summary of the selection process.
#         Returns:
#             DataFrame with selected features based on RFE method.
#     """
#     # Initialize RFE with the specified model and number of features
#     rfe = RFE(estimator = rfe_model, n_features_to_select = n_features)

#     # Fit RFE on the training data
#     fitted_rfe = rfe.fit(df, target)

#     # Get selected feature names
#     cols_to_keep_4 = df.columns[fitted_rfe.support_].tolist()

#     # Print summary if requested
#     if return_summary:
#         print(f"Total features kept: {len(cols_to_keep_4)}")
#         print(f"Features selected by RFE method: {cols_to_keep_4}")
#         print(f"Nº Features eliminated: {df.shape[1] - len(cols_to_keep_4)}")

#     return cols_to_keep_4


def rfe_selection(df_fit, df_to_apply, y_fit, y_apply, model, return_summary = False):

    #nº of features
    nof_list=np.arange(1,len(df_fit.columns)+1)            
    low_score = math.inf
    #Variable to store the optimum features
    nof=0           
    train_score_list =[]
    val_score_list = []

    for n in range(len(nof_list)):
        model_instance = model(random_state=42)
        
        rfe = RFE(estimator = model_instance,n_features_to_select = nof_list[n])
        X_train_rfe = rfe.fit_transform(df_fit,y_fit)
        X_val_rfe = rfe.transform(df_to_apply)
        model_instance.fit(X_train_rfe,y_fit)
        
        #storing results on training data
        train_score = model_instance.score(X_train_rfe,y_fit)
        train_score_list.append(train_score)
        
        #storing results on validation data
        val_score = model_instance.score(X_val_rfe,y_apply)
        val_score_list.append(val_score)

        #calculating MAE
        train_mae = mean_absolute_error(y_fit, model_instance.predict(X_train_rfe))
        val_mae = mean_absolute_error(y_apply, model_instance.predict(X_val_rfe))
        
        #check best score
        if val_mae < low_score:
            low_score = val_mae
            nof = nof_list[n]
            features_to_select = pd.Series(rfe.support_, index = df_fit.columns)

    cols_to_keep_4 = features_to_select[features_to_select==True].index.tolist()

    if return_summary:
        print("Total Features Kept: %d" %nof)
        print(f"Features Selected by RFE Method: {cols_to_keep_4}")
        print("Score with %d features: %f" % (nof, low_score))
        print(f"Nº Features eliminated: {df_fit.shape[1] - len(cols_to_keep_4)}")

    return cols_to_keep_4







# ------------------------------------------------------ EMBEDDED METHODS -------------------------------------------- #

# ------------------ LASSO ------------------ #

# Select features using Lasso regression method
def lasso_selection(df, target, threshold, return_summary=False):
    """ Selects features using Lasso regression method.
        Parameters:
            df: DataFrame to apply the feature selection.
            target: Series or array-like target variable corresponding to df.
            threshold: Coefficient threshold below which features will be removed.
            return_summary: If True, prints a summary of the selection process.
        Returns:
            DataFrame with selected features based on Lasso method.
    """
    # Fit Lasso model on the training data 
    fitted_lasso = Lasso(max_iter = 15000,random_state=42).fit(df, target)

    # Get selected feature names
    cols_to_keep_5 = df.columns[np.abs(fitted_lasso.coef_) > threshold].tolist()

    # Print summary if requested
    if return_summary:
        print(f"Total features kept: {len(cols_to_keep_5)}")
        print(f"Features selected by Lasso method: {cols_to_keep_5}")
        print(f"Nº Features eliminated: {df.shape[1] - len(cols_to_keep_5)}")
    
    return cols_to_keep_5



# --------------------------------------------- COMPARISON BETWEEN MODELS --------------------------------------- #

def comparison_feature_selection(df, filter_cols, wrapped_cols, embedded_cols, feature_importance_cols, n_agreed, return_summary=False):
    # Create comparison table for feature selection methods
    comparison_df = pd.DataFrame({'Feature': df.columns})
    comparison_df['Lasso'] = comparison_df['Feature'].isin(embedded_cols).astype(int)
    comparison_df['RFE'] = comparison_df['Feature'].isin(wrapped_cols).astype(int)
    comparison_df['Spearman'] = comparison_df['Feature'].isin(filter_cols).astype(int)
    comparison_df['Feature Importance'] = comparison_df['Feature'].isin(feature_importance_cols).astype(int)

    # Add 'Sum' column to count how many methods selected each feature
    comparison_df['Sum'] = comparison_df[['Lasso', 'RFE', 'Spearman', 'Feature Importance']].sum(axis=1)

    # Select features agreed upon by all three methods
    final_features = comparison_df[comparison_df['Sum'] >= n_agreed]['Feature'].tolist()

    # Print summary if requested
    if return_summary:
        print("Total features before selection:", df.shape[1])
        print(f"Total features selected by at least {n_agreed} methods: {len(final_features)}")
        print(f"Selected features: {final_features}")

    return final_features, comparison_df

