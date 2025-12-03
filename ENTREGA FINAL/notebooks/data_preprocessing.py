# ----------------- LIBRARIES ----------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler


# ----------------- CORRECT VALUES ----------------- #

#def correct_values(train):

# ----------------- OUTLIERS TREATMENT ----------------- #

def treat_outliers_custom(df_to_apply, col_thresholds):
    """
    Caps outliers per column based on custom IQR multipliers or manual bounds.
    """
    
    df_clean = df_to_apply.copy()

    for col, rules in col_thresholds.items():
        # Compute quartiles and IQR
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1

        # Use IQR multiplier OR manual bounds
        if "iqr_mult" in rules:
            mult = rules['iqr_mult']
            lower = q1 - mult * iqr
            upper = q3 + mult * iqr
        else:
            lower = rules.get('lower', -np.inf)
            upper = rules.get('upper', np.inf)

        # Cap the outliers
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)

    return df_clean

# ----------------- ENCODING ----------------- #

# Function to encode categorical features using One-Hot and Ordinal Encoding
def encoding_features(df_fit, df_to_apply, ordinal_cols=None, one_hot_cols=None,):
    """ Encode categorical features using One-Hot and Ordinal Encoding. 
    
    Parameters:
    df_fit: DataFrame to fit the encoders (training set)
    df_to_apply: DataFrame to apply the fitted encoders (training/validation/test set)
    one_hot_cols: List of columns to use One-Hot Encoding
    ordinal_cols: List of columns to use Ordinal Encoding
    """

    # -------- ORDINAL ENCODING --------
    if ordinal_cols:
        method1 = OrdinalEncoder()
        ordinal_fit = method1.fit(df_fit[ordinal_cols])
        df_to_apply[ordinal_cols] = ordinal_fit.transform(df_to_apply[ordinal_cols])

    # -------- ONE HOT ENCODING --------
    if one_hot_cols:
        method2 = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore') #sparse_output=False outputs a numpy array, not a sparse matrix
        onehot_fit = method2.fit(df_fit[one_hot_cols])
        onehot_transformed = onehot_fit.transform(df_to_apply[one_hot_cols])

        one_hot_feat_names = onehot_fit.get_feature_names_out(one_hot_cols)
        encoded_df = pd.DataFrame(onehot_transformed, index=df_to_apply.index, columns=one_hot_feat_names)

        # Drop original categorical columns & concatenate encoded ones
        df_to_apply = df_to_apply.drop(columns=one_hot_cols)
        df_to_apply = pd.concat([df_to_apply, encoded_df], axis=1)
        
    return df_to_apply


# ----------------- SCALING ----------------- #

# Function to scale features using different scaling methods
def scaling_features(df_fit, df_to_apply, metric_cols, method):
    """ Scales the features of the train and validation sets according to the specified method.
    Args:
        df_fit (pd.DataFrame): The dataframe to fit the scaler.
        df_to_apply (pd.DataFrame): The dataframe to apply the scaler.
        metric_cols (list): List of numeric columns to scale.
        method (str): The scaling method to use. Options are 'minmax' - between 0 and 1, 'minmax2' - between -1 and 1, 
        'standard', and 'robust'.
    Returns:
        scaled_df_to_apply (np.ndarray): The scaled dataframe to which the scaler is applied.
    """

    if method == 'minmax':
        #scale your data using MinMaxScaler[0,1]
        min_max = MinMaxScaler().fit(df_fit[metric_cols])
        # Transform the data from df_to_apply by applying the scale obtained in the previous command
        scaled_df_to_apply = min_max.transform(df_to_apply[metric_cols])
    elif method == 'minmax2':
        # Create a MinMaxScaler instance that will range between -1 and 1 and fit to your train data
        min_max = MinMaxScaler(feature_range=(-1, 1)).fit(df_fit[metric_cols])
        # Transform your the data from df_to_apply by applying the scale obtained in the previous command
        scaled_df_to_apply = min_max.transform(df_to_apply[metric_cols])
    elif method == 'standard':
        # Create a StandardScaler instance and fit to your train data
        standard = StandardScaler().fit(df_fit[metric_cols])
        # Transform your the data from df_to_apply by applying the scale obtained in the previous command
        scaled_df_to_apply = standard.transform(df_to_apply[metric_cols])
    else: 
        robust = RobustScaler().fit(df_fit[metric_cols])
        # Transform your the data from df_to_apply by applying the scale obtained in the previous command
        scaled_df_to_apply = robust.transform(df_to_apply[metric_cols])
    return scaled_df_to_apply

# ----------------- MISSING VALUES IMPUTATION ----------------- #

# ---------- SIMPLE IMPUTATION ---------- #
# Function to impute missing values based on specified methods
def simple_imputation(df_fit, df_to_apply):
    """ 
    Imputes missing values in the dataframe using median/mode for missing values.
    Parameters:
        df_fit: dataframe to fit the imputation
        df_to_apply: dataframe to apply the imputation
        threshold: % below which missing values are considered low
    """
    # make a copy of the dataframe to avoid modifying the original data
    df_fit = df_fit.copy()
    df_to_apply = df_to_apply.copy()

    # define categorical and numerical columns
    categorical = ['Brand', 'model', 'transmission', 'fuelType', 'hasDamage', 'is_recent_car', 'mileage_category',
                   'is_hybrid_or_electric', 'is_automatic', 'is_first_owner']
    numerical = df_fit.drop(categorical, axis=1).columns.tolist()

    for col in numerical:
        median_value = df_fit[col].median()
        df_to_apply[col] = df_to_apply[col].fillna(median_value)
    for col in categorical:
        mode_value = df_fit[col].mode().iloc[0]
        df_to_apply[col] = df_to_apply[col].fillna(mode_value)

    return df_to_apply

# ---------- KNN IMPUTATION ---------- #

def knn_imputation(df_fit, df_to_apply, neighbors=5):
    """ 
    Imputes missing values in the df_to_apply using KNN imputation.
    Parameters:
        df_fit: train dataframe (used to fit imputation models)
        df_to_apply: dataframe to apply the imputation
        neighbors: n_neighbors for KNN
    """

    # make a copy of the dataframe to avoid modifying the original data
    df_fit = df_fit.copy()
    df_to_apply = df_to_apply.copy()

    # define categorical and numerical columns
    categorical = ['Brand', 'model', 'transmission', 'fuelType', 'hasDamage', 'is_recent_car', 'mileage_category',
                   'is_hybrid_or_electric', 'is_automatic', 'is_first_owner']
    numerical = df_fit.drop(categorical, axis=1).columns.tolist()

    for col in numerical:
     # Fit the KNNImputer on the training set
            knn_imputer = KNNImputer(n_neighbors=neighbors, weights='distance')
            knn_imputer.fit(df_fit[[col]])
            # Transform df to apply 
            df_to_apply[[col]] = pd.DataFrame(knn_imputer.transform(df_to_apply[[col]]),
                                            columns=[col],
                                            index=df_to_apply.index)
    return df_to_apply

# ----------------- DATA PREPARATION COMPILATION ----------------- #

# def data_preparation(df_fit, df_to_apply, col_thresholds, ordinal_cols, one_hot_cols, metric_cols, scaling_method):
# # def data_preparation(df_fit, df_to_apply, col_thresholds, ordinal_cols, one_hot_cols, metric_cols, scaling_method, neighbors=5):

#     """
#     Compiles data preparation steps: outliers treatment, encoding, scaling, and missing values imputation.
#     Parameters:
#     df_fit: DataFrame to fit the transformations (training set)
#     df_to_apply: DataFrame to apply the transformations (training/validation/test set)
#     col_thresholds: Dictionary with outlier treatment rules per column
#     ordinal_cols: List of columns to use Ordinal Encoding
#     one_hot_cols: List of columns to use One-Hot Encoding
#     metric_cols: List of numeric columns to scale
#     scaling_method: Method to use for scaling (e.g., 'standard', 'minmax')
#     """

#     # Criar função de correção dados manuais
#     #df_to_apply = correct_values(parametros)

#     # Outliers Treatment
#     df_to_apply = treat_outliers_custom(df_to_apply, col_thresholds)

#     # Encoding
#     df_to_apply = encoding_features(df_fit, df_to_apply, ordinal_cols, one_hot_cols)

#     # Scaling
#     scaled_metrics = scaling_features(df_fit, df_to_apply, metric_cols, scaling_method)

#     # Missing Values Imputation
#     df_to_apply = simple_imputation(df_fit, df_to_apply)
#     #df_to_apply = knn_imputation(df_fit, df_to_apply, neighbors=5)

#     return df_to_apply
