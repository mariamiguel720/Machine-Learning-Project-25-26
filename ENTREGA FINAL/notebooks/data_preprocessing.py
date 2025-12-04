# ----------------- LIBRARIES ----------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns
from math import ceil
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler


# ----------------- CORRECT VALUES ----------------- #

#def correct_values(train):

# ----------------- OUTLIERS TREATMENT ----------------- #

# Function to treat outliers based on custom rules
def treat_outliers_custom(df_fit, df_to_apply):

    df_to_apply = df_to_apply.copy()

    # fixed limits
    df_to_apply['year'] = df_to_apply['year'].clip(lower=1990, upper=2020)
    df_to_apply['engineSize'] = df_to_apply['engineSize'].clip(lower=0.9, upper=5.5)
    df_to_apply['previousOwners'] = df_to_apply['previousOwners'].clip(lower=0, upper=8)

    # IQR based limits
    for col in ['mileage', 'tax', 'mpg']:
        # Calculate IQR based limits from df_fit
        q1 = df_fit[col].quantile(0.25)
        q3 = df_fit[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 2.2 * iqr
        upper = q3 + 2.2 * iqr

        # Cap the outliers
        df_to_apply[col] = df_to_apply[col].clip(lower=lower, upper=upper)

    return df_to_apply

# ----------------- ENCODING ----------------- #

# Function to encode categorical features using One-Hot and Ordinal Encoding
def encoding_features(df_fit, df_to_apply, method):
    """ Encode categorical features using One-Hot. 
    
    Parameters:
    df_fit: DataFrame to fit the encoders (training set)
    df_to_apply: DataFrame to apply the fitted encoders (training/validation/test set)
    method: 'ordinal' for Ordinal Encoding, 'onehot' for One-Hot Encoding
        """

    df_transformed = df_to_apply.copy()

    # Define categorical columns from df_fit
    cat_cols = df_fit.select_dtypes(exclude=['number']).columns.tolist()

    # Ensure columns exist in df_to_apply
    cat_cols = [c for c in cat_cols if c in df_to_apply.columns]

    if not cat_cols:
        return df_transformed
    
    if method == 'ordinal':
        ordinal_encoder = OrdinalEncoder()
        ordinal_fit = ordinal_encoder.fit(df_fit[cat_cols])
        df_transformed[cat_cols] = ordinal_fit.transform(df_transformed[cat_cols])

    elif method == 'onehot':
        one_hot = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore') #sparse_output=False outputs a numpy array, not a sparse matrix
        onehot_fit = one_hot.fit(df_fit[cat_cols])
        onehot_transformed = onehot_fit.transform(df_transformed[cat_cols])

        one_hot_feat_names = onehot_fit.get_feature_names_out(cat_cols)
        one_hot_feat_names = ['ohe_' + name for name in one_hot_feat_names]

        encoded_df = pd.DataFrame(onehot_transformed, index=df_to_apply.index, columns=one_hot_feat_names)

        # Drop original categorical columns & concatenate encoded ones
        df_transformed = df_transformed.drop(columns=cat_cols)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
            
        return df_transformed
    
    else:
        raise ValueError("Invalid method. Choose 'ordinal' or 'onehot'.")


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

    df_to_apply = df_to_apply.copy()

    if method == 'minmax':
        #scale your data using MinMaxScaler[0,1]
        min_max = MinMaxScaler().fit(df_fit[metric_cols])
        # Transform the data from df_to_apply by applying the scale obtained in the previous command
        scaled_array = min_max.transform(df_to_apply[metric_cols])
    elif method == 'minmax2':
        # Create a MinMaxScaler instance that will range between -1 and 1 and fit to your train data
        min_max2 = MinMaxScaler(feature_range=(-1, 1)).fit(df_fit[metric_cols])
        # Transform your the data from df_to_apply by applying the scale obtained in the previous command
        scaled_array = min_max2.transform(df_to_apply[metric_cols])
    elif method == 'standard':
        # Create a StandardScaler instance and fit to your train data
        standard = StandardScaler().fit(df_fit[metric_cols])
        # Transform your the data from df_to_apply by applying the scale obtained in the previous command
        scaled_array = standard.transform(df_to_apply[metric_cols])
    else: 
        robust = RobustScaler().fit(df_fit[metric_cols])
        # Transform your the data from df_to_apply by applying the scale obtained in the previous command
        scaled_array = robust.transform(df_to_apply[metric_cols])

    # Replace the original metric columns with the scaled values
    df_to_apply[metric_cols] = scaled_array

    return df_to_apply

# ----------------- MISSING VALUES IMPUTATION ----------------- #

# ---------- SIMPLE IMPUTATION ---------- #

# Function to impute missing values based on specified methods
def simple_imputation(df_fit, df_to_apply):
    """ 
    Imputes missing values in the df_to_apply using SimpleImputer with 'median' for numeric columns and 'most_frequent' for categorical columns.
    Parameters:
        df_fit: dataframe to fit the imputation
        df_to_apply: dataframe to apply the imputation
    """

    # make a copy of the dataframe to avoid modifying the original data
    df_to_apply = df_to_apply.copy()

    # Define categorical and numerical columns from df_fit
    metric_cols = df_fit.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df_fit.select_dtypes(exclude=['number']).columns.tolist()

    # Ensure columns exist in df_to_apply
    metric_cols = [m for m in metric_cols if m in df_to_apply.columns]
    cat_cols = [c for c in cat_cols if c in df_to_apply.columns]

    # Impute missing values for categorical with mode
    if cat_cols:
        imputer = SimpleImputer(strategy='most_frequent').fit(df_fit[cat_cols])
        df_to_apply[cat_cols] = imputer.transform(df_to_apply[cat_cols])

    # Impute missing values for metric with median
    if metric_cols:
        imputer = SimpleImputer(strategy='median').fit(df_fit[metric_cols])
        df_to_apply[metric_cols] = imputer.transform(df_to_apply[metric_cols])

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
    metric_cols = df_fit.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df_fit.select_dtypes(exclude=['number']).columns.tolist()

    # Ensure columns exist in df_to_apply
    metric_cols = [c for c in metric_cols if c in df_to_apply.columns]
    cat_cols = [c for c in cat_cols if c in df_to_apply.columns]

    # Impute metric columns using KNNImputer
    for col in metric_cols:
     # Fit the KNNImputer on the training set
            knn_imputer = KNNImputer(n_neighbors=neighbors, weights='distance')
            knn_imputer.fit(df_fit[[col]])
            # Transform df to apply 
            df_to_apply[[col]] = pd.DataFrame(knn_imputer.transform(df_to_apply[[col]]),
                                            columns=[col],
                                            index=df_to_apply.index)

    return df_to_apply


# ----- MISSING VALUES JUNTAS ----- #

def impute_missing(df_fit, df_to_apply, method="simple", neighbors=5):
    """
    Imputes missing values in df_to_apply using specified method for metric cols.
    Imputes categorical cols using most frequent value.
    Parameters:
        df_fit: DataFrame to fit the imputation models (training set)
        df_to_apply: DataFrame to apply the imputation (training/validation/test set)
        method: 'simple' for median imputation, 'knn' for KNN imputation
        neighbors: number of neighbors for KNN imputation
    """

    df_to_apply = df_to_apply.copy()

    # define categorical and numerical columns
    metric_cols = df_fit.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df_fit.select_dtypes(exclude=['number']).columns.tolist()

    # Ensure columns exist in df_to_apply
    metric_cols = [c for c in metric_cols if c in df_to_apply.columns]
    cat_cols = [c for c in cat_cols if c in df_to_apply.columns]

    # Imputation for categorical columns: most frequent value
    if cat_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        imp_cat.fit(df_fit[cat_cols])
        df_to_apply[cat_cols] = imp_cat.transform(df_to_apply[cat_cols])

    # Imputation for metric columns: KNN or median
    if metric_cols:
        if method == "knn":
            imp_num = KNNImputer(n_neighbors=neighbors, weights="distance")
        elif method == "simple":
            imp_num = SimpleImputer(strategy="median")
        else :
            raise ValueError("Invalid method. Choose 'simple' or 'knn'.")

        imp_num.fit(df_fit[metric_cols])
        df_to_apply[metric_cols] = imp_num.transform(df_to_apply[metric_cols])

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
