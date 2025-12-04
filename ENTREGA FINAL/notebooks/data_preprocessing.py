# ----------------- LIBRARIES ----------------- #
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns
from math import ceil
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler


# --------------------------------------------------- CORRECT VALUES --------------------------------------------------- #

# ----------------- INVALID VALUES ----------------- #

# Function to correct invalid values in the DataFrame
def correct_wrong_values(df):
    """Corrects invalid values in the DataFrame:
    Parameters:
        df (DataFrame): The dataset
    Returns:
        df (DataFrame): The dataset with corrected values
    """
    numeric_cols = df.select_dtypes(include=['number']).columns # Select numeric columns
    for col in numeric_cols:
        if (df[col] < 0).any():  # Check for negative values
            df.loc[df[col] < 0, col] = np.nan # Replace negative values with NaN
    for col in ['year', 'previousOwners']: #Rounds numeric values: values with decimal part >= 0.5 go up, others go down.
        df[col] = df[col].astype('Int64') # Use 'Int64' to allow for NaN values
    return df 

# ----------------- CATEGORICAL CORRECTIONS ----------------- #

# Function to calculate similarity between two strings
def similar(a, b):
    """Calculates similarity ratio between two strings using SequenceMatcher."""
    return SequenceMatcher(None, a, b).ratio()


# Function to clean categorical column using difflib
def clean_with_diff(df, column, threshold_short, threshold_long):
    """
    Cleans a categorical column by grouping similar values based on defined thresholds.
    Parameters:
        df: DataFrame containing the column to clean
        column: Name of the column to clean
        threshold_short: Similarity threshold for short strings (length <= 5)
        threshold_long: Similarity threshold for long strings (length > 5)
    Returns:
        df: DataFrame with cleaned column
    """
    # Get unique values in the column
    values = df[column].dropna().unique()
    groups = []
    used = set()
    
    # Group similar values
    for v in values:
        if v in used:
            continue
        
        # Start a new group
        group = [v]
        used.add(v)
        
        # Compare with other values
        for other in values:
            if other in used:
                continue

            # Choose the thresholds based on length
            if len(v) <= 5 and len(other) <= 5:
                threshold = threshold_short
            else:
                threshold = threshold_long

            # Check similarity. If similar, add to group.
            if similar(v.lower(), other.lower()) >= threshold:
                group.append(other)
                used.add(other)

        # Append the group to the list of groups
        groups.append(group)
    
    # Create mapping for each group and replace values with mode
    mapping = {}
    for group in groups:
        moda = df[df[column].isin(group)][column].mode()[0]
        for item in group:
            mapping[item] = moda
    
    df[column] = df[column].map(mapping).fillna(df[column])
    
    return df



# --------------------------------------------------- OUTLIERS TREATMENT --------------------------------------------------- #

# Function to treat outliers based on custom rules
def treat_outliers_custom(df_fit, df_to_apply):
    """ Treats outliers in df_to_apply based on custom rules defined for each column.
    Parameters:
        df_fit (pd.DataFrame): The dataframe to fit the outlier treatment rules.
        df_to_apply (pd.DataFrame): The dataframe to apply the outlier treatment.
    Returns:
        df_to_apply (pd.DataFrame): The dataframe with treated outliers.
    """

    df_to_apply = df_to_apply.copy()

    # Fixed limits
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

# --------------------------------------------------- ENCODING --------------------------------------------------- #

# Function to encode categorical features using One-Hot Encoding
def encoding_features(df_fit, df_to_apply):
    """ Encode categorical features using One-Hot Encoding. 
    Parameters:
        df_fit: DataFrame to fit the encoders (training set)
        df_to_apply: DataFrame to apply the fitted encoders (training/validation/test set)
    Returns:
        df_transformed: DataFrame with encoded categorical features
    """

    df_transformed = df_to_apply.copy()

    # Define categorical columns from df_fit
    cat_cols = df_fit.select_dtypes(exclude=['number']).columns.tolist()

    # Apply One-Hot Encoding
    one_hot = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore') #sparse_output=False outputs a numpy array, not a sparse matrix
    onehot_fit = one_hot.fit(df_fit[cat_cols])
    onehot_transformed = onehot_fit.transform(df_transformed[cat_cols])

    # Get features names
    one_hot_feat_names = onehot_fit.get_feature_names_out(cat_cols)
    one_hot_feat_names = ['ohe_' + name for name in one_hot_feat_names]

    # Create DataFrame with encoded features
    encoded_df = pd.DataFrame(onehot_transformed, index=df_to_apply.index, columns=one_hot_feat_names)

    # Drop original categorical columns & concatenate encoded ones
    df_transformed = df_transformed.drop(columns=cat_cols)
    df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        
    return df_transformed
    

# --------------------------------------------------- SCALING --------------------------------------------------- #

# Function to scale features using different scaling methods
def scaling_features(df_fit, df_to_apply, method):
    """ Scales the features of the train and validation sets according to the specified method.
    Args:
        df_fit (pd.DataFrame): The dataframe to fit the scaler.
        df_to_apply (pd.DataFrame): The dataframe to apply the scaler.
        metric_cols (list): List of numeric columns to scale.
        method (str): The scaling method to use. Options are 'minmax' - between 0 and 1, 'minmax2' - between -1 and 1, 
        'standard', and 'robust'.
    Returns:
        df_to_apply (np.ndarray): The scaled dataframe to which the scaler is applied.
    """

    df_to_apply = df_to_apply.copy()

    metric_cols = df_fit.select_dtypes(include=['number']).columns.tolist()

    if method == 'minmax':
        # Scale your data using MinMaxScaler[0,1]
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

# --------------------------------------------------- MISSING VALUES IMPUTATION --------------------------------------------------- #

# Function to impute missing values using Simple or KNN imputation
def impute_missing(df_fit, df_to_apply, method="simple", neighbors=5):
    """
    Imputes missing values in df_to_apply using specified method for metric cols.
    Imputes categorical cols using most frequent value.
    Parameters:
        df_fit: DataFrame to fit the imputation models (training set)
        df_to_apply: DataFrame to apply the imputation (training/validation/test set)
        neighbors: number of neighbors for KNN imputation
    Returns:
        df_to_apply: DataFrame with imputed missing values
    """
    # Create a copy of df_to_apply to avoid modifying the original DataFrame
    df_to_apply = df_to_apply.copy()

    # Define categorical and numerical columns
    metric_cols = df_fit.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df_fit.select_dtypes(exclude=['number']).columns.tolist()

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
        # Ensure the input method is valid
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
