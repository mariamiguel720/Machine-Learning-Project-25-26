# # ------------------------------------------------------ LIBRARIES -------------------------------------------- #
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns
from math import ceil
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, TargetEncoder, OneHotEncoder, OrdinalEncoder

    
# --------------------------------------------------- CORRECT VALUES -------------------------------------------- #

# ----------------- INVALID VALUES ----------------- #

# Function to correct invalid values in the DataFrame
def correct_metric_features(df):
    """ Corrects invalid values in the DataFrame by replacing negative numeric values with NaN and 
    rounding specific columns to integers.
    Parameters:
        df: DataFrame to correct
    Returns:
        df: Corrected DataFrame
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns # Select numeric columns

    # Replace negative values with NaN
    for col in numeric_cols:
        if (df[col] < 0).any():  # Check for negative values
            df.loc[df[col] < 0, col] = np.nan # Replace negative values with NaN
    
    # Round specific columns to integers
    for col in ['year', 'previousOwners', 'hasDamage']: #Rounds numeric values: values with decimal part >= 0.5 go up, others go down.
        df[col] = df[col].round().astype('Int64') # Use 'Int64' to allow for NaN values
    
    # Fill NaN values in 'hasDamage' with 1 (assuming missing means there is damage)
    df['hasDamage'] = df['hasDamage'].fillna(1)

    return df 

# ----------------- CATEGORY REPLACEMENTS ----------------- #

# Function to replace 'Other' category in 'transmission' column
def replace_category_transmission(df):
    """ Replaces 'Other' category in 'transmission' column with 'unknown'.
    Parameters:
        df: DataFrame to correct
    Returns:
        df: Corrected DataFrame
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Replace 'Other' with 'unknown' in 'transmission' column
    df['transmission'] = df['transmission'].replace('Other', 'unknown')

    return df 


# --------------------------------------------------- CATEGORICAL CORRECTIONS ----------------------------------- #

# Function to calculate similarity between two strings
def similar(a, b):
    """Calculates similarity ratio between two strings using SequenceMatcher.
    Parameters:
        a: First string
        b: Second string
    Returns:
        Similarity ratio (float)
    """
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
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

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
    
    # Apply the mapping to the column
    df[column] = df[column].map(mapping).fillna(df[column])
    
    return df



# --------------------------------------------------- OUTLIERS TREATMENT --------------------------------------------------- #

# Function to treat outliers based on custom rules
def treat_outliers_custom(df_fit, df_to_apply, threshold=2.2):
    """ Treats outliers in df_to_apply based on custom rules defined for each column.
    Parameters:
        df_fit (pd.DataFrame): The dataframe to fit the outlier treatment rules.
        df_to_apply (pd.DataFrame): The dataframe to apply the outlier treatment.
        threshold (float): The multiplier for the IQR to define outlier limits.
    Returns:
        df_to_apply (pd.DataFrame): The dataframe with treated outliers.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_to_apply = df_to_apply.copy()

    # Fixed limits
    df_to_apply['year'] = df_to_apply['year'].clip(lower=1990, upper=2020)
    df_to_apply['engineSize'] = df_to_apply['engineSize'].clip(lower=0.9, upper=5.5)
    df_to_apply['previousOwners'] = df_to_apply['previousOwners'].clip(lower=0, upper=6)

    # IQR based limits
    for col in ['mileage', 'tax', 'mpg']:
        # Calculate IQR based limits from df_fit
        q1 = df_fit[col].quantile(0.25)
        q3 = df_fit[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr

        # Cap the outliers
        df_to_apply[col] = df_to_apply[col].clip(lower=lower, upper=upper)

    return df_to_apply

# --------------------------------------------------- ENCODING --------------------------------------------------- #

# Function to encode categorical features using different Encoding methods
def encoding_features(X_fit, Y_fit, df_to_apply):
    """     Encode categorical features using Target, One-Hot and Ordinal Encoding. 
    Parameters:
        X_fit: DataFrame to fit the encoders (training set)
        Y_fit: Series or DataFrame with target variable for target encoding
        df_to_apply: DataFrame to apply the fitted encoders (training/validation/test set)
    Returns:
        df_to_apply: DataFrame with encoded categorical features
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_to_apply = df_to_apply.copy()

    # Define categorical columns for each encoding method
    target_cols = ['Brand', 'model']
    one_hot_cols = ['transmission', 'fuelType']
    ordinal_cols = ['mileage_category']

    # Identify numeric features
    numeric_features = X_fit.columns.drop(target_cols + one_hot_cols + ordinal_cols)

    # -----------  TARGET ENCODING ----------- #
    Y_fit_continuous = Y_fit.astype(float) # Ensure Y_fit is continuous
    # Call Target Encoder and fit to train data
    # CV is only going to be aplied during training fitting phase
    target_enc = TargetEncoder(cv=5, smooth='auto', random_state=42, target_type='continuous').fit(X_fit[target_cols], Y_fit_continuous)        

    # Transform the data from df_to_apply by applying the encoding obtained in the previous command
    data_encoded = target_enc.transform(df_to_apply[target_cols])

    # Convert to DataFrame
    df_target = pd.DataFrame(
        data_encoded,
        columns=[f'{col}_target' for col in target_cols],
        index=df_to_apply.index
    )

    # -----------  ONE-HOT ENCODING ----------- #
    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore') #sparse_output=False outputs a numpy array, not a sparse matrix
    onehot_fit = ohe.fit(X_fit[one_hot_cols])
    onehot_transformed = onehot_fit.transform(df_to_apply[one_hot_cols])

    one_hot_feat_names = onehot_fit.get_feature_names_out(one_hot_cols)
    one_hot_feat_names = [f"{name}_ohe" for name in one_hot_feat_names]

    df_ohe = pd.DataFrame(onehot_transformed, index=df_to_apply.index, columns=one_hot_feat_names)
    

    # -----------  ORDINAL ENCODING ----------- #
    categories = [['Very Low', 'Low', 'Medium', 'High', 'Very High']]
    enc = OrdinalEncoder(
        categories=categories,
        handle_unknown='use_encoded_value', # Handle unknown categories by encoding them with a specific value
        unknown_value=-1, dtype=int) # Ensure integer type
    enc.fit(X_fit[ordinal_cols])

    ordinal_transformed = enc.transform(df_to_apply[ordinal_cols])
    
    df_ordinal = pd.DataFrame(
        ordinal_transformed,
        index=df_to_apply.index,
        columns=[f'{col}_ordinal' for col in ordinal_cols]
    )

    df_to_apply = pd.concat([df_to_apply[numeric_features], df_target, df_ohe, df_ordinal], axis=1)

    return df_to_apply
    

# --------------------------------------------------- SCALING --------------------------------------------------- #

# Function to scale features using different scaling methods
def scaling_features(df_fit, df_to_apply, method):
    """ Scales the features of the train and validation sets according to the specified method.
    Args:
        df_fit (pd.DataFrame): The dataframe to fit the scaler.
        df_to_apply (pd.DataFrame): The dataframe to apply the scaler.
        method (str): The scaling method to use. Options are 'minmax' - between 0 and 1, 'minmax2' - between -1 and 1, 
        'standard', and 'robust'.
    Returns:
        df_to_apply (np.ndarray): The scaled dataframe to which the scaler is applied.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_to_apply = df_to_apply.copy()

    # Identify one-hot encoded columns and metric columns
    one_hot_cols = [col for col in df_fit.columns if col.endswith('_ohe')]

    metric_cols = [
        col for col in df_fit.columns
        if col not in one_hot_cols and pd.api.types.is_numeric_dtype(df_fit[col])
    ]
    if method == 'minmax':
        # Scale the data using MinMaxScaler[0,1]
        min_max = MinMaxScaler().fit(df_fit[metric_cols])
        # Transform the data from df_to_apply by applying the scale obtained in the previous command
        scaled_array = min_max.transform(df_to_apply[metric_cols])
    elif method == 'minmax2':
        # Create a MinMaxScaler instance that will range between -1 and 1 and fit to the train data
        min_max2 = MinMaxScaler(feature_range=(-1, 1)).fit(df_fit[metric_cols])
        # Transform the the data from df_to_apply by applying the scale obtained in the previous command
        scaled_array = min_max2.transform(df_to_apply[metric_cols])
    elif method == 'standard':
        # Create a StandardScaler instance and fit to the train data
        standard = StandardScaler().fit(df_fit[metric_cols])
        # Transform the the data from df_to_apply by applying the scale obtained in the previous command
        scaled_array = standard.transform(df_to_apply[metric_cols])
    elif method == 'robust': 
        robust = RobustScaler().fit(df_fit[metric_cols])
        # Transform the the data from df_to_apply by applying the scale obtained in the previous command
        scaled_array = robust.transform(df_to_apply[metric_cols])

    # Replace the original metric columns with the scaled values
    df_to_apply[metric_cols] = scaled_array

    return df_to_apply

# --------------------------------------------- MISSING VALUES IMPUTATION ----------------------------------------- #

# Function to impute missing values using Simple or KNN imputation
def impute_missing(df_fit, df_to_apply, method="simple", neighbors=5):
    """
    Imputes missing values in df_to_apply using specified method for metric cols.
    Imputes categorical cols using most frequent value.
    Parameters:
        df_fit: DataFrame to fit the imputation models (training set)
        df_to_apply: DataFrame to apply the imputation (training/validation/test set)
        method: imputation method for metric columns ("simple" for median, "knn" for KNN)
        neighbors: number of neighbors for KNN imputation
    Returns:
        df_to_apply: DataFrame with imputed missing values
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_to_apply = df_to_apply.copy()

    # Define categorical and numerical columns
    metric_cols = df_fit.select_dtypes(include=['number']).columns.tolist()


    # IMPUTATION FOR METRIC COLUMNS: KNN OR MEDIAN
    # Create and fit the imputer based on the selected method
    if method == "knn":
        imp_num = KNNImputer(n_neighbors=neighbors, weights="distance")
    
    elif method == "simple":
        # IMPUTATION FOR CATEGORICAL COLUMNS: MOST FREQUENT VALUE
        # Define categorical columns for simple imputation
        cat_cols = df_fit.select_dtypes(exclude=['number']).columns.tolist()
        # Create and fit the imputer
        imp_cat = SimpleImputer(strategy="most_frequent")
        imp_cat.fit(df_fit[cat_cols])
        # Apply the imputer to df_to_apply
        df_to_apply[cat_cols] = imp_cat.transform(df_to_apply[cat_cols])
        # IMPUTATION FOR METRIC COLUMNS
        imp_num = SimpleImputer(strategy="median")
    
    # Ensure the input method is valid
    else :
        raise ValueError("Invalid method. Choose 'simple' or 'knn'.")
    # Fit and transform the metric columns

    # Fit and transform the metric columns
    imp_num.fit(df_fit[metric_cols])
    df_to_apply[metric_cols] = imp_num.transform(df_to_apply[metric_cols])

    return df_to_apply

# --------------------------------------------- DATA PREPROCESSING ----------------------------------------- #

def data_preprocessing(df_fit, target_fit, df_to_apply, neighbors=5, imputation_method="knn", scaling_method="standard"):
    """ Preprocess the data by treating outliers, imputing missing values, encoding categorical features, and scaling numeric features.
    Parameters:
        df_fit: Training feature set
        target_fit: Training target variable
        df_to_apply: Validation feature set
        treat_outliers: Boolean indicating whether to treat outliers
        neighbors: Number of neighbors for KNN imputation
        imputation_method: Method for imputing missing values ("simple" or "knn")
        scaling_method: Method for scaling features ("minmax", "minmax2", "standard", "robust")
    Returns:
        X_apply: Preprocessed dataframe
    """

    X_fit = df_fit.copy()
    X_apply = df_to_apply.copy()
    y_fit = target_fit.copy()


    # If missing values imputation is simple, scaling and encoding comes after imputation
    # If missing values imputation is KNN, scaling and encoding comes before imputation
    # Impute missing values
    if imputation_method == "simple":
        X_apply = impute_missing(X_fit, X_apply, method="simple")
        X_fit = impute_missing(X_fit, X_fit, method="simple")

        X_apply = encoding_features(X_fit, y_fit, X_apply)
        X_fit = encoding_features(X_fit, y_fit, X_fit)

        X_apply = scaling_features(X_fit, X_apply, method=scaling_method)
        X_fit = scaling_features(X_fit, X_fit, method=scaling_method)

    elif imputation_method == "knn":
        X_apply = encoding_features(X_fit, y_fit, X_apply)
        X_fit = encoding_features(X_fit, y_fit, X_fit)

        X_apply = scaling_features(X_fit, X_apply, method=scaling_method)
        X_fit = scaling_features(X_fit, X_fit, method=scaling_method)

        X_apply = impute_missing(X_fit, X_apply, method="knn", neighbors=neighbors)
        X_fit = impute_missing(X_fit, X_fit, method="knn", neighbors=neighbors)
    
    else:
        raise ValueError("Invalid imputation method. Choose 'simple' or 'knn'.")
    
    return X_apply