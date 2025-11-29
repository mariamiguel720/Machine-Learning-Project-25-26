# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from sklearn.impute import KNNImputer #,IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer


# ----------------- BOXPLOTS ----------------- #

# Function to create boxplots for numeric columns with consistent formatting
def create_boxplots(df, numeric_cols, n_cols=2, figsize=(20, 12)):
    """
    Creates boxplots for numeric columns with consistent formatting.
    
    Parameters:
        df (DataFrame): The dataset
        numeric_cols (list): List of numeric columns to plot.
        n_cols (int): Number of subplot columns (default=2)
        figsize (tuple): Base figure size; height is scaled dynamically
    """

    n_features = len(numeric_cols)
    n_rows = ceil(n_features / n_cols)

    # Dynamic height scaling
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(figsize[0], figsize[1])
    )
    axes = axes.flatten()

    sns.set_theme(style="whitegrid", palette="pastel")

    for i, col in enumerate(numeric_cols):
        ax = axes[i]

        sns.boxplot(
            x=df[col],
            ax=ax,
            color='skyblue',
            medianprops={"color": "darkblue", "linewidth": 2},
            boxprops={"alpha": 0.7}
        )

        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Boxplots of Numeric Features", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# ----------------- HEATMAPS ----------------- #

def create_heatmap(df, method, numeric_cols, figsize=(10, 8)):
    """
    Creates a heatmap for the correlation matrix of numeric columns.
    
    Parameters:
        df (DataFrame): The dataset
        method (str): Correlation method to use (e.g., "pearson", "spearman", "kendall").
        numeric_cols (list): List of numeric columns to include in the correlation matrix.
    """
    corr = df[numeric_cols].corr(method=method).round(2)
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))  
    # Visualize correlation matrix
    fig = plt.figure(figsize=figsize)

    sns.heatmap(
    corr,
    mask=mask,                # hide upper triangle
    annot=True,               # show values
    cmap="coolwarm",          # divergent color map
    center=0,                 # center colormap in 0
    linewidths=0.5,           # lines between cells to help visualization
    vmin=-1, vmax=1,          # fix scale
    square=True               # make cells square-shaped
    )

    plt.title(f"Correlation Matrix ({method})", fontsize=14, pad=15)
    plt.tight_layout() # improve layout by reducing overlaps
    plt.show()


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
def scaling_features(df_fit, df_to_apply, method):
    """ Scales the features of the train and validation sets according to the specified method.
    Args:
        df_fit (pd.DataFrame): The dataframe to fit the scaler.
        df_to_apply (pd.DataFrame): The dataframe to apply the scaler.
        method (str): The scaling method to use. Options are 'minmax' - between 0 and 1, 'minmax2' - between -1 and 1, 
        'standard', and 'robust'.
    Returns:
        scaled_df_to_apply (np.ndarray): The scaled dataframe to which the scaler is applied.
    """

    if method == 'minmax':
        #scale your data using MinMaxScaler[0,1]
        min_max = MinMaxScaler().fit(df_fit)
        # Transform the data from df_to_apply by applying the scale obtained in the previous command
        scaled_df_to_apply = min_max.transform(df_to_apply)
    elif method == 'minmax2':
        # Create a MinMaxScaler instance that will range between -1 and 1 and fit to your train data
        min_max = MinMaxScaler(feature_range=(-1, 1)).fit(df_fit)
        # Transform your the data from df_to_apply by applying the scale obtained in the previous command
        scaled_df_to_apply = min_max.transform(df_to_apply)
    elif method == 'standard':
        # Create a StandardScaler instance and fit to your train data
        standard = StandardScaler().fit(df_fit)
        # Transform your the data from df_to_apply by applying the scale obtained in the previous command
        scaled_df_to_apply = standard.transform(df_to_apply)
    else: 
        robust = RobustScaler().fit(df_fit)
        # Transform your the data from df_to_apply by applying the scale obtained in the previous command
        scaled_df_to_apply = robust.transform(df_to_apply)
    return scaled_df_to_apply

# ----------------- MISSING VALUES ----------------- #

# Function to calculate the percentage of missing values in each column and return a DataFrame
def missing_values_table(df):
    " This function shows the number and percentage of missing values in each column of the dataframe 'data'."
    
    # Number of rows in the dataset
    rows_number = df.shape[0]
    # Number of missing values per column
    missing_counts = df.isnull().sum()

    # Percentage of missing values per column
    missing_percentage = (missing_counts / rows_number) * 100

    # Show in DataFrame format, sorted from highest to lowest
    missing_df = missing_percentage.sort_values(ascending=False).reset_index()
    missing_df.columns = ['Feature', 'Missing_Percent']
    return missing_df


# Function to impute missing values based on specified methods
def imputation(df_fit, df_to_apply, impute_method, scaling_method, threshold=5.0, neighbors=5):

    """
    Impute missing values in train_set, val_set, test_set datasets.

    Parameters:
        df_fit: original dataframe (used for missing percentages) and to fit imputation models
        df_to_apply: dataframe to apply the imputation
        impute_method: method choosen for treatment of high missing values. When None all missing values are treated with median/mode.
        scaling_method: scaling method to use on categorical features before KNN imputation
        threshold: % below which missing values are considered low
        neighbors: n_neighbors for KNN

    Returns:
        df_to_apply: dataframe with imputed missing values
    """

    # Create copies of the training and validation datasets
    df_fit = df_fit.copy()
    df_to_apply = df_to_apply.copy()


    categorical = ['Brand', 'model', 'transmission', 'fuelType', 'hasDamage', 'is_recent_car', 'mileage_category',
            'is_hybrid_or_electric', 'is_automatic', 'paintQuality_category', 'has_damage_or_low_paint', 'is_first_owner']
    numerical = df_fit.drop(categorical, axis=1).columns.tolist()

    missing_percentages_df = missing_values_table(df_fit)
    low_missing_values = missing_percentages_df[missing_percentages_df['Missing_Percent'] <= threshold]['Feature'].tolist()
    high_missing_values = missing_percentages_df[missing_percentages_df['Missing_Percent'] > threshold]['Feature'].tolist()

    
    # -------------  LOW MISSING  ------------- #
    
    for col in low_missing_values or method is None: # Impute using Median/Mode
        if col in numerical:
            median_value = df_fit[col].median()
            # fill missing values in the df to apply with train median
            df_to_apply[col].fillna(median_value, inplace=True)

        elif col in categorical:
            mode_value = df_fit[col].mode().iloc[0]
            # fill missing values in the df to apply with train mode
            df_to_apply[col].fillna(mode_value, inplace=True)


    # -------------  HIGH MISSING  ------------- #

    for col in high_missing_values and method is not None: # Impute using specified method
        if impute_method == "KNN":
            # Scale categorical features before KNN Imputation because KNN is distance-based and only works with numerical data
            scaling_features(df_fit[categorical], method=scaling_method)
            scaling_features(df_to_apply[categorical], method=scaling_method)

            # Fit the KNNImputer on the training set
            knn_imputer = KNNImputer(n_neighbors=neighbors, weights='distance')
            knn_imputer.fit(df_fit[[col]])
            # Transform df to apply 
            df_to_apply[[col]] = pd.DataFrame(knn_imputer.transform(df_to_apply[[col]]),
                                            columns=[col],
                                            index=df_to_apply.index)
        
        elif impute_method == "RF":
        
    return df_to_apply