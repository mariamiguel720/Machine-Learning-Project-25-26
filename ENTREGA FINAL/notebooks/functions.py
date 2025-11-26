# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


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


# ----------------- SCALING ----------------- #

# Function to scale features using different scaling methods
def scaling_features(train_set, val_set, method):
    """ Scales the features of the train and validation sets according to the specified method.
    Args:
        train_set (pd.DataFrame): The training data to be scaled.
        val_set (pd.DataFrame): The validation data to be scaled.
        method (str): The scaling method to use. Options are 'minmax' - between 0 and 1, 'minmax2' - between -1 and 1, 
        'standard', and 'robust'.
    Returns:
        scaled_X_train (np.ndarray): The scaled training data.
        scaled_X_val (np.ndarray): The scaled validation data.
    """

    if method == 'minmax':
        #scale your data using MinMaxScaler[0,1]
        min_max = MinMaxScaler().fit(train_set)
        # Transform your train data by applying the scale obtained in the previous command
        scaled_X_train = min_max.transform(train_set)
        # Transform your validation data by applying the scale obtained in the first command
        scaled_X_val = min_max.transform(val_set)
    elif method == 'minmax2':
        # Create a MinMaxScaler instance that will range between -1 and 1 and fit to your train data
        min_max = MinMaxScaler(feature_range=(-1, 1)).fit(train_set)
        # Transform your train data by applying the scale obtained in the previous command
        scaled_X_train = min_max.transform(train_set)
        # Transform your validation data by applying the scale obtained in the first command
        scaled_X_val = min_max.transform(val_set)
    elif method == 'standard':
        # Create a StandardScaler instance and fit to your train data
        standard = StandardScaler().fit(train_set)
        # Transform your train data by applying the scale obtained in the previous command
        scaled_X_train = standard.transform(train_set)
        # Transform your validation data by applying the scale obtained in the first command
        scaled_X_val = standard.transform(val_set)
    else: 
        robust = RobustScaler().fit(train_set)
        # Transform your train data by applying the scale obtained in the previous command
        scaled_X_train = robust.transform(train_set)
        # Transform your validation data by applying the scale obtained in the first command
        scaled_X_val = robust.transform(val_set)      
    return scaled_X_train, scaled_X_val

# ----------------- MISSING VALUES ----------------- #

# Function to calculate the percentage of missing values in each column and return a DataFrame
def missing_values_table(data):
    " This function shows the number and percentage of missing values in each column of the dataframe 'data'."
    
    # Number of rows in the dataset
    rows_number = data.shape[0]

    # Number of missing values per column
    missing_counts = data.isnull().sum()

    # Percentage of missing values per column
    missing_percentage = (missing_counts / rows_number) * 100

    # Show in DataFrame format, sorted from highest to lowest
    missing_df = missing_percentage.sort_values(ascending=False).reset_index()
    missing_df.columns = ['Feature', 'Missing_Percent']
    return missing_df

# Function to impute missing values based on specified methods
def imputation(train_set, val_set, test_set, num_method, cat_method, threshold=5.0, neighbors=5):

    """
    Impute missing values in train_set, val_set, test_set datasets.

    Parameters:
        data: original dataframe (used for missing percentages)
        train_set, val_set, test_set: pd.DataFrame
        num_method: method for high missing numerical columns ('KNN', 'Iterative', 'RF')
        cat_method: method for high missing categorical columns ('mode', 'RF')
        threshold: % below which missing values are considered low
        neighbors: n_neighbors for KNN

    Returns:
        train_set_copy, val_set_copy, test_set_copy: the dataframes with imputed values
    """

    # Create copies of the training and validation datasets
    train_set_copy = train_set.copy().reset_index()
    val_set_copy = val_set.copy().reset_index()
    test_set_copy = test_set.copy().reset_index()

    categorical = ['Brand', 'model', 'transmission', 'fuelType', 'hasDamage', 'is_recent_car', 'mileage_category',
            'is_hybrid_or_electric', 'is_automatic', 'paintQuality_category', 'has_damage_or_low_paint', 'is_first_owner']
    numerical = train_set_copy.drop(categorical, axis=1).columns.tolist()

    missing_percentages_df = missing_values_table(train_set_copy)
    low_missing_values = missing_percentages_df[missing_percentages_df['Missing_Percent'] <= threshold]['Feature'].tolist()
    high_missing_values = missing_percentages_df[missing_percentages_df['Missing_Percent'] > threshold]['Feature'].tolist()

    # -------------  LOW MISSING  ------------- #
    
    # Numerical Variables - Median Imputation
    num_low = [n for n in low_missing_values if n in numerical]
    if num_low:
        # calculate Median from training set
        median_value = train_set_copy[num_low].median()

        # fill missing values in train, val, and test sets with train median
        for df in [train_set_copy, val_set_copy, test_set_copy]:
            df[num_low].fillna(median_value, inplace=True)

    # Categorical Variables - Mode Imputation
    cat_low = [c for c in low_missing_values if c in categorical]
    if cat_low:
        # calculate Mode from training set
        mode_value = train_set_copy[cat_low].mode().iloc[0]
        # fill missing values in train, val, and test sets with train Mode
        for df in [train_set_copy, val_set_copy, test_set_copy]:
            df[cat_low].fillna(mode_value, inplace=True)
    
    # -------------  HIGH MISSING  ------------- #
    num_high = [n for n in high_missing_values if n in numerical]
    cat_high = [c for c in high_missing_values if c in categorical]

    # Numerical Variables
    if num_high:
        # KNN Imputation
        if num_method == 'KNN':
            # Fit the KNNImputer on the training set
            knn_imputer = KNNImputer(n_neighbors=neighbors)
            knn_imputer.fit(train_set_copy[num_high])

            # Transform training, validation, and test sets
            for df in [train_set_copy, val_set_copy, test_set_copy]:
                df[num_high] = pd.DataFrame(knn_imputer.transform(df[num_high]),
                                            columns=num_high,
                                            index=df.index)
                   
        # MICE Imputation
        elif num_method == 'Iterative':
            # Fit the IterativeImputer on the training set
            iterative_imputer = IterativeImputer(random_state=40111)
            iterative_imputer.fit(train_set_copy[num_high])

            # Transform training, validation, and test sets
            for df in [train_set_copy, val_set_copy, test_set_copy]:
                df[num_high] = pd.DataFrame(iterative_imputer.transform(df[num_high]),
                                            columns=num_high,
                                            index=df.index)
        # Random Forest Imputation for Numerical Columns
        elif num_method == 'RF':
            for col in num_high:
                not_missing = train_set_copy[train_set_copy[col].notna()]
                missing = train_set_copy[train_set_copy[col].isna()]
                if not not_missing.empty:
                    rf = RandomForestRegressor(n_estimators=200, random_state=40111, n_jobs=-1)
                    # Fit RandomForestRegressor with training data without missing values
                    rf.fit(not_missing.drop(columns=[col]), not_missing[col])
                    
                    # Fill train missing
                    if not missing.empty:
                        train_set_copy.loc[missing.index, col] = rf.predict(missing.drop(columns=[col]))
                    
                    # Fill val missing
                    mask = val_set_copy[col].isna()
                    if mask.sum() > 0:
                        val_set_copy.loc[mask, col] = rf.predict(val_set_copy.loc[mask].drop(columns=[col]))
                    
                    # Fill test missing
                    mask = test_set_copy[col].isna()
                    if mask.sum() > 0:
                        test_set_copy.loc[mask, col] = rf.predict(test_set_copy.loc[mask].drop(columns=[col]))

    if cat_high:
        # Mode Imputation
        if cat_method == 'Mode':
            for col in cat_high:
                mode_value = train_set_copy[col].mode().iloc[0]
                # fill missing values in train, val, and test sets with train mode
                for df in [train_set_copy, val_set_copy, test_set_copy]:
                    df[col].fillna(mode_value, inplace=True)
        # Random Forest Imputation for Categorical Columns
        elif cat_method == 'RF':
            for col in cat_high:
                not_missing = train_set_copy[train_set_copy[col].notna()]
                missing = train_set_copy[train_set_copy[col].isna()]

                if not not_missing.empty:
                    # Define features (all columns except target)
                    feature_cols = [c for c in train_set_copy.columns if c != col]

                    # Train RandomForestClassifier
                    clf = RandomForestClassifier(n_estimators=200, random_state=40111, n_jobs=-1)
                    clf.fit(not_missing[feature_cols], not_missing[col])

                    # Fill train missing
                    if not missing.empty:
                        train_set_copy.loc[missing.index, col] = clf.predict(missing[feature_cols])

                    # Fill val missing
                    mask = val_set_copy[col].isna()
                    if mask.sum() > 0:
                        val_set_copy.loc[mask, col] = clf.predict(val_set_copy.loc[mask, feature_cols])

                    # Fill test missing
                    mask = test_set_copy[col].isna()
                    if mask.sum() > 0:
                        test_set_copy.loc[mask, col] = clf.predict(test_set_copy.loc[mask, feature_cols])
    
    return train_set_copy, val_set_copy, test_set_copy