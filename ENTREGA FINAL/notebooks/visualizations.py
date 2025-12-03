# ----------------- LIBRARIES ----------------- #
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
import pandas as pd
import numpy as np

# ----------------- BOXPLOTS ----------------- #

# Function to create boxplots for numeric columns with consistent formatting
def create_boxplots(df, numeric_cols, n_cols=2, figsize=(20, 8)):
    """
    Creates boxplots for numeric columns.
    
    Parameters:
        df (DataFrame): The dataset
        numeric_cols (list): List of numeric columns to plot.
        n_cols (int): Number of subplot columns (default=2)
        figsize (tuple): Base figure size; height is scaled dynamically
    """

    n_features = len(numeric_cols) # Total number of numeric features
    n_rows = ceil(n_features / n_cols) # Calculate number of rows needed using ceiling to round up

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
            x=df[col], # Data for the boxplot
            ax=ax, # Axis to plot on
            color='skyblue', # Box color
            medianprops={"color": "darkblue", "linewidth": 2}, # Used to style the median line for better visibility
        )

        ax.set_title(col, fontsize=12, fontweight="bold") # Bold title for each subplot

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Boxplots of Numeric Features", fontsize=18, fontweight="bold") # Overall title
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
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

# ----------------- OUTLIERS SUMMARY ----------------- #

def outlier_summary(df_to_apply, metric_cols):
    """Generates a summary table of outliers for each numeric column using IQR method."""

    summary = [] # List to hold summary data

    for col in metric_cols:  # Iterate over each numeric column
        Q1 = df_to_apply[col].quantile(0.25)
        Q3 = df_to_apply[col].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Boolean mask for outliers
        outliers = (df_to_apply[col] < lower_bound) | (df_to_apply[col] > upper_bound)

        total_outliers = outliers.sum()
        pct_outliers = 100 * total_outliers / len(df_to_apply)

        summary.append({
            "Column": col,
            "Total Outliers": total_outliers,
            "Percentage (%)": round(pct_outliers, 2)
        })

    return pd.DataFrame(summary)

# ----------------- VISUALISE MISSING VALUES ----------------- #

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

