# ------------------------------------------------------ LIBRARIES -------------------------------------------- #
from turtle import pd
import pandas as pd
import numpy as np


def create_features(df_fit, df_to_apply, current_year = 2020, threshold=3):
    """ Create new features based on existing ones in the dataframe.
    Parameters:
        df_fit (pd.DataFrame): DataFrame used to fit the feature engineering.
        df_to_apply (pd.DataFrame): DataFrame to which the new features will be added.
        current_year (int): The current year to calculate car age.
        threshold (int): Threshold in years to determine if a car is recent.
    Returns:
        pd.DataFrame: DataFrame with new features added.
    """

    df_to_apply = df_to_apply.copy()

    #---------------------- without using mean, median or mode ------------------------------------

    # Iterate over both dataframes to create features. 
    # We need to create them also in df_fit to compute group statistics in the next step.
    for df in [df_fit, df_to_apply]: 

        # Create Features: Car Age and Recent Car Indicator
        df["car_age"] = current_year - df["year"]
        df["is_recent_car"] = (df["car_age"] <= threshold).astype('Int64')

        # Create Features: Category Binning for Mileage
        bins = [0, 10_000, 50_000, 100_000, 150_000, float('inf')]
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        df['mileage_category'] = pd.cut(df['mileage'], bins=bins, labels=labels, include_lowest=True)

        # Create Features: Binary flags for eco-friendly cars and automatic transmission
        df["is_hybrid_or_electric"] = df["fuelType"].isin(["Hybrid", "Electric"]).astype('Int64')
        df["is_automatic"] = df["transmission"].isin(["Automatic", "Semi-Auto"]).astype('Int64')
        df["fuel_efficiency_score"] = df["mpg"] / df["engineSize"].replace(0, np.nan)

        # Create Features: Economic indicators based on tax and engine size
        df["tax_to_engine_ratio"] = df["tax"] / df["engineSize"].replace(0, np.nan)
        df["tax_efficiency"] = df["mpg"] / df["tax"].replace(0, np.nan)
        # Create Features: Ownership history
        df["is_first_owner"] = (df["previousOwners"] == 0).round().astype('Int64')

     #---------------------- using mean, median or mode ------------------------------------

    # Calculate group statistics using only X_train
    brand_median_mileage_map = df_fit.groupby('Brand')['mileage'].median().to_dict()
    brand_avg_engineSize_map = df_fit.groupby('Brand')['engineSize'].mean().to_dict()
    model_median_tax_map     = df_fit.groupby('model')['tax'].median().to_dict()
    fueltype_avg_mpg_map     = df_fit.groupby('fuelType')['mpg'].mean().to_dict()
    brand_avg_age_map        = df_fit.groupby('Brand')['car_age'].mean().to_dict()

     # Map precomputed values from the training set
    df_to_apply['brand_median_mileage'] = df_to_apply['Brand'].map(brand_median_mileage_map)
    df_to_apply['brand_avg_engineSize'] = df_to_apply['Brand'].map(brand_avg_engineSize_map)
    df_to_apply['model_median_tax']     = df_to_apply['model'].map(model_median_tax_map)
    df_to_apply['fueltype_avg_mpg']     = df_to_apply['fuelType'].map(fueltype_avg_mpg_map)
    df_to_apply['brand_avg_age']        = df_to_apply['Brand'].map(brand_avg_age_map)

    return df_to_apply