
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
