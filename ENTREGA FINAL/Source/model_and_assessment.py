# ---------------------LIBRARIES --------------------- #
from modulefinder import test
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import pandas as pd
import copy


# --------------------------------------- APPLY RANDOMIZED SEARCH CV -------------------------------------- #

# Function to apply RandomizedSearchCV
def apply_randomized_search_cv(model, param_grid, iterations, scoring, refit, pred_split, X_fit, y_fit):
    """ Apply RandomizedSearchCV to a given model with specified parameters.
    Parameters:
        model: ML model to be optimized
        param_grid: Dictionary with parameters to search
        iterations: Number of parameter settings that are sampled
        scoring: Scoring metrics to evaluate
        refit: Metric to refit the model
        pred_split: Cross-validation splitting strategy
        X_fit: Features for fitting
        y_fit: Target for fitting
    Returns:
        Fitted RandomizedSearchCV model
    """
    
    # Create RandomizedSearchCV
    randomized_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=iterations,
        scoring=scoring, 
        refit=refit,
        cv=pred_split,
        return_train_score=True, # return_train_score=True to get training scores as well and understand overfitting
        random_state=42
    )

    # Fit the model
    randomized_model.fit(X_fit, y_fit)
    
    return randomized_model


# --------------------------------------- EVALUATE MODEL WITH RANDOMIZED SEARCH -------------------------------------- #

# Function to evaluate model fitted with RandomizedSearchCV
def evaluate_model(randomized_model):
    """ Evaluate the model fitted with RandomizedSearchCV and return results DataFrame.
    Parameters:
        randomized_model: Fitted RandomizedSearchCV model
    Returns:
        DataFrame with Train and Validation R2 and MAE for each candidate, Gaps, and Best model based 
        on refit metric (MAE)
        Best parameters of the best model
    """
    # Get results dictionary
    results = randomized_model.cv_results_

    # Base DataFrame
    df = pd.DataFrame({
        "Train_R2": results["mean_train_r2"],
        "Val_R2": results["mean_test_r2"],
        "Train_MAE": -results["mean_train_mae"],
        "Val_MAE": -results["mean_test_mae"],
    })

    # Gaps
    df["Gap_R2_%"] = (
        (df["Train_R2"] - df["Val_R2"]) / df["Train_R2"] * 100
    )

    df["Gap_MAE_%"] = (
        abs(df["Val_MAE"] - df["Train_MAE"]) / df["Train_MAE"] * 100
    )

    # Identify best model
    best_idx = randomized_model.best_index_
    df["Best_Model"] = ""
    df.loc[best_idx, "Best_Model"] = "⬅ BEST MODEL"

    # Append best model row at the end
    df = pd.concat([
        df.drop(index=[best_idx]),
        df.loc[[best_idx]],
    ]).reset_index(drop=True)

    # Round for readability
    df = df.round(3)

    return df, randomized_model.best_params_


# --------------------------------------- GET RESULTS DATAFRAME -------------------------------------- #

# Function to get results dataframe from multiple models
def final_models_comparison(models):
    """ Create a comparison DataFrame for multiple models fitted with RandomizedSearchCV.
    Parameters:
        models: List of tuples (model_name, fitted_randomized_model)
    Returns:
        DataFrame with Train and Validation R2 and MAE for each model,
        Gaps, and Fit Time
    """
    # Initialize list to store rows
    rows = []
    # Iterate over models
    for name, rs in models:
        # Get best index and results
        best_idx = rs.best_index_
        results = rs.cv_results_

        # Calculate gaps
        gap_mae = abs(
            -results["mean_test_mae"][best_idx] + results["mean_train_mae"][best_idx]
        ) / -results["mean_train_mae"][best_idx] * 100
        gap_r2 = (
            results["mean_train_r2"][best_idx] - results["mean_test_r2"][best_idx]
        ) / results["mean_train_r2"][best_idx] * 100

        # Append row
        rows.append({
            "Algorithm": name,
            "Train MAE": -results["mean_train_mae"][best_idx],
            "Val MAE": -results["mean_test_mae"][best_idx],
            "Gap MAE (%)": gap_mae,
            "Train R2": results["mean_train_r2"][best_idx],
            "Val R2": results["mean_test_r2"][best_idx],
            "Gap R2 (%)": gap_r2,
            "Fit Time (s)": results["mean_fit_time"][best_idx],
        })

    # Create DataFrame
    df = pd.DataFrame(rows).round(3)

    return df


# --------------------------------------- SAVE BEST RESULT -------------------------------------- #

# Function to save predictions for Kaggle submission
def save_Kaggle(best_model, test, model_name):
    """ Save predictions of the best model on the test set in a CSV file for Kaggle submission.
    Parameters:
        best_model: Trained ML model
        test: Test features DataFrame
        model_name: Name of the model for file naming
    """
    # Make predictions
    predicts = best_model.predict(test.values)

    # Create directory if it doesn't exist
    selected_dir = "../results/kaggle_submissions/"
    os.makedirs(selected_dir, exist_ok=True)

    # Get car IDs
    car_ids = test.index.values
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        "carID": car_ids,
        "price": predicts
    })
    # Save to CSV
    submission_df.to_csv(
        f"{selected_dir}/predictions_{model_name}.csv",
        index=False
    )



# --------------------------------------- COMPARE FEATURE SETS -------------------------------------- #

# Function to fit and evaluate model
def fit_evaluate(X_train, y_train, X_val, y_val, model):
    """ Fit model and evaluate performance on training and validation sets. 
    Paramters:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        model: ML model to fit and evaluate
    Returns:
        Dictionary with Train MAE, Val MAE, Gap MAE (%), Train R2, Val R2
    """
    # Fit the model
    model.fit(X_train, y_train)
    # Make predictions
    train_preds = model.predict(X_train)
    val_preds   = model.predict(X_val)
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, train_preds)
    val_mae   = mean_absolute_error(y_val, val_preds)
    gap_mae   = abs(val_mae - train_mae) / train_mae * 100
    train_r2  = model.score(X_train, y_train)
    val_r2    = model.score(X_val, y_val)

    return {
        "Train_MAE": train_mae,
        "Val_MAE": val_mae,
        "Gap_MAE_%": gap_mae,
        "Train_R2": train_r2,
        "Val_R2": val_r2
    }

# Function to compare feature sets
def compare_feature_sets(fs_dict, model, X_train, y_train, X_val, y_val):
    """ Compare different feature sets using a given model and training/validation data.
    Parameters:
        fs_dict: Dictionary where keys are feature set names and values are lists of features
        model: ML model to fit and evaluate
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
    Returns:
        DataFrame with evaluation metrics for each feature set
    """

    # Store results
    results = []
    # Iterate over feature sets
    for key, values in fs_dict.items():
        # Subset training and validation data to current feature set
        X_train_fs = X_train[values]
        X_val_fs   = X_val[values]
        # Fit and evaluate model using current feature set and the function defined above
        metrics = fit_evaluate(
            X_train_fs, y_train, X_val_fs, y_val, copy.deepcopy(model) # copy to assure a fresh model each time
        )
        # Append number of features and metrics to results
        results.append({
            "Num_Features": len(values),
            **metrics
        })

    return pd.DataFrame(results, index = [key for key in fs_dict.keys()]).sort_values("Val_MAE")

# Function to compare data preprocessing methods in model results
def compare_model_dp(dp_dict, model):
    """ Compare different feature sets using a given model and training/validation data.
    Parameters:
        dp_dict: Dictionary where keys are feature set names and values are lists of features
        model: ML model to fit and evaluate
        X_train: Training df
        y_train: Training target
        X_val: Validation df
        y_val: Validation target
    Returns:
        DataFrame with evaluation metrics for each feature set
    """

    # Store results
    results = []
    # Iterate over different data preprocessed dataframes
    for key, (X_train, y_train, X_val, y_val) in dp_dict.items():
        metrics = fit_evaluate(X_train, y_train, X_val, y_val, copy.deepcopy(model)) # copy to assure a fresh model each time

        results.append({
            "Data_Preprocessing": key,
            **metrics
        })

    df = pd.DataFrame(results).set_index("Data_Preprocessing").sort_values("Val_MAE")
    return df