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
        verbose=3, # with verbose=3, we can see the progress of the search in more detail
        return_train_score=True, # return_train_score=True to get training scores as well and understand overfitting
        random_state=40111
    )

    # Fit the model
    randomized_model.fit(X_fit, y_fit)
    
    return randomized_model


# --------------------------------------- EVALUATE MODEL WITH RANDOMIZED SEARCH -------------------------------------- #

# Function to evaluate the model
def evaluate_model(randomized_model):
    """ Evaluate the model fitted with RandomizedSearchCV and print results.
    Parameters:
        randomized_model: Fitted RandomizedSearchCV model
    Prints:
        Train and Validation R2 and MAE for each candidate
        Best model based on refit metric (MAE)
    """

    # Get results dictionary
    results = randomized_model.cv_results_

    # Extract scores for R2 
    train_r2 = results['mean_train_r2']
    val_r2   = results['mean_test_r2']

    # Extract scores for MAE 
    train_mae = -results['mean_train_mae']
    val_mae   = -results['mean_test_mae']

    # Extract parameters
    parameters = results['params']

    # Print each candidate with R2 + MAE + gap
    for r2_t, r2_v, mae_t, mae_v, params in zip(train_r2, val_r2,
                                                train_mae, val_mae, parameters):
        print(
            f"Train R2={r2_t:.3f} | Val R2={r2_v:.3f} | "
            f"Train MAE={mae_t:.1f} | Val MAE={mae_v:.1f} | "
            f"Gap MAE={(abs(mae_v - mae_t)) / mae_t:.3f} | Params={params}"
        )

    # Best models based on refit metric (MAE)
    best_idx = randomized_model.best_index_

    best_train_r2 = results['mean_train_r2'][best_idx]
    best_val_r2   = results['mean_test_r2'][best_idx]

    best_train_mae = -results['mean_train_mae'][best_idx]
    best_val_mae   = -results['mean_test_mae'][best_idx]
    best_params = results['params'][best_idx]

    print("\n=== BEST MODEL (based on MAE) ===")
    print("Best train R2:", best_train_r2)
    print("Best validation R2:", best_val_r2)
    print("Gap R2:", (best_train_r2 - best_val_r2) / best_train_r2 * 100, "%")
    print("Best train MAE:", best_train_mae)
    print("Best validation MAE:", best_val_mae)
    print("Gap MAE:", (best_val_mae - best_train_mae) / best_train_mae * 100, "%")
    print("Best parameters:", best_params)


# --------------------------------------- GET RESULTS DATAFRAME -------------------------------------- #
# Function to get results dataframe from multiple models
def get_results_dataframe(models):

    results_df = pd.DataFrame()

    for model in models:
        # Choose the best score index based on refit metric (MAE)
        best_model = model.best_index_

        # Get results dictionary
        results = best_model.cv_results_

        # Extract scores for MAE 
        train_mae = -results['mean_train_mae']
        val_mae   = -results['mean_test_mae']
        test_mae  = -results['mean_test_mae'] 

        # Time to fit
        time_fit = results['mean_fit_time']

        # Create a DataFrame for the results of the current model
        model_results_df = pd.DataFrame({
            'Model': [type(model.estimator).__name__] * len(train_mae),
            'Time to Fit (s)': time_fit,
            'Train MAE': train_mae,
            'Validation MAE': val_mae,
            'Test MAE': test_mae
        })

        # Append to the overall results DataFrame
        results_df = pd.concat([results_df, model_results_df], ignore_index=True)

    return results_df


# --------------------------------------- SAVE BEST RESULT -------------------------------------- #
# Function to save the best model's predictions on the test set
def save_best_result(results_df, test):

    # Find the best model based on Validation MAE
    best_model= results_df.loc[results_df['Validation MAE'].idxmin()]

    best_model_estimator = best_model.best_estimator_
    predicts = best_model_estimator.predict(test.values)

    # Go one level up from the notebooks folder to reach the repo root
    selected_dir = "../results/kaggle_submissions/"
    os.makedirs(selected_dir, exist_ok=True)

    # Extract carID
    car_ids = test.index.values

    # Create DataFrame with best model predictions
    best_model_df = pd.DataFrame({
        'carID': car_ids,
        'price': predicts
    })

    # Save predictions to CSV
    best_model_df.to_csv(f"{selected_dir}/predictions_{best_model['Model']}.csv", index=False)



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

