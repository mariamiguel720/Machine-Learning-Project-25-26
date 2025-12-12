# ---------------------LIBRARIES --------------------- #
from turtle import pd
from sklearn.model_selection import RandomizedSearchCV


# --------------------------------------- APPLY RANDOMIZED SEARCH CV -------------------------------------- #

# Function to apply RandomizedSearchCV
def apply_randomized_search_cv(model, param_grid, iterations, scoring, refit, pred_split, X_fit, y_fit):

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



# --------------------------------------- EVALUATE MODEL -------------------------------------- #

# Function to evaluate the model
def evaluate_model(randomized_model):
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
        # Get results dictionary
        results = model.cv_results_

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