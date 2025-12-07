# ---------------------LIBRARIES --------------------- #
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
        verbose=1, # with verbose=1, we can see the progress of the search
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

    # --- Extract scores for R2 ---
    mean_train_r2 = results['mean_train_r2']
    mean_val_r2   = results['mean_test_r2']

    # --- Extract scores for MAE ---
    # Atenção: ainda vêm como NEG-MAE, por isso aplicamos -
    mean_train_mae = -results['mean_train_mae']
    mean_val_mae   = -results['mean_test_mae']

    # Extract parameters
    parameters = results['params']

    # Print each candidate with R2 + MAE + gap
    for r2_t, r2_v, mae_t, mae_v, params in zip(mean_train_r2, mean_val_r2,
                                                mean_train_mae, mean_val_mae, parameters):
        print(
            f"Train R2={r2_t:.3f} | Val R2={r2_v:.3f} | "
            f"Train MAE={mae_t:.1f} | Val MAE={mae_v:.1f} | "
            f"Gap R2={r2_t - r2_v:.3f} | Params={params}"
        )

    # --- Best models based on refit metric (MAE) ---
    best_idx = randomized_model.best_index_

    best_train_r2 = results['mean_train_r2'][best_idx]
    best_val_r2   = results['mean_test_r2'][best_idx]

    best_train_mae = -results['mean_train_mae'][best_idx]
    best_val_mae   = -results['mean_test_mae'][best_idx]

    best_params = results['params'][best_idx]

    print("\n=== BEST MODEL (based on MAE) ===")
    print("Best train R2:", best_train_r2)
    print("Best validation R2:", best_val_r2)
    print("R2 Gap:", (best_train_r2 - best_val_r2) / best_train_r2 * 100, "%")
    print("Best train MAE:", best_train_mae)
    print("Best validation MAE:", best_val_mae)
    print("MAE Gap:", (best_val_mae - best_train_mae) / best_train_mae * 100, "%")
    print("Best parameters:", best_params)

