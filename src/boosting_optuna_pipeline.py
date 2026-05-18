from utils.artifact_utils import generate_save_feature_plot, save_model_predictions, save_model_object
from utils.split_data import split_train_test_data
from src.target_feature_encoding import compute_town_street_avg
from src.optuna_objective_function import objective
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import optuna
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

def tune_train_evaluate_model(
    df: pd.DataFrame,
    target_variable: str,
    tscv: TimeSeriesSplit,
    model_name: str = "lightgbm"
):
    """
    Tune hyperparameters with Optuna, refit the best model on the full training data, and evaluate it on a held-out test set.

    This utility function performs the following steps:
    1. Creates an Optuna study (minimization) and optimizes the provided `objective` using the full dataset.
    2. Retrieves the best hyperparameters found by Optuna.
    3. Splits the full dataset into training and test sets using `split_train_test_data`.
    4. Applies target encoding (`compute_town_street_avg`) to the training features, then applies the same mapping to the test features.
    5. Instantiates and fits the model with the best parameters on the encoded training data.
    6. Generates predictions for the test set and calculates the final RMSE.
    7. Generates and saves feature importance artifacts (table and plot).
    8. Persists the model, predictions, and evaluation metrics via helper save functions.

    Parameters
    ----------
    df : pandas.DataFrame
        Complete dataset containing both features and the target variable.
    target_variable : str
        The name of the column in `df` representing the target.
    tscv : TimeSeriesSplit
        The TimeSeriesSplit object used for cross-validation within the objective function.
    model_name : str, default="lightgbm"
        Label used when saving feature-importance plots, model artifacts, and for logging.

    Returns
    -------
    dict
        Dictionary containing:
        - "model": fitted estimator instance trained on the encoded training data.
        - "study": optuna.study.Study object containing all trial results.
        - "best_params": dict of best hyperparameters from Optuna.
        - "feature_importance": pandas.DataFrame with columns ["feature", "importance"], sorted descending.
        - "test_predictions": array-like predictions from `model.predict(X_test_fe)`.
        - "rmse": float representing the root mean squared error on the test set.
    """
    y_train, X_train, y_test, X_test = split_train_test_data(df=df, target_variable=target_variable)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, x_train=X_train, y_train=y_train, tscv=tscv, model_type=model_name), 
        n_trials=50 
    )

    best_params = study.best_params
    print("Best parameters: ", best_params)
    print("Number of finished trials: {}".format(len(study.trials)))
    print('Best trial:', study.best_trial.params)

    X_train_fe, mapping, global_mean = compute_town_street_avg(
        X_train,
        y_train,
        training=True
    )

    X_test_fe, _, _ = compute_town_street_avg(
        X_test,
        mapping=mapping,
        global_mean=global_mean,
        training=False
    )

    if model_name == "lightgbm":
        model = lgb.LGBMRegressor(
            **best_params
        )
    else:
        model = xgb.XGBRegressor(
            **best_params
        )

    model.fit(X_train_fe, y_train)
    test_preds = model.predict(X_test_fe)
    rmse_final = root_mean_squared_error(y_train, test_preds)
    print(f"RMSE for {model_name}: ", round(rmse_final, 2))

    feat_imp_df = generate_save_feature_plot(df=X_train_fe, model= model, model_type=model_name)
    save_model_predictions(pred_values=test_preds, y_test=y_test, x_test=X_test_fe, model_name=model_name)
    save_model_object(model=model, best_params=best_params, eval_metric= rmse_final, feature_vals=feat_imp_df["importance"], model_name=model_name)

    return {
        "model": model,
        "study": study,
        "best_params": best_params,
        "feature_importance": feat_imp_df["importance"],
        "test_predictions": test_preds,
        "rmse": rmse_final
    }