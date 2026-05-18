from utils.split_data import split_train_test_data
from src.target_feature_encoding import compute_town_street_avg
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

def objective(trial, df: pd.DataFrame, target_variable: str, tscv: TimeSeriesSplit, model_type: str = "lightgbm"):
    """
    Optuna objective function to perform time-series cross-validation and hyperparameter tuning for LightGBM or XGBoost regression models.

    This function defines the search space for model hyperparameters based on the `model_type`, 
    splits the data using `TimeSeriesSplit` to respect temporal ordering, applies target encoding 
    (`compute_town_street_avg`), trains the model with early stopping, and returns the mean RMSE 
    across all folds.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object used to suggest hyperparameter values.
    df : pandas.DataFrame
        The full dataset containing features and the target variable.
    target_variable : str
        The name of the target column in `df`.
    tscv : sklearn.model_selection.TimeSeriesSplit
        The TimeSeriesSplit object used to generate train/validation indices.
    model_type : str, default="lightgbm"
        The type of model to tune. Supported values are "lightgbm" or "xgboost".

    Returns
    -------
    float
        The mean Root Mean Squared Error (RMSE) across all cross-validation folds. 
        This value is minimized by the Optuna study.
    """
    if model_type == "lightgbm":
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "n_estimators": 10000,
            "num_leaves": trial.suggest_int("num_leaves", 16, 512),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 300),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }

    else: 
        params = {
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
            'n_estimators': 10000,
            'max_depth': trial.suggest_categorical('max_depth', [3,5,7,9,11,13,15,17]),
            'random_state': trial.suggest_categorical('random_state', [2020]),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        }

    rmses = []

    y_train, X_train, _, _ = split_train_test_data(df=df, target_variable=target_variable)

    for _, (train_idx, val_idx) in enumerate(tscv.split(X_train)):

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
        X_tr_fe, mapping, global_mean = compute_town_street_avg(
                                            X_tr, y_tr, training=True
                                        )

        X_val_fe, _, _ = compute_town_street_avg(
            X_val,
            mapping=mapping,
            global_mean=global_mean,
            training=False
        )

        if model_type == "lightgbm":
            model = lgb.LGBMRegressor(
                **params
            )

            model.fit(
                X_tr_fe, y_tr,
                eval_set=[(X_val_fe, y_val)],
                callbacks=[
                    lgb.early_stopping(100)
                ]
            )

        else:
            model = xgb.XGBRegressor(
                **params
            )

            model.fit(
                X_tr_fe, y_tr,
                eval_set=[(X_val_fe, y_val)],
                early_stopping_rounds=100,
                enable_categorical=True,
                tree_method="hist"
            )

        preds = model.predict(X_val_fe)
        rmse = root_mean_squared_error(y_val, preds)
        rmses.append(rmse)

    print("\n" + "="*50)
    print(f"Average CV RMSE: {np.mean(rmses):.2f} (+/- {np.std(rmses):.2f})")
    print("="*50)

    return np.mean(rmses)