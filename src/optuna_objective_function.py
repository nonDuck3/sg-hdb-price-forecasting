from src.target_feature_encoding import compute_town_street_avg
from utils.compute_print_metrics import calculate_model_metrics, print_metrics
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

def objective(trial, x_train: pd.DataFrame, y_train: pd.DataFrame, tscv: TimeSeriesSplit, model_type: str = "lightgbm"):
    """
    Optuna objective function to perform time-series cross-validation and hyperparameter tuning for LightGBM or XGBoost regression models.

    This function defines a dynamic search space for model hyperparameters based on the chosen `model_type`. 
    It splits the pre-separated training features and labels using `TimeSeriesSplit` to strictly respect 
    temporal ordering, applies target encoding (`compute_town_street_avg`) within each cross-validation fold 
    to prevent data leakage, trains the models using early stopping, and tracks evaluation metrics.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object used to dynamically suggest hyperparameter values.
    x_train : pandas.DataFrame
        The training feature matrix (without the target variable) used for temporal cross-validation.
    y_train : pandas.DataFrame or pandas.Series
        The target variable (e.g., house resale prices) corresponding to the records in `x_train`.
    tscv : sklearn.model_selection.TimeSeriesSplit
        The TimeSeriesSplit configuration object used to safely generate train/validation index arrays.
    model_type : str, default="lightgbm"
        The type of tree-based framework to optimize. Supported values are "lightgbm" or "xgboost".

    Returns
    -------
    float
        The average Root Mean Squared Error (RMSE) calculated across all evaluated cross-validation folds. 
        This scalar score is returned to Optuna to minimize during the optimization process.
    """
    if model_type == "lightgbm":
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "n_estimators": 5000,
            "num_leaves": trial.suggest_int("num_leaves", 16, 63),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "verbose": -1,
            "n_jobs": -1
        }

    else: 
        params = {
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.008,0.01,0.012,0.014,0.016,0.018, 0.02]),
            'n_estimators': 5000,
            'max_depth': trial.suggest_categorical('max_depth', [3,5,7,9,11,13,15,17]),
            'random_state': trial.suggest_categorical('random_state', [2020]),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
            'n_jobs': -1
        }

    rmses = []
    maes = []
    r2s = []
    for _, (train_idx, val_idx) in enumerate(tscv.split(x_train)):

        X_tr, X_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
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
                    lgb.early_stopping(50, verbose=False)
                ]
            )

        else:
            model = xgb.XGBRegressor(
                **params
            )

            model.fit(
                X_tr_fe, y_tr,
                eval_set=[(X_val_fe, y_val)],
                early_stopping_rounds=50,
                verbose=False,
                enable_categorical=True,
                tree_method="hist"
            )

        preds = model.predict(X_val_fe)
        rmse, mae, r2 = calculate_model_metrics(y_test=y_val, y_preds=preds)
        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)

    avg_rmse, _, _ = print_metrics(rmse=rmses, mae=maes, r2=r2s, model_name=model_type, is_cv=True)
    return avg_rmse