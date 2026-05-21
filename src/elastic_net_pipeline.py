from utils.split_data import split_train_test_data
from utils.artifact_utils import generate_save_feature_plot, save_model_predictions, save_model_object
from utils.compute_print_metrics import print_metrics, calculate_model_metrics
from src.target_feature_encoding import compute_town_street_avg
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
import yaml
import warnings

warnings.filterwarnings('ignore')

def build_and_train_pipeline(input_df: pd.DataFrame, target_variable: str, tscv: TimeSeriesSplit, config_file: str = './configs/encoding_config.yaml'):
    """
    Constructs, tunes, and trains a complete ElasticNet regression pipeline with time-series-aware hyperparameter optimization.

    This function orchestrates the full modeling lifecycle:
    1. Loads feature engineering configurations (ordinal, categorical, numeric) from a YAML file.
    2. Preprocesses data by mapping ordinal features, scaling numeric/ordinal columns, and one-hot encoding nominal categories.
    3. Implements a custom grid search over ElasticNet hyperparameters (alpha, l1_ratio) using the provided TimeSeriesSplit validator.
    4. Applies target transformation (log1p) and custom target encoding (town/street averages) within each cross-validation fold to prevent data leakage.
    5. Trains the final model on the full training set using the best hyperparameters found.
    6. Evaluates performance on a held-out test set, generates feature importance visualizations, and persists model artifacts.

    Args:
        input_df (pd.DataFrame): The raw input DataFrame containing features and the target variable.
        target_variable (str): The name of the column to predict.
        tscv (TimeSeriesSplit): A TimeSeriesSplit instance for chronological cross-validation during tuning.
        config_file (str, optional): Path to the YAML file defining feature categories. Defaults to './configs/elastic_net.yaml'.

    Returns:
        dict: A dictionary containing:
            - 'model': The fully trained Pipeline instance.
            - 'best_params': The optimal hyperparameters (alpha, l1_ratio) found during tuning.
            - 'model_coefficients': Absolute values of the learned feature coefficients.
            - 'test_predictions': Predicted values on the test set (inverse-transformed from log scale).
            - 'rmse': The final Root Mean Squared Error on the test set.
            - 'mae': The final Mean Absolute Error on the test set.
            - 'r2': The final R-squared (coefficient of determination) on the test set.
    """
    
    df = input_df.copy()

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    nominal_cols = config['features']['categorical']
    numeric_cols = config['features']['numeric']
    numeric_and_ordinal_cols = list(config['features']['ordinal'].keys()) + numeric_cols

    preprocessor = ColumnTransformer(transformers=[
        ('scaled', StandardScaler(), numeric_and_ordinal_cols),
        ('nom', OneHotEncoder(drop='first', handle_unknown='ignore'), nominal_cols)
    ])

    en_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', 
         ElasticNet(
            max_iter=10000,
            tol=1e-4
        ))
    ])

    best_val = np.inf
    alphas = [0.01, 0.1, 1.0]
    l1_ratios = [0.1, 0.5, 0.9]
    total_combos = len(alphas) * len(l1_ratios)
    combo_count = 0
    y_train, X_train, y_test, X_test = split_train_test_data(df=df, target_variable=target_variable)

    hyperparam_list = []
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            combo_count += 1
            print("="*50)
            print(f"Testing combination {combo_count}/{total_combos} => alpha: {alpha}, l1_ratio: {l1_ratio}")
            rmses = []
            maes = []
            r2s = []
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

                en_pipeline.set_params(
                    model__alpha=alpha,
                    model__l1_ratio=l1_ratio
                )

                y_tr_log = np.log1p(y_tr)
                en_pipeline.fit(X_tr_fe, y_tr_log)
                pred_log = en_pipeline.predict(X_val_fe)

                pred = np.expm1(pred_log)

                rmse, mae, r2 = calculate_model_metrics(y_test=y_val, y_preds=pred)

                rmses.append(rmse)
                maes.append(mae)
                r2s.append(r2)

            avg_rmse, _, _ = print_metrics(rmse=rmses, mae=maes, r2=r2s, is_cv=True)
            if avg_rmse < best_val:
                best_val = avg_rmse
                print(f"Best hyperparameters for alpha: {alpha} and l1_ratio: {l1_ratio}." + "\n")
                hyperparam_list.extend([alpha, l1_ratio])

    best_params = hyperparam_list[-2:]

    full_en_train_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ("model", ElasticNet(
            alpha=best_params[0],
            l1_ratio=best_params[1],
            max_iter=10000,
            tol=1e-4
        ))
    ])

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

    log_y_train = np.log1p(y_train)
    full_en_train_pipeline.fit(X_train_fe, log_y_train)
    test_pred_log = en_pipeline.predict(X_test_fe)
    test_set_pred = np.expm1(test_pred_log)

    final_rmse, final_mae, final_r2 = calculate_model_metrics(y_test=y_test, y_preds=test_set_pred)
    print_metrics(rmse=final_rmse, mae=final_mae, r2=final_r2)

    coef_df = generate_save_feature_plot(df=X_train_fe, model=full_en_train_pipeline)
    save_model_predictions(pred_values=test_set_pred, y_test=y_test, x_test=X_test_fe)
    save_model_object(
        model=full_en_train_pipeline, 
        best_params=best_params, 
        rmse_value=final_rmse,
        mae_value=final_mae,
        r2_value=final_r2,
        feature_vals=coef_df["coefficient_absolute_value"]
    )

    return {
        "model": full_en_train_pipeline,
        "best_params": best_params,
        "model_coefficients": coef_df["coefficient_absolute_value"],
        "test_predictions": test_set_pred,
        "rmse": final_rmse,
        "mae": final_mae,
        "r2": final_r2
    }
