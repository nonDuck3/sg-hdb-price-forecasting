import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

def calculate_model_metrics(y_test: pd.DataFrame, y_preds: pd.DataFrame):
    """
    Calculate regression model evaluation metrics.
    Args:
        y_test (pd.DataFrame): True target values.
        y_preds (pd.DataFrame): Predicted target values.
    Returns:
        tuple: A tuple containing:
            - rmse (float): Root Mean Squared Error.
            - mae (float): Mean Absolute Error.
            - r2 (float): R-squared score.
    """
    rmse = root_mean_squared_error(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)
    r2 = r2_score(y_test, y_preds)

    return rmse, mae, r2

def print_metrics(
    rmse: list[float] | float,
    mae: list[float] | float,
    r2: list[float] | float,
    model_name: str = "elastic_net",
    is_cv: bool = False,
):
    """
    Print regression model evaluation metrics with optional cross-validation formatting.

    Args:
        rmse (list[float] | float): Root Mean Squared Error values. If a list, represents 
            performance across CV folds; if a float, represents a single test set value.
        mae (list[float] | float): Mean Absolute Error values. Format matches `rmse`.
        r2 (list[float] | float): R-squared (coefficient of determination) values. 
            Format matches `rmse`.
        model_name (str): Identifier for the model being evaluated. Defaults to "elastic_net".
        is_cv (bool): When True, aggregates metrics by computing mean ± standard deviation 
            across CV folds. When False, displays single test set values. Defaults to False.

    Returns:
        tuple[float, float, float] | None: If `is_cv` is True, returns a tuple of 
            (avg_rmse, avg_mae, avg_r2). Returns None otherwise.
    """
    sep = "-" * 30
    if is_cv:
        avg_rmse = np.mean(rmse)
        avg_mae = np.mean(mae)
        avg_r2 = np.mean(r2)
        print(sep)
        print(f"{model_name} average performance (CV folds)")
        print(sep)
        print(f"  RMSE: {avg_rmse:.3f} ± {np.std(rmse):.3f}")
        print(f"  MAE : {avg_mae:.3f} ± {np.std(mae):.3f}")
        print(f"  R²  : {avg_r2:.3f} ± {np.std(r2):.3f}")
        print(sep)
        return avg_rmse, avg_mae, avg_r2
    else:
        print(sep)
        print(f"{model_name} final performance on test set")
        print(sep)
        print(f"  {'RMSE':<10}: {rmse:>10.3f}")
        print(f"  {'MAE':<10}: {mae:>10.3f}")
        print(f"  {'R²':<10}: {r2:>10.3f}")
        print(sep)