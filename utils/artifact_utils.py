import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np

def generate_save_feature_plot(df: pd.DataFrame, model: object, model_type: str = "elastic_net"):
    """
    Generate, save, and display a horizontal bar plot of feature importances or coefficient magnitudes.

    This function creates a visualization based on the specified model type.
    - For 'elastic_net', it extracts non-zero coefficients, calculates their absolute values,
      and plots the magnitude of coefficients.
    - For other model types, it uses the model's `feature_importances_` attribute.

    The resulting plot is sorted by importance in descending order, saved to
    `./outputs/plots/{model_type}_{metric}.png` (300 DPI), and displayed.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the feature names as column headers. These are used
        to map coefficients or importances to specific features.
    model : object
        A trained scikit-learn compatible model.
        - If `model_type` is "elastic_net", the model must have a `coef_` attribute.
        - Otherwise, the model must have a `feature_importances_` attribute.
    model_type : str, default="elastic_net"
        The type of model being visualized. Determines the metric used (coefficients
        vs. feature importances) and affects the plot title and output filename.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the 'feature' names and the calculated metric 
        ('coefficient_absolute_value' for elastic_net, 'importance' for others),
        sorted in descending order by that metric.
    """
    if model_type == "elastic_net":
        y_axis_value, title_text = "coefficient_absolute_value", "Magnitude of Coefficients"
        feature_df = pd.DataFrame({
            "feature": df.columns,
            "coefficients": model.coef_
        })

        feature_df = feature_df[feature_df["coefficients"] != 0]
        feature_df["coefficient_absolute_value"] = feature_df["coefficients"].abs()

    else: 
        y_axis_value, title_text = "importance", "Feature Importance"
        feature_df = pd.DataFrame({
            "feature": df.columns,
            "importance": model.feature_importances_
        })

    sorted_df = feature_df.sort_values(y_axis_value, ascending=False)
    sorted_df.plot(
        x="feature", y=y_axis_value, kind="barh", figsize=(15,9), color="#2a9d8f"
    )

    plt.title(f"{title_text} of {model_type} model")
    plt.savefig(f"./outputs/plots/{model_type}_{y_axis_value}.png", bbox_inches="tight", dpi=300)
    plt.show() 

    return sorted_df 

def save_model_object(model, best_params, eval_metric, feature_vals, model_name: str = "elastic_net"):
    """
    Serialize and save a trained model, hyperparameters, metrics, and feature values to disk.

    This function bundles the provided model object along with its best hyperparameters,
    evaluation metrics (RMSE), and feature-specific values (coefficients or importances) 
    into a single dictionary and persists it using `joblib`. The output file is saved to 
    `./outputs/models/{model_name}_artifacts.pkl`.

    Parameters
    ----------
    model : object
        The trained model or pipeline object to be persisted. Must be serializable 
        by `joblib` (e.g., scikit-learn estimators or pipelines).
    best_params : dict
        Mapping of the best hyperparameters found during tuning (e.g., from 
        GridSearchCV or RandomizedSearchCV).
    eval_metric : float or scalar
        The evaluation metric value (e.g., RMSE) to be stored alongside the model.
    feature_vals : array-like or list
        The feature values to persist. For "elastic_net" models, this typically 
        represents non-zero coefficients. For other models, it represents feature 
        importances. The length of this list may differ from the original feature 
        set length (e.g., if only non-zero features are included).
    model_name : str, default="elastic_net"
        Base name used to construct the output filename. Determines the column key 
        name in the saved dictionary:
        - "elastic_net" -> key: "feature_coefficients"
        - Other -> key: "feature_importances"
    """
    feature_col_name = "feature_coefficients" if model_name == "elastic_net" else "feature_importances"
    joblib.dump({
        "model": model,
        "best_params": best_params,
        "rmse": eval_metric,
        feature_col_name: feature_vals
    }, f"./outputs/models/{model_name}_artifacts.pkl")

def save_model_predictions(pred_values, y_test, x_test: pd.DataFrame, model_name: str = "elastic_net"):
    """
    Create a DataFrame of predictions and save it to a CSV file.

    This function constructs a results table containing the sample identifier 
    (derived from the index of `x_test`), the actual target values, predicted 
    values, and error metrics (absolute and signed error). The resulting DataFrame 
    is saved to `./outputs/predictions/{model_name}_predictions.csv`.

    Parameters
    ----------
    pred_values : array-like
        The predicted values generated by the model. Must be aligned with the 
        rows of `x_test` and `y_test`.
    y_test : array-like
        The ground truth (actual) target values. Must be aligned with `x_test` 
        and `pred_values`.
    x_test : pandas.DataFrame
        The test feature DataFrame. Its index is used to generate the unique 
        sample identifier (`id`) in the output file.
    model_name : str, default="elastic_net"
        The base name used to construct the output filename. The file will be 
        saved as `{model_name}_predictions.csv`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the columns:
        - `id`: Sample identifier from `x_test.index`.
        - `actual`: Ground truth values.
        - `prediction`: Model predictions.
        - `error`: Signed difference (`actual - prediction`).
        - `abs_error`: Absolute difference (`|actual - prediction|`).
    """

    prediction_df = pd.DataFrame({
        "id": x_test.index,
        "actual": y_test,
        "prediction": pred_values,
        "error": y_test - pred_values,
        "abs_error": np.abs(y_test - pred_values)
    })

    prediction_df.to_csv(f"./outputs/predictions/{model_name}_predictions.csv", index=False)

    return prediction_df
