import matplotlib.pyplot as plt
import joblib
import pandas as pd

def generate_save_feat_imp_plot(feature_importance: pd.DataFrame, model_type: str):
    """
    Generate and save a horizontal bar plot of feature importances.
    This function sorts the provided feature-importance table by importance,
    creates a horizontal bar chart, sets an informative title, saves the plot
    to ``./outputs/plots/{model_type}_feature_importance.png`` and then displays
    it.
    Parameters
    ----------
    feature_importance : pandas.DataFrame
        DataFrame containing at least two columns:
        - ``feature``: feature names (strings)
        - ``importance``: numeric importance values
    model_type : str
        Label for the model used to format the plot title and output filename.
    """

    feature_importance.sort_values("importance").plot(
        x="feature", y="importance", kind="barh", figsize=(15,9), color="#2a9d8f"
    )

    plt.title(f"Feature Importance of {model_type} model")
    plt.savefig(f"./outputs/plots/{model_type}_feature_importance.png", bbox_inches="tight", dpi=300)
    plt.show()  

def save_model_object(model, best_params, feature_importance, model_name: str):
    """
    Serialize and save a trained model pipeline and related metadata to disk.
    This function bundles the provided model object along with its best
    hyperparameters and feature importance information into a single dictionary
    and persists it using joblib to `./outputs/models/{model_name}_pipeline.pkl`.
    Args:
        model: The trained model or pipeline object to be persisted (must be
            joblib-serializable).
        best_params: Mapping of the best hyperparameters found during tuning
            (e.g., from GridSearchCV/RandomizedSearchCV).
        feature_importance: Feature importance information associated with the
            model (e.g., array-like, dict, or DataFrame).
        model_name: str
            Base name used to construct the output filename.
    """
    joblib.dump({
        "model": model,
        "best_params": best_params,
        "feature_importance": feature_importance
    }, f"./outputs/models/{model_name}_pipeline.pkl")

def save_model_predictions(pred_values, y_test, x_test: pd.DataFrame, model_name: str):
    """
    Save model predictions and actual values to a CSV file.

    Creates a CSV containing the sample identifier (from ``x_test.index``),
    the actual target values, and the model's predicted values. The file is
    saved to ``./outputs/predictions/{model_name}_predictions.csv``.

    Args:
        pred_values: Array-like of predicted values aligned with ``x_test`` rows.
        y_test: Array-like of actual target values aligned with ``x_test`` rows.
        x_test (pd.DataFrame): Test features DataFrame whose index serves as
            the sample identifier.
        model_name (str): Name used to construct the output filename.
    """

    prediction_df = pd.DataFrame({
        "id": x_test.index,
        "actual": y_test,
        "prediction": pred_values
    })

    prediction_df.to_csv(f"./outputs/predictions/{model_name}_predictions.csv", index=False)
