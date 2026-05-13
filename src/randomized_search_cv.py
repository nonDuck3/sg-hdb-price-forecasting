from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import loguniform, uniform
from sklearn.pipeline import Pipeline

def execute_randomized_search(tscv: TimeSeriesSplit, pipeline_model: Pipeline):
    """
    Configures a RandomizedSearchCV instance for time series hyperparameter tuning.

    This function sets up a randomized search over a specified parameter distribution
    using time series cross-validation to prevent data leakage. It is designed for
    optimizing models within a `Pipeline` structure, specifically targeting parameters
    like regularization strength (`alpha`), mixing ratio (`l1_ratio`), and convergence
    settings.

    Args:
        tscv (TimeSeriesSplit): The time series cross-validator instance to use for
            splitting data. This ensures that training data always precedes test data
            chronologically.
        pipeline_model (Pipeline): The scikit-learn Pipeline containing the estimator
            to be tuned.

    Returns:
        RandomizedSearchCV: A configured `RandomizedSearchCV` object ready to be
            fitted on data.
    """
    
    param_dist = {
        "alpha": loguniform(1e-5, 1e2),
        "l1_ratio": uniform(0, 1),
        "max_iter": [5000, 10000],
        "tol": [1e-4, 1e-3, 1e-2]
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline_model,
        param_distributions=param_dist,
        n_iter=50,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    return random_search