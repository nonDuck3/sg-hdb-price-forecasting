import pandas as pd
def compute_town_street_avg(X: pd.DataFrame, y: pd.Series = None, mapping: pd.Series = None, global_mean: float = None, training: bool = True):
    """
    Compute and apply town-street level average target encoding.

    In training mode, calculates the mean target value for each unique 
    'town_street' combination and the global mean. In inference mode, 
    applies the provided mapping to encode 'town_street' and fills missing 
    categories with the global mean. The original 'town_street' column is 
    dropped and replaced with 'town_street_avg'.

    Args:
        X (pd.DataFrame): Input DataFrame containing a 'town_street' column.
        y (pd.Series, optional): Target values required only when `training=True`.
        mapping (pd.Series, optional): Pre-computed mapping of town-street to 
            average target. Required when `training=False`.
        global_mean (float, optional): Global average target value. Required 
            when `training=False` for filling missing categories.
        training (bool): If True, computes statistics from `y`. If False, 
            applies existing `mapping` and `global_mean`.

    Returns:
        tuple:
            - X (pd.DataFrame): Transformed DataFrame with 'town_street_avg' 
              column and 'town_street' removed.
            - mapping (pd.Series): The computed town-street averages (if 
              training=True), or the input mapping otherwise.
            - global_mean (float): The computed global mean (if training=True), 
              or the input global_mean otherwise.
    """
    X = X.copy()

    if training:
        temp_df = X.copy()
        temp_df["resale_price"] = y

        mapping = (
            temp_df.groupby("town_street")["resale_price"]
            .mean()
        )

        global_mean = y.mean()

    X["town_street_avg"] = X["town_street"].map(mapping)
    X["town_street_avg"] = X["town_street_avg"].fillna(global_mean)
    X = X.drop(columns=["town_street"])

    return X, mapping, global_mean