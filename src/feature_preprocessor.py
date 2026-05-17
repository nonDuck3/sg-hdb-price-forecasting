import pandas as pd
import yaml

def preprocess_features(input_df: pd.DataFrame, config_file: str = './configs/encoding_config.yaml'):
    """
    Preprocesses input features based on configuration defined in a YAML file.

    This function applies two main transformations:
    1. Ordinal encoding for specified columns using category ordering defined in the config.
    2. Conversion of specified categorical columns to pandas 'category' dtype.

    The preprocessing rules are loaded from a YAML configuration file, which should
    define ordinal feature mappings and categorical feature lists.

    Parameters
    ----------
    input_df : pd.DataFrame
        Input dataframe containing raw features to be preprocessed.

    config_file : str, default './configs/common.yaml'
        Path to the YAML configuration file containing feature processing rules.
        Expected structure:
        {
            "features": {
                "ordinal": {
                    "column_name": ["low", "medium", "high"],
                    ...
                },
                "categorical": ["col1", "col2", ...]
            }
        }

    Returns
    -------
    pd.DataFrame
        A copy of the input dataframe with:
        - Ordinal features encoded as integers
        - Categorical columns converted to 'category' dtype
    """
    df = input_df.copy()

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    ordinal_config = config['features']['ordinal']
    for col, categories in ordinal_config.items():
        mapping = {val: i for i, val in enumerate(categories)}
        df[col] = df[col].map(mapping)
    
    categorical_cols = config['features']['categorical']

    for col in categorical_cols:
        df[col] = df[col].astype("category")

    return df
