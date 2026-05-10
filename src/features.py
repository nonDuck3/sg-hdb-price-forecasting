from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from lightgbm import LGBMRegressor
import yaml

def build_pipeline(config_file: str = './config.yaml', dynamic_columns: list[str] = None):
    """
    Construct a machine learning pipeline for HDB resale price prediction 
    based on configuration settings.

    This function reads a YAML configuration file to dynamically define 
    preprocessing strategies and model hyperparameters. It builds a 
    `ColumnTransformer` to handle three types of features:
    1. **Ordinal**: Encoded using `OrdinalEncoder` with specific category 
       orders defined in the config. Unknown values are mapped to -1.
    2. **Nominal**: Encoded using `OneHotEncoder` with `drop='first'` to 
       avoid multicollinearity. Unknown categories are ignored.
    3. **Numeric**: Passed through unchanged (or scaled, depending on your pipeline setup).

    Additionally, it supports **dynamic columns** (e.g., target-encoded features like 
    `town_street_avg`) that are not present in the static config but are created 
    programmatically during cross-validation. These columns are appended to the 
    numeric feature list.

    The preprocessed data is then fed into a `LGBMRegressor` with a 
    specific set of hyperparameters tuned for this task.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file containing feature definitions 
        and model parameters. Expected structure:
        ```yaml
        features:
          ordinal:
            column_name: ["cat1", "cat2", ...]
          dummy_coded:
            - column_name1
            - column_name2
          numeric:
            - column_name1
            - column_name2
        ```
    dynamic_columns : list[str], optional
        A list of column names to be added to the numeric feature set. 
        These columns are typically derived dynamically during cross-validation 
        (e.g., time-series target encodings) and do not exist in the static 
        configuration file. Defaults to None.

    Returns
    -------
    Pipeline
        A scikit-learn `Pipeline` object containing:
        - 'preprocessing': The configured `ColumnTransformer` (including dynamic columns).
        - 'model': The initialized `LGBMRegressor`.
    """

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    ordinal_cols = list(config['features']['ordinal'].keys())
    ordinal_category = [config['features']['ordinal'][col] for col in ordinal_cols]
    dummy_coded_cols = config['features']['dummy_coded']
    numeric_cols = config['features']['numeric']

    if dynamic_columns:
        numeric_cols.extend(dynamic_columns)

    ordinal_preprocessor = OrdinalEncoder(
        categories=ordinal_category,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    dummy_coded_preprocessor = OneHotEncoder(
        handle_unknown="ignore",
        drop='first'
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", ordinal_preprocessor, ordinal_cols),
            ("nom", dummy_coded_preprocessor, dummy_coded_cols),
            ('numeric', 'passthrough', numeric_cols)
        ],
        remainder="drop" 
    )

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        ))
    ])

    return pipeline

