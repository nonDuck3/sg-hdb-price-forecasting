from src.feature_preprocessor import preprocess_features
import pandas as pd

def split_train_test_data(df: pd.DataFrame, target_variable: str = "resale_price"):
    """
    Splits the input DataFrame into training and testing sets based on registration year,
    after preprocessing the features.

    The data is split chronologically:
    - Training set: Rows where 'rego_year' <= 2024
    - Testing set: Rows where 'rego_year' >= 2025

    The 'rego_year' and target variable columns are removed from the feature sets (X).

    Args:
        df (pd.DataFrame): The raw input DataFrame containing features and the target variable.
        target_variable (str, optional): The name of the column representing the target variable. Defaults to 'resale_price'.

    Returns:
        tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
            - y_train: Target values for the training set.
            - X_train: Feature DataFrame for the training set (excluding target and 'rego_year').
            - y_test: Target values for the testing set.
            - X_test: Feature DataFrame for the testing set (excluding target and 'rego_year').
    """
    cleaned_df = preprocess_features(input_df=df)

    train_df = cleaned_df[cleaned_df['rego_year'] <= 2024].copy()
    test_df = cleaned_df[cleaned_df['rego_year'] >= 2025].copy()
    
    print(f"Total Train Rows: {len(train_df)}, Total Test Rows: {len(test_df)}")
    
    y_train = train_df[target_variable]
    X_train = train_df.drop(columns=[target_variable, "rego_year"])

    y_test = test_df[target_variable]
    X_test = test_df.drop(columns=[target_variable, "rego_year"])

    return y_train, X_train, y_test, X_test