import pandas as pd
from datetime import date

def clean_preprocess_data(csv_file: str, file_name: str = "cleaned_hdb_resale_price"):
    """
    Load, clean, and preprocess HDB resale price data from a CSV file, 
    then save the result as a Parquet file.

    This function performs the following transformations:
    1. **Loading**: Reads the input CSV file into a DataFrame.
    2. **Deduplication**: Removes fully duplicate rows.
    3. **Renaming**: Standardizes column names (e.g., 'flat_type' -> 'no_of_rooms_structure', 
       'flat_model' -> 'building_type', 'lease_commence_date' -> 'lease_commence_year').
    4. **Lease Parsing**: Splits the 'remaining_lease' string (e.g., "99 years 6 months") into 
       separate integer columns for years and months, then calculates a precise decimal 
       'remaining_lease_exact_years' value.
    5. **Date Extraction**: Parses the 'month' column to extract 'rego_year' and 'rego_month'.
    6. **Feature Engineering**:
       - Calculates 'lease_age' based on the current year and 'lease_commence_year'.
       - Creates 'street_freq' as a count of units per street (proxy for density).
       - Creates 'town_street' by combining town and street names for location grouping.
    7. **Type Conversion**: Ensures 'resale_price', lease components, and date parts are integers.
    8. **Sorting**: Orders the dataset chronologically by the 'month' column.
    9. **Cleanup**: Drops original 'month', 'remaining_lease', and 'lease_commence_year' columns 
       after extraction.
    10. **Persistence**: Saves the cleaned DataFrame as a Parquet file in the `./data/processed/` 
        directory.

    Parameters
    ----------
    csv_file : str
        Path to the input CSV file containing raw HDB resale data.
    file_name : str, optional
        The base name for the output Parquet file (without extension). 
        Defaults to "cleaned_hdb_resale_price".

    Returns
    -------
    pd.DataFrame
        The cleaned, transformed, and enriched DataFrame ready for modeling.
    """

    df = pd.read_csv(csv_file)

    # drop full row record duplicates
    df = df.drop_duplicates()

    # rename columns
    column_to_rename_list = ["flat_type", "flat_model", "lease_commence_date"]
    new_col_name_list = ["no_of_rooms_structure", "building_type", "lease_commence_year"]

    mapping = dict(zip(column_to_rename_list, new_col_name_list))

    df = df.rename(columns=mapping)

    # convert remaining_lease values to individual columns of years and months
    df[['remaining_lease_years', 'remaining_lease_months']] = df['remaining_lease'].str.split(' years', expand=True)

    # remove text
    df['remaining_lease_months'] = (
        df['remaining_lease_months']
        .str.replace('months', '', regex=False)
        .str.replace('month', '', regex=False)
        .str.strip() 
    )

    # convert string to int for years and months
    df[['remaining_lease_years', 'remaining_lease_months']] = (
        df[['remaining_lease_years', 'remaining_lease_months']]
        .replace('', 0)
        .astype(int)
    )

    # create another column with the accurate number of years and months rounded to 2 dp
    df['remaining_lease_exact_years'] = round((df['remaining_lease_months'] / 12.0) + df['remaining_lease_years'], 2)

    # convert resale price to int
    df['resale_price'] = df['resale_price'].astype(int)

    # convert column to datetime and then to year
    df["lease_commence_year"] = pd.to_datetime(df["lease_commence_year"], format='%Y').dt.year

    # create a new column to get the lease age
    df["lease_age"] = date.today().year - df['lease_commence_year']

    # get the counts of hdb units in each street to use as a potential proxy to gauge how dense the area is
    df["street_freq"] = df["street_name"].map(df["street_name"].value_counts())

    # split registration year into separate columns for year and month
    df[['rego_year', 'rego_month']] = df['month'].str.split('-', expand=True)
    df[['rego_year', 'rego_month']] = df[['rego_year', 'rego_month']].astype(int)

    # sort data in chronological order
    df = df.sort_values('month').reset_index(drop=True)

    # creates another column with the town name and street name combined
    df["town_street"] = df["town"] + "_" + df["street_name"]

    # drop the original remaining_lease and month columns
    df = df.drop(columns=['month', 'remaining_lease', 
                          'lease_commence_year', 'remaining_lease_months', 
                          'remaining_lease_years', 'block', 'street_name'])

    # save the cleaned df as a csv 
    print("Writing cleaned data to parquet file.")
    df.to_parquet(f'./data/processed/{file_name}.parquet', index=False, engine='pyarrow')

    return df