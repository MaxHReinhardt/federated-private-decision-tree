import pandas as pd
import numpy as np


# Main preprocessing function
def preprocessing(df):
    # Drop rows with missing values
    df = df.dropna(axis=0)

    # Drop ID
    df = df.drop(columns=['ID'])

    # Group numerical variables based on quartiles
    for column in ['Age', 'Work_Experience', 'Family_Size']:
        df[column] = categorize_quartiles(df[column])

    return df


# Function to categorize based on quartiles
def categorize_quartiles(series):
    quartiles = series.quantile([0.25, 0.5, 0.75])
    return pd.cut(series, bins=[-np.inf, quartiles[0.25], quartiles[0.5], quartiles[0.75], np.inf],
                  labels=['Q1', 'Q2', 'Q3', 'Q4'])


def split_dataset_in_n_equally_sized_client_datasets(df, n):
    # Shuffle the df
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

    num_rows = len(shuffled_df)
    rows_per_split = num_rows // n

    split_dfs = []

    for i in range(n):
        start_index = i * rows_per_split
        # The last client gets all remaining rows
        if i == n - 1:
            split_dfs.append(shuffled_df.iloc[start_index:])
        else:
            split_dfs.append(shuffled_df.iloc[start_index:start_index + rows_per_split])

    return split_dfs


