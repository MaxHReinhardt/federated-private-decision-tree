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
    bin_edges = [-np.inf] + list(quartiles) + [np.inf]

    # Ensure distinct bin edges
    for i in range(1, len(bin_edges) - 1):
        if bin_edges[i] == bin_edges[i + 1]:
            # If two edges are the same, move the upper bin edge slightly to avoid overlap
            bin_edges[i + 1] += 1e-9

    return pd.cut(series, bins=bin_edges, labels=['Q1', 'Q2', 'Q3', 'Q4'])


def split_dataset_in_n_equally_sized_client_datasets(df, n, seed=1):
    # Shuffle the df
    shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

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


