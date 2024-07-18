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


