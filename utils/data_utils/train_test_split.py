# utils/data_utils.py

import pandas as pd

def load_train_test_split(filepath="data/final_multivariatetimeseries.csv", test_size=0.2, parse_dates=True, save_files=False):
    """
    Load final multivariate time series, split into train/test sets.

    Args:
        filepath (str): Path to the final time series CSV.
        test_size (float): Fraction to use for test set.
        parse_dates (bool): Whether to parse the index as datetime.
        save_files (bool): Whether to save train/test to disk.

    Returns:
        df_train (DataFrame): Training data.
        df_test (DataFrame): Test data.
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=parse_dates)
    df = df.sort_index()

    n_test = int(len(df) * test_size)
    df_test = df.tail(n_test).copy()
    df_train = df.drop(df_test.index).copy()

    if save_files:
        df_train.to_csv("data/train.csv")
        df_test.to_csv("data/test_final.csv")

    return df_train, df_test
