import pandas as pd
from datasetsforecast.hierarchical import HierarchicalData


def load_tourism_data():
    """
    Loads the tourism dataset using HierarchicalData.

    Returns:
        tuple: A tuple containing the dataframe, the summing matrix, and the tags.
    """
    Y_df, S_df, tags = HierarchicalData.load("./data", "TourismLarge")
    Y_df["ds"] = pd.to_datetime(Y_df["ds"])
    S_df = S_df.reset_index(names="unique_id")
    return Y_df, S_df, tags


def split_data(Y_df, horizon):
    """
    Splits the data into training and testing sets.

    Args:
        Y_df (pd.DataFrame): The input dataframe.
        horizon (int): The forecast horizon.

    Returns:
        tuple: A tuple containing the training and testing dataframes.
    """
    Y_test_df = Y_df.groupby("unique_id", as_index=False).tail(horizon)
    Y_train_df = Y_df.drop(Y_test_df.index)
    return Y_train_df, Y_test_df
