from typing import List
import pandas as pd

from .lag import group_lag


def last_k_avg(df_data: pd.DataFrame, cols: List[str], k: int, historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the average of data
    :param df_data: input data
    :param cols: the columns for the average calculation
    :param k: the number of last k records
    :param historical: If True, return the average of the last k records for each actual record.
        If False, only return the average of the last k records of the data
    :return: the result of the average calculation
    """
    if historical:
        return df_data[cols].rolling(k, min_periods=1).mean().shift(1)
    else:
        return df_data[cols].tail(k).mean(axis=0).to_frame().transpose()


def group_last_k_avg(df_data: pd.DataFrame, cols: List[str], k: int, group_by: List[str],
                     historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the average of data within groups
    :param df_data: input data
    :param cols: the columns for average calculation
    :param k: the number of last k records
    :param group_by: the columns for dividing the data into groups
    :param historical: If True, return the average of the last k records for each actual record within each group.
        If False, only return the average of the last k records of the data within each group
    :return: the result of the average calculation within groups
    """
    if historical:
        df_result: pd.DataFrame = df_data \
                                      .groupby(group_by)[cols].rolling(k, min_periods=1).mean()
        df_result = group_lag(df_result, k=1, group_by=group_by, historical=True, resort_index=df_data.index)
        return df_result
    else:
        return df_data.groupby(group_by)[cols].apply(lambda x: x.tail(k).mean(axis=0))


def avg(df_data: pd.DataFrame, cols: List[str], historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the average of data
    :param df_data: input data
    :param cols: the columns for the average calculation
    :param historical: If True, return the average of the previous records for each actual record.
        If False, only return the average of all the data
    :return: the result of the average calculation
    """
    if historical:
        return df_data[cols].expanding().mean().shift(1)
    else:
        return df_data[cols].mean(axis=0)


def group_avg(df_data: pd.DataFrame, cols: List[str], group_by: List[str],
              historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the average of data within groups
    :param df_data: input data
    :param cols: the columns for average calculation
    :param group_by: the columns for dividing the data into groups
    :param historical:If True, return the average of the previous records for each actual record within each group.
        If False, only return the average of all of the data within each group
    :return: the result of the average calculation within groups
    """
    if historical:
        df_result: pd.DataFrame = df_data \
                                      .groupby(group_by)[cols].expanding().mean()
        df_result = group_lag(df_result, k=1, group_by=group_by, historical=True, resort_index=df_data.index)
        return df_result
    else:
        return df_data.groupby(group_by)[cols].mean()


def moment_avg(df_data: pd.DataFrame, cols: List[str], alpha: float, historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the moment average of data using exponential weighted (EW) functions.
    :param df_data: input data
    :param cols: the columns for the moment average calculation
    :param alpha: smoothing factor, 0 < alpha <= 1
    :param historical: If True, return the moment average of the previous records for each actual record.
        If False, only return the moment average of all the data
    :return: the result of the moment average calculation
    """
    if historical:
        return df_data[cols].ewm(alpha=alpha, adjust=False).mean().shift(1)
    else:
        return df_data[cols].ewm(alpha=alpha, adjust=False).mean().tail(1).reset_index(drop=True)


def group_moment_avg(df_data: pd.DataFrame, cols: List[str], alpha: float, group_by: List[str],
                     historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the moment average of data within groups using exponential weighted (EW) functions.
    :param df_data: input data
    :param cols: the columns for the moment average calculation
    :param alpha: smoothing factor, 0 < alpha <= 1
    :param group_by: the columns for dividing the data into groups
    :param historical: If True, return the moment average of the previous records for each actual record.
        If False, only return the moment average of all the data
    :return: the result of the moment average calculation within groups
    """
    if historical:
        df_result = df_data \
                   .groupby(group_by)[cols].ewm(alpha=alpha, adjust=False).mean()
        df_result = group_lag(df_result, k=1, group_by=group_by, historical=True, resort_index=df_data.index)
        return df_result

    else:
        return df_data \
            .groupby(group_by)[cols].ewm(alpha=alpha, adjust=False).mean() \
            .groupby(group_by).tail(1).reset_index(1, drop=True)
