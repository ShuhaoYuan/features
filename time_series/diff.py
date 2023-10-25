from typing import List
import pandas as pd

from .sum import last_k_sum, group_last_k_sum
from .avg import last_k_avg, group_last_k_avg


def last_k_diff_sum(df_data: pd.DataFrame, a_cols: List[str], b_cols: List[str], k: int, historical: bool = True) \
        -> pd.DataFrame:
    """
    This function is to calculate the difference of the sum of the last k records within column a and within column b
    :param df_data: input data
    :param a_cols: the columns for the sum calculation
    :param b_cols: the columns for the sum calculation
    :param k: the number of last k records
    :param historical: If True, return the difference of the sum of the last k records for the actual records
        in column a and column b. If False, only return the difference of the sum of the last k records of all
        the records in column a and column b
    :return: the result of difference
    """
    return last_k_sum(df_data=df_data, cols=a_cols, k=k, historical=historical) -\
        last_k_sum(df_data=df_data, cols=b_cols, k=k, historical=historical)


def group_last_k_diff_sum(df_data: pd.DataFrame, a_cols: List[str], b_cols: List[str], k: int, group_by: List[str],
                          historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the difference of the group sum of the last k records within column a and within
        column b
    :param df_data: input data
    :param a_cols: the columns for the sum calculation
    :param b_cols: the columns for the sum calculation
    :param k: the number of last k records
    :param group_by: the columns for dividing the data into groups
    :param historical: If True, return the difference of the group sum of the last k records for the actual records
        in column a and column b. If False, only return the difference of the group sum of the last k records of all
        the records in column a and column b
    :return: the result of difference
    """
    return group_last_k_sum(df_data=df_data, cols=a_cols, k=k, group_by=group_by,
                            historical=historical) - group_last_k_sum(df_data=df_data, cols=b_cols, k=k,
                                                                      group_by=group_by, historical=historical)


def last_k_diff_avg(df_data: pd.DataFrame, a_cols: List[str], b_cols: List[str], k: int,
                    historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the difference of the average the last k records within column a and within column b
    :param df_data: input data
    :param a_cols: the columns for the average calculation
    :param b_cols: the columns for the average calculation
    :param k: the number of last k records
    :param historical: If True, return the difference of the average of the last k records for the actual records
        in column a and column b. If False, only return the difference of the average of the last k records of all
        the records in column a and column b
    :return: the result of difference
    """
    return last_k_avg(df_data=df_data, cols=a_cols, k=k, historical=historical) - last_k_avg(df_data=df_data,
                                                                                             cols=b_cols, k=k,
                                                                                             historical=historical)


def group_last_k_diff_avg(df_data: pd.DataFrame, a_cols: List[str], b_cols: List[str], k: int, group_by: List[str],
                          historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the difference of the group average of the last k records within column a and
        within column b
    :param df_data: input data
    :param a_cols: the columns for the average calculation
    :param b_cols: the columns for the average calculation
    :param k: the number of last k records
    :param group_by: the columns for dividing the data into groups
    :param historical: If True, return the difference of the group average of the last k records for the actual
        records in column a and column b. If False, only return the difference of the group average of the last
        k records of all the records in column a and column b
    :return: the result of difference
    """
    return group_last_k_avg(df_data=df_data, cols=a_cols, k=k, group_by=group_by,
                            historical=historical) - group_last_k_avg(df_data=df_data, cols=b_cols, k=k,
                                                                      group_by=group_by, historical=historical)
