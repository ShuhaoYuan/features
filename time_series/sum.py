import math
from typing import List

import pandas as pd
from isodate import DT_BAS_ORD_COMPLETE
from tqdm import tqdm

from .lag import group_lag

tqdm.pandas()


def last_k_sum(df_data: pd.DataFrame, cols: List[str], k: int, historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the sum of data
    :param df_data: input data
    :param cols: the columns for the sum calculation
    :param k: the number of last k records
    :param historical: If True, return the sum of the last k records for each actual record.
        If False, only return the sum of the last k records of the data
    :return: the result of the sum calculation
    """
    if historical:
        return df_data[cols].rolling(k).sum().shift(1)
    else:
        return df_data[cols].tail(k).sum(min_count=k).to_frame().transpose()


def group_last_k_sum(df_data: pd.DataFrame, cols: List[str], k: int, group_by: List[str],
                     historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the sum of group data
    :param df_data: input data
    :param cols: the columns for the sum calculation
    :param k: the number of last k records
    :param group_by: the columns for dividing the data into groups
    :param historical: If True, return the sum of the last k records for each actual record within each group.
        If False, only return the sum of the last k records of the data within each group
    :return: the result of the sum calculation
    """
    if historical:
        df_result: pd.DataFrame = df_data \
            .groupby(group_by)[cols].rolling(k).sum()
        df_result = group_lag(df_result, k=1, group_by=group_by, historical=True, resort_index=df_data.index)
        return df_result
    else:
        return df_data.groupby(group_by)[cols].apply(lambda x: x.tail(k).sum(min_count=k))


def cum_sum(df_data: pd.DataFrame, cols: List[str], historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the cumulated sum of data
    :param df_data: input data
    :param cols: the columns for the cumulated sum calculation
    :param historical: If True, return the cumulated sum of the previous records for each actual record.
        If False, only return the sum of the data
    :return: the result of sum
    """
    if historical:
        return df_data[cols].expanding().sum().shift(1)
    else:
        return df_data[cols].sum(axis=0)


def group_sum(df_data: pd.DataFrame, cols: List[str], group_by: List[str],
              historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the cumulated sum of data within groups
    :param df_data: input data
    :param cols: the columns for the cumulated sum calculation
    :param group_by: the columns for dividing the data into groups
    :param historical: If True, return the cumulated sum of the previous records for each actual record
        within each group. If False, only return the sum of the last k records of the data within each group
    :return: the result of sum calculation within each group
    """
    if historical:
        df_result: pd.DataFrame = df_data \
            .groupby(group_by)[cols].expanding().sum()
        df_result = group_lag(df_result, k=1, group_by=group_by, historical=True, resort_index=df_data.index)
        return df_result
    else:
        return df_data.groupby(group_by)[cols].sum()


def group_sum_index(df_data: pd.DataFrame, cols: List[str], group_by: List[str],
                    index: str, interval: float) -> pd.DataFrame:
    """
    Provide group expanding window cumulated sum by fixed index step.
    :param df_data: input data
    :param cols: the columns for the cumulated sum calculation
    :param group_by: the columns for dividing the data into groups
    :param index: the index for segment.
    :param interval: the interval per segment of index.
    :return: the result of cumulated sum calculation within each group
    """

    def _segment(_df: pd.DataFrame) -> pd.DataFrame:
        maxv = _df[index].max()
        c = math.ceil(maxv / interval)
        idx = 0
        ds_result = []
        for i in range(c):
            current_line = interval * (i + 1)
            while idx < _df.shape[0]:
                # if last
                if idx == _df.shape[0] - 1:
                    ds = _df.iloc[idx][cols]
                    ds[index] = current_line
                    ds_result.append(ds)
                    break

                if _df.iloc[idx][index] <= current_line < _df.iloc[idx + 1][index]:
                    ds = _df.iloc[idx][cols]
                    ds[index] = current_line
                    ds_result.append(ds)
                    break
                elif _df.iloc[idx][index] > current_line:
                    if idx == 0:
                        ds = _df.iloc[idx][cols].copy()
                        ds[:] = 0
                    else:
                        ds = _df.iloc[idx - 1][cols]
                    ds[index] = current_line
                    ds_result.append(ds)
                    break
                elif _df.iloc[idx + 1][index] <= current_line:
                    idx = idx + 1
                    continue
        try:
            df_res = pd.concat(ds_result, axis=1).transpose().reset_index(drop=True)
        except ValueError as e:
            print(_df)
        return df_res

    if df_data.shape[0] == 0:
        df_group = df_data[cols]
    else:
        df_group = df_data.groupby(group_by)[cols].cumsum()
    df = pd.concat([df_data[group_by + [index]], df_group], axis=1)
    df = df.sort_values(by=index)
    df = df.groupby(group_by)[cols + [index]].progress_apply(_segment)
    return df
