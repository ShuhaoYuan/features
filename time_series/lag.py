from typing import List, Optional, Union

import pandas as pd


def lag(df_data: pd.DataFrame, cols: List[str], k: int, historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the average of data
    :param df_data: origin data
    :param cols: the columns for average
    :param k: the number of last k
    :param historical: if True, stats all the rows. if False, only output the one row.
    :return: the result of average
    """
    df = df_data[cols].shift(k)
    if historical:
        return df
    else:
        return df.tail(1).reset_index(drop=True)


def group_lag(df_data: pd.DataFrame, k: int, group_by: List[str], cols: Optional[List[str]] = None,
              historical: bool = True, resort_index: Optional[Union[pd.Series, pd.Index]] = None) -> pd.DataFrame:
    """
    This function is to calculate the average of data
    :param df_data: origin data
    :param cols: the columns for average
    :param k: the number of last k
    :param historical:
    :param group_by:
    :param resort_index:
    :return: the result of average
    """
    group_df = df_data.groupby(group_by)
    if cols is not None:
        group_df = df_data.groupby(group_by)[cols]
    if historical:
        df_result = group_df.shift(k)
        if resort_index is not None:
            df_result = df_result.reset_index(0, drop=True).loc[resort_index, :]
        return df_result
    else:
        return group_df.apply(lambda x: x.iloc[- k - 1, :] if k < x.shape[0] else None)
