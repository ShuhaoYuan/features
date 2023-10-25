from typing import List, Optional

import pandas as pd

from .sum import group_sum_index


def last_k_count(df_data: pd.DataFrame, cols: List[str], k: int, historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the count of data
    :param df_data: input data
    :param cols: the columns for the count calculation
    :param k: the number of last k records
    :param historical: If True, return the count for unique values within the last k records for each actual record.
        If False, only return the count for unique values of the last k records of the data
    :return: the result of the count calculation
    """
    max_cat_num = max(df_data[cols].nunique().to_list())
    if max_cat_num > 20:
        raise Exception("category number should not be more than 20")

    if historical:
        return pd.get_dummies(df_data[cols].astype("category")).shift(1).rolling(k).sum()
    else:
        return pd.get_dummies(df_data[cols].tail(k).astype("category")).sum(min_count=k).to_frame().transpose()


def group_last_k_count(df_data: pd.DataFrame, cols: List[str], k: int, group_by: List[str],
                     historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the count of data within groups
    :param df_data: input data
    :param cols: the columns for the count calculation
    :param k: the number of last k records
    :param group_by: the columns for dividing the data into groups
    :param historical: If True, return the count for unique values within the last k records for each actual
        record within each group. If False, only return the count for unique values of the last k records of
        the data within each group
    :return: the result of the count calculation within groups
    """
    max_cat_num = max(df_data[cols].nunique().to_list())
    if max_cat_num > 20:
        raise Exception("category number should not be more than 20")

    if historical:
        gb_cols = group_by + cols
        df = df_data.groupby(gb_cols).size().to_frame('count').drop('count', 1).reset_index(level=gb_cols)
        df_join = df[group_by].join(pd.get_dummies(df[cols].astype("category")))
        df_join.index.name = 'num'
        df_result = df_join.set_index(group_by, append=True).groupby(level=group_by).shift(1).\
            groupby(level=group_by).rolling(k).sum().reset_index(group_by)
        return df_result
    else:
        gb_cols = group_by + cols
        return df_data.groupby(group_by).tail(k).groupby(gb_cols).size().to_frame('count').reset_index(level=gb_cols)


def count(df_data: pd.DataFrame, cols: List[str], historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the count of data
    :param df_data: input data
    :param cols: the columns for the count calculation
    :param historical: If True, return the count for unique values of previous records for each actual record.
        If False, only return the count for unique values of all the data
    :return: the result of the count calculation
    """
    max_cat_num = max(df_data[cols].nunique().to_list())
    if max_cat_num > 20:
        raise Exception("category number should not be more than 20")

    if historical:
        return pd.get_dummies(df_data[cols].astype("category")).expanding().sum().shift(1)
    else:
        return pd.get_dummies(df_data[cols].astype("category")).sum().to_frame().transpose()


def group_count(df_data: pd.DataFrame, cols: List[str], group_by: List[str],
              historical: bool = True) -> pd.DataFrame:
    """
    This function is to calculate the count of data within groups
    :param df_data: input data
    :param cols: the columns for the count calculation
    :param group_by: the columns for dividing the data into groups
    :param historical: If True, return the count for unique values of previous records for each actual record
        within groups. If False, only return the count for unique values of all the data within groups.
    :return: the result of the count calculation within groups
    """
    max_cat_num = max(df_data[cols].nunique().to_list())
    if max_cat_num > 20:
        raise Exception("category number should not be more than 20")

    if historical:
        gb_cols = group_by + cols
        df = df_data.groupby(gb_cols).size().to_frame('count').drop('count', 1).reset_index(level=gb_cols)
        df_join = df[group_by].join(pd.get_dummies(df[cols].astype("category")))
        df_join.index.name = 'num'
        df_result = df_join.set_index(group_by, append=True).groupby(level=group_by).shift(1).\
            groupby(level=group_by).expanding().sum().reset_index(group_by)
        return df_result
    else:
        gb_cols = group_by + cols
        return df_data.groupby(gb_cols).size().to_frame('count').reset_index(level=gb_cols)


def group_category_count_index(df_data: pd.DataFrame, category_index: str, group_by: List[str],
              index: str, interval: float, category_columns: Optional[List[str]]=None) -> pd.DataFrame:
    """
    Provide group expanding window cumulated count by fixed index step.
    :param df_data: input data
    :param category_index: category field name.
    :param group_by: the columns for dividing the data into groups
    :param index: the index for segment.
    :param interval: the interval per segment of index.
    :param category_columns: all category names.
    :return: the result of the count calculation within groups
    """
    assert df_data[category_index].isna().sum() == 0, 'The nan data exists'
    df_c = pd.get_dummies(df_data[category_index], columns=category_columns)
    df = pd.concat([df_data[group_by+[index]], df_c], axis=1)
    df = group_sum_index(df, cols=df_c.columns.to_list(), group_by=group_by, index=index, interval=interval)
    return df

