import numpy as np
import pandas as pd

from .lag import group_lag


def weight_streak(df_data: pd.DataFrame, team_id: str, k: int, outcome_col: str,
                  historical: bool = True) -> pd.DataFrame:
    """
    The streak method is from paper <<Predictive analysis and modelling football results using machine learning approach
        for English Premier League>> which encapsulates the recent performances of a team.
    :param df_data: input data
    :param team_id: the name of the column containing team ids
    :param k: last k match
    :param outcome_col: 0 means a loss, 1 means a draw, and 2 means a win
    :param historical: if True, return the actual streak scores obtained from the previous matches for the actual team.
        If False, only return the final streak score for each team.
    :return: the result of the streak score
    """
    outcome = df_data[outcome_col].map({0: 0, 1: 1, 2: 3})
    df_data = pd.concat([df_data[team_id], outcome], axis=1)

    def streak(df: pd.Series):
        L = len(df)
        w = np.arange(1, L + 1)
        return 2 * (df.to_numpy() * w).sum() / (3 * L * (L + 1))

    if historical:
        df_result = df_data.groupby(team_id)[[outcome_col]].rolling(k, min_periods=1).apply(streak)
        df_result = group_lag(df_result, k=1, group_by=[team_id], historical=True, resort_index=df_data.index)
        return df_result
    else:
        return df_data.groupby(team_id)[[outcome_col]].apply(lambda x: x.tail(k).apply(streak))
