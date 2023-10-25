import pandas as pd


def form(df_data: pd.DataFrame, a_id_col: str, b_id_col: str, gamma: float, state_col: str, historical: bool = True) \
        -> pd.DataFrame:
    """
    The method is from paper <<Predictive analysis and modelling football results using machine learning
        approach for English Premier League>> which evaluates a team's performances in individual matches
    :param df_data: input data
    :param a_id_col: the name of the column containing ids team a
    :param b_id_col: the name of the column containing ids team b
    :param gamma: the stealing fraction, 0 < gamma < 1
    :param state_col: 0 means that A losses, 1 means a draw, and 2 means that A wins
    :param historical: if True, return the actual form scores for both teams which plays in the same game.
        If False, only return the final form score for each team.
    :return: the result of the form score
    """
    a_ids = df_data[a_id_col]
    b_ids = df_data[b_id_col]
    ids = set(a_ids) | set(b_ids)

    # init
    s = {_id: 1.0 for _id in ids}

    # update
    results = []
    for i, row in df_data[[a_id_col, b_id_col, state_col]].iterrows():
        a_id = row[a_id_col]
        b_id = row[b_id_col]
        state = row[state_col]

        results.append([s[a_id], s[b_id]])
        if state == 2:
            s[a_id], s[b_id] = s[a_id] + gamma * s[b_id], s[b_id] - gamma * s[b_id]
        elif state == 0:
            s[b_id], s[a_id] = s[b_id] + gamma * s[b_id], s[a_id] - gamma * s[a_id]
        elif state == 1:
            s[a_id], s[b_id] = \
                s[a_id] - gamma * (s[a_id] - s[b_id]), \
                s[b_id] - gamma * (s[b_id] - s[a_id])
        else:
            raise Exception("state value must be 0, 1, or 2")

    if historical:
        return pd.DataFrame(results, columns=['a_score', 'b_score'], index=df_data.index)
    else:
        return pd.DataFrame(s, index=['score']).transpose()
