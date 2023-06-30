"""It contains operations to post-process the data from the comparator."""

import pandas as pd
import ast
from typing import List, Dict, Any, Union


def explode_numeric_columns(df_selection, columns_to_explode):
    """Explode the numerical columns passed."""
    tmp_df_selction = df_selection
    df_selection = tmp_df_selction.copy()
    del tmp_df_selction

    def transform_to_list(x):
        try:
            res = ast.literal_eval(x)
        except ValueError as e:
            print(e)
            print(x)
        return res

    # explode all the lists
    for col_list in columns_to_explode:
        col_short = col_list.replace("_list", "")
        # convert the col_list to a proper python list
        df_selection.loc[:, (col_list)] = df_selection.apply(
            lambda row: transform_to_list(row[col_list]), axis=1)
        df_selection = df_selection.explode(col_list)
        df_selection.reset_index(drop=True, inplace=True)
        # rename col_list to col_short
        df_selection.rename(columns={col_list: col_short}, inplace=True)
        # convert col_short to numeric column
        df_selection[col_short] = pd.to_numeric(df_selection[col_short])
    return df_selection


def explode_column_with_list_of_tuples(
        df_selection,
        columns_to_explode: List[str],
        new_column_names: List[str]):
    """Explode the columns passed which must be list of tuples.

    Note that the columns_to_explode must be contain "_list" in the name.

    and example could be the following coming from corelation and pvalue
    of a test:
    [[0.2, 0.0007], [0.23, 0.0003], [0.223, 0.9]]
    this will explode each element of the list in a new row, and create two
    columns with the name expressed in the new_column_names parameter (e.g.
    columns_to_explode = "spearman_res_list"
    new_column_names = ["corelation", "pvalue"]
    it will produce two new columns:
    spearman_res.corelation and spearman_res.pvalue
    """
    tmp_df_selction = df_selection
    df_selection = tmp_df_selction.copy()
    del tmp_df_selction

    def transform_to_list(x):
        try:
            res = ast.literal_eval(x)
        except ValueError as e:
            print(e)
            print(x)
        return res

    # explode all the lists
    for col_list in columns_to_explode:
        col_short = col_list.replace("_list", "")
        # convert the col_list to type string
        df_selection[col_list] = df_selection[col_list].astype(str)
        # convert the col_list to a proper python list
        df_selection.loc[:, (col_list)] = df_selection.apply(
            lambda row: transform_to_list(row[col_list]), axis=1)
        df_selection = df_selection.explode(col_list)
        df_selection.reset_index(drop=True, inplace=True)
        # rename col_list to col_short
        df_selection.rename(columns={col_list: col_short}, inplace=True)
        # convert the short column with a pair to a list

        if new_column_names == None:
            new_column_names = ["el_0", "el_1"]
        for i, element_name in enumerate(new_column_names):
            df_selection[f"{col_short}.{element_name}"] = df_selection.apply(
                lambda row: transform_to_list(str(row[col_short]))[i],
                axis=1
            )

        df_selection[col_short] = df_selection[col_short].str.split(",")

    return df_selection