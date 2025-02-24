import os
import pandas as pd
from typing import Union, List

from utils.variables import ROOT_FOLDER


def load_df(path: str, ext: str):
    if ext == "csv":
        return pd.read_csv(path)
    elif ext == "parquet":
        return pd.read_parquet(path)
    else:
        raise NotImplementedError("check extension pls.")


def load_df(filenames: list, root_folder: str = ROOT_FOLDER) -> pd.DataFrame:
    df = pd.concat([
        pd.read_parquet(os.path.join(root_folder, fn)) 
        for fn in filenames])
    return df


def read_combine_parquets(fn_list: list, t_return: str = "df") -> Union[pd.DataFrame, 
                                                                        List[pd.DataFrame]]:
    df: pd.DataFrame = None
    df_list = []
    for fn in fn_list:
        df_list.append(pd.read_parquet(fn))
    if t_return == "df":
        df = df_list[0]
        for e in df_list[1:]:
            df.join(e)

    return df if t_return == "df" else df_list


