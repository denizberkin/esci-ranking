import pandas as pd
from typing import Union, List

def load_df(path: str, ext: str):
    if ext == "csv":
        return pd.read_csv(path)
    elif ext == "parquet":
        return pd.read_parquet(path)
    else:
        raise NotImplementedError("check extension pls.")
    

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


def p2csv(df: pd.DataFrame, path: str, 
          idx: int = None, 
          columns_to_drop: list = None,
          get_small_us: bool = False):
    """ parquet to csv while necessary formatting options """
    if not idx:
        idx = len(df[df.columns[0]])
    if columns_to_drop:
        df.drop(columns_to_drop, inplace=True, axis=1)
    if get_small_us:
        df = df[df["small_version"] == 1]
        df = df[df["product_locale"] == "us"]
        df.drop(["product_locale", "small_version", "large_version"], inplace=True, axis=1)

    df[: idx].to_csv(path, index=False)
