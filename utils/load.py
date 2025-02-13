import pandas as pd


def load_df(path: str, ext: str):
    if ext == "csv":
        return pd.read_csv(path)
    elif ext == "parquet":
        return pd.read_parquet(path)
    else:
        raise NotImplementedError("check extension pls.")