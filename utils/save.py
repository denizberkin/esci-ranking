import os
import joblib

import numpy as np
import pandas as pd

from utils.variables import EMBEDDING_FOLDER, ROOT_FOLDER


def load_df(filenames: list, root_folder: str = ROOT_FOLDER) -> pd.DataFrame:
    df = pd.concat([
        pd.read_parquet(os.path.join(root_folder, fn)) 
        for fn in filenames])
    return df


def save_embeddings2npy(embeddings: dict, fn: str = ""):
    """ input query and feature embeddings as dict, save them to EMBEDDING_FOLDER"""
    if not os.path.exists(EMBEDDING_FOLDER):
        os.makedirs(EMBEDDING_FOLDER)
    
    np.save(os.path.join(EMBEDDING_FOLDER, f"{fn}_query_embeddings.npy"), embeddings["query"])
    np.save(os.path.join(EMBEDDING_FOLDER, f"{fn}_feature_embeddings.npy"), embeddings["feature"])


def load_embeddings(fn: str = "") -> tuple[np.ndarray, np.ndarray]:
    """ load embeddings from EMBEDDING_FOLDER """
    query_embeddings = np.load(os.path.join(EMBEDDING_FOLDER, f"{fn}query_embeddings.npy"), 
                               allow_pickle=True
                               )
    feature_embeddings = np.load(os.path.join(EMBEDDING_FOLDER, f"{fn}feature_embeddings.npy"), 
                                 allow_pickle=True
                                 )
    return (query_embeddings, feature_embeddings)

    
def save_model(model, fn: str, as_txt: bool = False):
    """ only .pkl for now """
    if as_txt:
        with open(fn, "w") as f:
            f.write(str(model))
    else:
        joblib.dump(model, fn)


def load_model(fn: str):
    """ only .pkl for now"""
    return joblib.load(fn)


# save and overwrite the file given df columns
def save_df_columns(df: pd.DataFrame, 
            columns: list[str],
            fn: str):
    df[columns].to_parquet(os.path.join(EMBEDDING_FOLDER, fn), index=False)