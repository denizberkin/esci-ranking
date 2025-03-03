import os
import joblib

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

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


def save_df_columns(df: pd.DataFrame, 
            columns: list[str],
            fn: str,
            embedding_folder: str = None):
    # use the provided embedding folder or the default one
    if embedding_folder is None:
        embedding_folder = EMBEDDING_FOLDER
        
    # create the directory if it doesn't exist
    os.makedirs(embedding_folder, exist_ok=True)
    df[columns].to_parquet(os.path.join(embedding_folder, fn), index=False)


def save_vectorizer(queries: np.ndarray, 
                    fn: str,
                    embedding_folder: str = None):
    """ loads embeddings from .npy files, constructs TFIDFVectorizer and saves it to file."""
    if embedding_folder is None:
        embedding_folder = EMBEDDING_FOLDER
    batch_size = 1000
    len_df = len(queries)

    # save the vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(2, 2), stop_words="english")
    for i in tqdm(range(0, len_df, batch_size), desc="re-fitting vectorizer"):
        end_idx = min(i + batch_size, len_df)
        batch = queries[i:end_idx]
        
        # Calculate TF-IDF for just this batch
        q_tfidf = vectorizer.fit(batch)

    joblib.dump(vectorizer, os.path.join(embedding_folder, fn))

