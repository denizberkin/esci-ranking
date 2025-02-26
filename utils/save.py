import os
import numpy as np
import joblib

from utils.variables import EMBEDDING_FOLDER


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