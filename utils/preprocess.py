import re
import pandas as pd
import numpy as np

from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from scipy.spatial.distance import cosine


def preprocess_text(txt: str):
    if txt is None:
        return ""
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9\s+]", "", txt)
    return txt.strip()


def levenshtein(a: str, b: str):
    la, lb = len(a) + 1, len(b) + 1
    d = np.zeros((la, lb))

    d[:, 0] = range(la)
    d[0, :] = range(lb)

    for j in range(1, lb):
        for i in range(1, la):
            if b[j - 1] == a[i - 1]:
                cost = 0
            else: cost = 1

            d[i, j] = min(d[i - 1, j] + 1,  # delete
                          min(d[i, j - 1] + 1, # insert
                              d[i - 1, j - 1] + cost  # sub 
                          )) 
    return d[-1, -1]


def levenshtein_norm(a: str, b: str) -> np.float64:
    return levenshtein(a, b) / max(max(len(a), len(b)), 1)  # second max in the case of both strings being empty


def qf_overlap_ratio(query: str, feature: str) -> float:
    """ calculating overlap ratio -IOU- between query and given feature """
    query_tokens = set(query.split())  # split so each token is a word
    feature_tokens = set(feature.split())
    return len(query_tokens.intersection(feature_tokens)) / (len(query_tokens.union(feature_tokens)) + 1e-5)  # avoid zerodiv


# re-wrote tfidf to obtain cosine sim matrice - can extract the diagonal to get similarities between query and feature
def tfidf_cosine_sim(df: pd.DataFrame) -> pd.DataFrame:
    """ calculate cosine similarity between query and features """
    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    qf_df = pd.concat([df["query"], df["combined"]], axis=1)
    tfidf.fit(qf_df)
    
    query_tfidf = tfidf.transform(df["query"])
    combined_tfidf = tfidf.transform(df["combined"])
    sim_matrice = cosine_similarity(query_tfidf, combined_tfidf)
    df["tfidf_cosine_sim"] = np.diag(sim_matrice)
    return df


# default is https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, 22.7M params, 384 dim
def sentence_transformer_cosine_sim(df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    """ calculate cosine similarity between query and features using sentence transformer """
    model = SentenceTransformer(model_name)
    query_embeddings = model.encode(df["query"].tolist(), show_progress_bar=True)
    combined_embeddings = model.encode(df["combined"].tolist(), show_progress_bar=True)
    sim_matrice = cosine_similarity(query_embeddings, combined_embeddings)
    
    df["st_cosine_sim"] = np.diag(sim_matrice)
    return df
