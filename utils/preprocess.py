import re
import pandas as pd
import numpy as np

from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from utils.variables import COLUMNS_TO_PROCESS, ST_MODEL_NAME


def process_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ fill na with '' and apply `preprocess_text` function to columns"""
    for col in COLUMNS_TO_PROCESS:
        df[col] = df[col].fillna("").apply(lambda x: preprocess_text(x))
    return df


def combined_column(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, list[str]]:
    """ combine columns into one"""
    df["combined"] = df[columns].apply(lambda r: " ".join(r), axis=1)
    return df, ["combined"]


def preprocess_text(query: str) -> str:
    if query is None:
        return ""
    query = query.lower()
    query = re.sub(r"[^a-z0-9\s+]", "", query)
    return query.strip()


def levenshtein(a: str, b: str) -> np.float64:
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


def additional_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df["token_overlap"] = df.apply(lambda r: qf_overlap_ratio(r["query"], r["combined"]), axis=1)
    df["query_length"] = df["query"].apply(lambda r: len(r.split()))
    df["combined_length"] = df["combined"].apply(lambda r: len(r.split()))  # may be unnecessary, more so if combined is not used
    df["length_ratio"] = df["query_length"] / (df["combined_length"] + 1e-5)  # avoid zerodiv
    return df, ["token_overlap", "query_length", "combined_length", "length_ratio"]


def qf_overlap_ratio(query: str, feature: str) -> float:
    """ calculating overlap ratio -IOU- between query and given feature """
    query_tokens = set(query.split())  # split so each token is a word
    feature_tokens = set(feature.split())
    return len(query_tokens.intersection(feature_tokens)) / (len(query_tokens.union(feature_tokens)) + 1e-5)  # avoid zerodiv again


# re-wrote tfidf to obtain cosine sim matrice - can extract the diagonal to get similarities between query and feature
def tfidf_cosine_sim(df: pd.DataFrame
                     ) -> tuple[pd.DataFrame, list[str]]:
    """ calculate cosine similarity between query and features """
    tfidf = TfidfVectorizer(ngram_range=(1, 3), stop_words="english", min_df=2)
    qf_df = pd.concat([df["query"], df["combined"]], axis=1)
    tfidf.fit(qf_df)
    
    query_tfidf = tfidf.transform(df["query"])
    combined_tfidf = tfidf.transform(df["combined"])
    sim_matrice = cosine_similarity(query_tfidf, combined_tfidf)
    df["tfidf_cosine_sim"] = np.diag(sim_matrice)
    return df, [tfidf_cosine_sim]


# default is https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, 22.7M params, 384 dim
def sentence_transformer_cosine_sim(df: pd.DataFrame, 
                                    model_name: str = ST_MODEL_NAME
                                    ) -> tuple[pd.DataFrame, list[str]]:
    """ calculate cosine similarity between query and features using sentence transformer """
    model = SentenceTransformer(model_name)
    query_embeddings = model.encode(df["query"].tolist(), show_progress_bar=True)
    combined_embeddings = model.encode(df["combined"].tolist(), show_progress_bar=True)
    sim_matrice = cosine_similarity(query_embeddings, combined_embeddings)
    
    df["st_cosine_sim"] = np.diag(sim_matrice)
    return df, ["st_cosine_sim"]
