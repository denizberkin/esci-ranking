import os
import re

import numpy as np
import pandas as pd
import scipy.sparse
import torch
import scipy
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from utils.variables import COLUMNS_TO_PROCESS, ST_MODEL_NAME, EMBEDDING_FOLDER, NUM_THREADS
from utils.save import save_embeddings2npy, load_embeddings


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
    """ calculate additional features """
    df["longest_common_substring_ratio"] = df.apply(lambda r: longest_common_substring(r["query"], r["combined"]), axis=1)
    df["longest_common_subsequence_ratio"] = df.apply(lambda r: longest_common_subsequence(r["query"], r["combined"]), axis=1)
    df["token_overlap"] = df.apply(lambda r: qf_IOU(r["query"], r["combined"]), axis=1)
    df["query_length"] = df["query"].apply(lambda r: len(r.split()))  # ????????????????????
    df["combined_length"] = df["combined"].apply(lambda r: len(r.split()))  # may be unnecessary, more so if combined is not used
    df["length_ratio"] = df["query_length"] / (df["combined_length"] + 1e-5)  # avoid zerodiv
    return df, ["longest_common_substring_ratio", "longest_common_subsequence_ratio",
                "token_overlap", "query_length", "combined_length", "length_ratio"]


def longest_common_substring(query: str, feature: str) -> float:
    """ calculating longest common substring ratio between query and given feature """
    m, n = len(query), len(feature)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if query[i - 1] == feature[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])
    return max_len / max(m, n, 1)  # avoid zerodiv


def longest_common_subsequence(query: str, feature: str) -> float:
    """ calculating longest common subsequence ratio between query and given feature """
    m, n = len(query), len(feature)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if query[i - 1] == feature[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n] / max(m, n, 1)  # avoid zerodiv


def qf_IOU(query: str, feature: str) -> float:
    """ calculating overlap ratio -IOU- between query and given feature """
    query_tokens = set(query.split())  # split so each token is a word
    feature_tokens = set(feature.split())
    return len(query_tokens.intersection(feature_tokens)) / (len(query_tokens.union(feature_tokens)) + 1e-5)  # avoid zerodiv again


def prefix_match(query: str, feature: str) -> float:
    # Normalize texts as done in preprocess_text()
    if feature.startswith(query):
        return 1.0
    return 0.0


def postfix_match(query: str, feature: str) -> float:
    if feature.endswith(query):
        return 1.0
    return 0.0


# re-wrote tfidf to obtain cosine sim matrice - can extract the diagonal to get similarities between query and feature
def tfidf_cosine_sim(df: pd.DataFrame,
                     save_embeddings: bool = False
                     ) -> tuple[pd.DataFrame, list[str]]:
    """ calculate cosine similarity between query and features """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfidf = TfidfVectorizer(ngram_range=(2, 2), stop_words="english")
    query_tfidf = tfidf.fit_transform(df["query"])
    combined_tfidf = tfidf.transform(df["combined"])
    
    # tfidf outputs these as sparse matrices so we need to convert them to numpy arrays
    query_tfidf = query_tfidf.toarray()
    combined_tfidf = combined_tfidf.toarray()
    
    # save embeddings to npy file
    if save_embeddings:
        save_embeddings2npy({"query": query_tfidf,
                             "feature": combined_tfidf},
                             fn="tfidf_firstsample"
                             )
    
    print("STARTING COSSIM!!")
    cos_sim_func = cosine_sim_gpu if device == "cuda" else cosine_sim_by_batch
    df["tfidf_cosine_sim"] = cos_sim_func(query_tfidf, combined_tfidf)[: len(df[df.columns[0]])]  # match and only take the loaded sims
    del query_tfidf, combined_tfidf
    print("COSSIM FINISHED!!")
    return df, ["tfidf_cosine_sim"]


# default is https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, 22.7M params, 384 dim
def sentence_transformer_cosine_sim(df: pd.DataFrame, 
                                    model_name: str = ST_MODEL_NAME,
                                    save_embeddings: bool = False
                                    ) -> tuple[pd.DataFrame, list[str]]:
    """ calculate cosine similarity between query and features using sentence transformer """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.path.exists(EMBEDDING_FOLDER) and len(os.listdir(EMBEDDING_FOLDER)) > 0:  # bad check
        print("LOADING EMBEDDINGS!")  # TODO: timeit
        query_embeddings, combined_embeddings = load_embeddings()
        
    else:  # compute them
        print("COMPUTING EMBEDDINGS!")
        model = SentenceTransformer(model_name, device=device)
        if device == "cpu":  # mp
            print("Embedding with multiprocessing!")
            pool = model.start_multi_process_pool(["cpu"] * NUM_THREADS)
            query_embeddings = model.encode_multi_process(df["query"].tolist(), pool, chunk_size=100, show_progress_bar=True)
            combined_embeddings = model.encode_multi_process(df["combined"].tolist(), pool, chunk_size=100, show_progress_bar=True)
            model.stop_multi_process_pool(pool)
        else:
            query_embeddings = model.encode(df["query"].tolist(), show_progress_bar=True)
            combined_embeddings = model.encode(df["combined"].tolist(), show_progress_bar=True)
    
        if save_embeddings:
            save_embeddings2npy({"query": query_embeddings, 
                            "feature": combined_embeddings}
                            )

    print("STARTING COSSIM!!")
    cos_sim_func = cosine_sim_gpu if device == "cuda" else cosine_sim_by_batch
    df["st_cosine_sim"] = cos_sim_func(query_embeddings, combined_embeddings)[: len(df[df.columns[0]])]  # match and only take the loaded sims
    del query_embeddings, combined_embeddings
    print("COSSIM FINISHED!!")
    return df, ["st_cosine_sim"]


def cosine_sim_by_batch(q_embeddings: np.ndarray, 
                            f_embeddings: np.ndarray,
                            batch_size: int=20000) -> np.ndarray:
    """ relieved from memory constraints """
    similarities = []
    for i in tqdm(range(0, len(q_embeddings), batch_size), desc="Calculating cossim...", ):
        q_batch = q_embeddings[i:i+batch_size]
        f_batch = f_embeddings[i:i+batch_size]
        batch_similarities = cosine_similarity(q_batch, f_batch)
        similarities.extend(np.diagonal(batch_similarities))
    return similarities


def cosine_sim_gpu(q_embeddings, f_embeddings) -> np.ndarray:
    """ calculate cosine similarity between query and features """
    q_tensor = torch.tensor(q_embeddings, device="cuda")
    f_tensor = torch.tensor(f_embeddings, device="cuda")
    
    # Q * F = |Q| * |F| * cos(theta)  [consider theta=0 after normalization]
    q_tensor = torch.nn.functional.normalize(q_tensor, dim=1)
    f_tensor = torch.nn.functional.normalize(f_tensor, dim=1)
    
    # i noticed this after implementing
    # similarities = torch.nn.functional.cosine_similarity(q_tensor, f_tensor, dim=1)
    
    similarities = torch.sum(q_tensor * f_tensor, dim=1)
    return similarities.cpu().numpy()