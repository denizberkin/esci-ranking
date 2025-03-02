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

from utils.variables import COLUMNS_TO_PROCESS, ST_MODEL_NAME, EMBEDDING_FOLDER, \
      ST_COS_SIM_FN, TFIDF_COS_SIM_FN, NUM_THREADS
from utils.save import save_embeddings2npy, load_embeddings, save_df_columns
from utils.logger import log_time



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


@log_time
def time_longest_common_substring(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: longest_common_substring(r["query"], r["product_title"]), axis=1)

@log_time
def time_longest_common_subsequence(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: longest_common_subsequence(r["query"], r["product_title"]), axis=1)

@log_time
def time_qf_IOU(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: qf_IOU(r["query"], r["product_title"]), axis=1)


def additional_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """ calculate additional features """
    df["longest_common_substring_ratio"] = time_longest_common_substring(df)
    df["longest_common_subsequence_ratio"] = time_longest_common_subsequence(df)
    df["token_overlap"] = time_qf_IOU(df)
    df["query_length"] = df["query"].apply(lambda r: len(r.split()))  # ????????????????????
    df["product_title_length"] = df["product_title"].apply(lambda r: len(r.split()))  # may be unnecessary, more so if product_title is not used
    df["length_ratio"] = df["query_length"] / (df["product_title_length"] + 1e-5)  # avoid zerodiv
    return df, ["longest_common_substring_ratio", "longest_common_subsequence_ratio",
                "token_overlap", "query_length", "product_title_length", "length_ratio"]


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

def tfidf_cosine_sim(df: pd.DataFrame,
                     save_embeddings: bool = False,
                     batch_size: int = 1000,
                     embedding_folder: str = None
                     ) -> tuple[pd.DataFrame, list[str]]:
    # Use the provided embedding folder or the default one
    if embedding_folder is None:
        embedding_folder = EMBEDDING_FOLDER
    
    # Create the embedding folder if it doesn't exist
    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cos_sim_func = cosine_sim_gpu if device == "cuda" else cosine_sim_by_batch
    
    # Check if we already have pre-calculated cosine similarities
    cos_sim_path = os.path.join(embedding_folder, TFIDF_COS_SIM_FN)
    if os.path.exists(cos_sim_path):
        print(f"FOUND CALCULATED {TFIDF_COS_SIM_FN} in {embedding_folder}, loading...")
        cosine_sims = pd.read_parquet(cos_sim_path)["tfidf_cosine_sim"].tolist()
        len_cosine_sims = len(cosine_sims)
        len_df = len(df[df.columns[0]])
        
        # Only calculate additional similarities if needed
        if len_df > len_cosine_sims:
            print(f"Computing additional similarities for {len_df - len_cosine_sims} records...")
            # Create TF-IDF vectorizer just for new data
            tfidf = TfidfVectorizer(max_features=1000, ngram_range=(2, 2), stop_words="english")
            
            # Process new data in batches to save memory
            new_sims = []
            for i in range(len_cosine_sims, len_df, batch_size):
                end_idx = min(i + batch_size, len_df)
                batch_df = df.iloc[i:end_idx]
                
                # Calculate TF-IDF for just this batch
                q_tfidf = tfidf.fit_transform(batch_df["query"]).toarray()
                f_tfidf = tfidf.transform(batch_df["product_title"]).toarray()
                
                # Calculate similarities for this batch
                batch_sims = cos_sim_func(q_tfidf, f_tfidf)
                new_sims.extend(batch_sims)
                
                # Clean up to free memory
                del q_tfidf, f_tfidf
                
            cosine_sims.extend(new_sims)
            
            # Save the updated similarities
            if save_embeddings:
                save_df = pd.DataFrame({"tfidf_cosine_sim": cosine_sims, "example_id": df["example_id"][:len(cosine_sims)]})
                save_df.to_parquet(cos_sim_path)
        
        df["tfidf_cosine_sim"] = cosine_sims[:len_df]
    
    else:
        print("Calculating all similarities from scratch...")
        # Process in batches to avoid memory issues
        all_sims = []
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(2, 2), stop_words="english")
        
        for i in tqdm(range(0, len(df), batch_size), desc="computing tf-idf similarities"):
            end_idx = min(i + batch_size, len(df))
            batch_df = df.iloc[i: end_idx]
            
            q_tfidf = tfidf.fit_transform(batch_df["query"]).toarray()
            f_tfidf = tfidf.transform(batch_df["product_title"]).toarray()
            
            batch_sims = cos_sim_func(q_tfidf, f_tfidf)
            all_sims.extend(batch_sims)
            
            del q_tfidf, f_tfidf
        
        df["tfidf_cosine_sim"] = all_sims
        
        # Save the results
        if save_embeddings:
            save_df = pd.DataFrame({"tfidf_cosine_sim": all_sims, "example_id": df["example_id"][:len(all_sims)]})
            save_df.to_parquet(cos_sim_path)
    
    print("TFIDF COSSIM FINISHED!!")
    return df, ["tfidf_cosine_sim"]


def sentence_transformer_cosine_sim(df: pd.DataFrame, 
                                    model_name: str = ST_MODEL_NAME,
                                    save_embeddings: bool = False,
                                    embedding_folder: str = None
                                    ) -> tuple[pd.DataFrame, list[str]]:
    # Use the provided embedding folder or the default one
    if embedding_folder is None:
        embedding_folder = EMBEDDING_FOLDER
    
    # Create the embedding folder if it doesn't exist
    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cos_sim_path = os.path.join(embedding_folder, ST_COS_SIM_FN)
    cos_sim_found = os.path.exists(cos_sim_path)
    
    query_embeddings_path = os.path.join(embedding_folder, "query_embeddings.npy")
    feature_embeddings_path = os.path.join(embedding_folder, "feature_embeddings.npy")
    
    if os.path.exists(query_embeddings_path) and os.path.exists(feature_embeddings_path) and not cos_sim_found:
        print("LOADING EMBEDDINGS!")
        query_embeddings = np.load(query_embeddings_path, allow_pickle=True)
        combined_embeddings = np.load(feature_embeddings_path, allow_pickle=True)
        
    elif not cos_sim_found:  # compute them
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
            # Save embeddings to the specified folder
            np.save(query_embeddings_path, query_embeddings)
            np.save(feature_embeddings_path, combined_embeddings)
    else: 
        print(f"FOUND CALCULATED {ST_COS_SIM_FN} in {embedding_folder}, passing the load of embeddings!")

    print("STARTING COSSIM!!")
    cos_sim_func = cosine_sim_gpu if device == "cuda" else cosine_sim_by_batch

    if os.path.exists(cos_sim_path):  # if embeddings were saved
        cosine_sims = pd.read_parquet(cos_sim_path)["st_cosine_sim"].tolist()
        len_cosine_sims = len(cosine_sims)
        len_df = len(df[df.columns[0]])
        if len_df > len_cosine_sims:  # if lens not match, compute the rest and combine with loaded
            cosine_sims.extend(cos_sim_func(query_embeddings[len_cosine_sims: len_df], combined_embeddings[len_cosine_sims: len_df]))
        df["st_cosine_sim"] = cosine_sims[: len_df]  # covers else: where len_df < len_cosine_sims
    else:  # compute from scratch if nothing is loaded
        df["st_cosine_sim"] = cos_sim_func(query_embeddings, combined_embeddings)[: len(df[df.columns[0]])]  # match and only take the loaded sims
    
    # save st_cosine_sim column, with example_id to be able to re-track
    save_df_columns(df, ["st_cosine_sim", "example_id"], ST_COS_SIM_FN, embedding_folder=embedding_folder)
    
    if not cos_sim_found:
        del query_embeddings, combined_embeddings
    
    print("ST COSSIM FINISHED!!")
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