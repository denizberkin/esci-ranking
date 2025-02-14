import re
import pandas as pd
import numpy as np

from nltk import word_tokenize
import string

from scipy.spatial.distance import cosine


def lowercase(txt: str):
    return txt.lower()

# def remove_punctuation(txt: str):
#     return ''.join([w for w in word_tokenize(txt) if w not in string.punctuation])

def remove_whitespace(txt: str):
    return txt

def remove_punctuation(txt: str):
    return re.sub(r"\s+", " ", txt).strip()


def preprocess_pipeline(txt: str):
    txt = lowercase(txt)
    print(txt)
    txt = remove_punctuation(txt)
    print(txt)
    txt = remove_whitespace(txt)
    print(txt)
    quit()


def preprocess_df(df: pd.DataFrame, columns: list):
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype(str).map(preprocess_pipeline)
    return df


def scoring_function(x: str, map: dict) -> float:
    """ """
    return map[x]


def cosine_sim(x):
    # query - title similarity #  TODO: cosine sim for now, change it to something you have read in the documents maybe
    if np.isnan(cosine(x["query_embed"], x["title_embed"])):
        cos = 0
    else: 
        cos = 1 - cosine(x["query_embed"], x["title_embed"])
    
    return cos
    # cos_sim = lambda row: 1 - cosine(row["query_embed"], row["title_embed"]) if not \
    # np.isnan(cosine(row['query_embed'], row['title_embed'])) else 0