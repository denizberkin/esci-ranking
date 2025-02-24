import re
import pandas as pd
import numpy as np

from nltk import word_tokenize
import string

from scipy.spatial.distance import cosine



def preprocess_text(txt: str):
    if txt is None:
        return ""
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9\s+]", "", txt)
    return txt.strip()


def cosine_sim(x):
    # query - title similarity #  TODO: cosine sim for now, change it to something you have read in the documents maybe
    if np.isnan(cosine(x["query_embed"], x["title_embed"])):
        cos = 0
    else: 
        cos = 1 - cosine(x["query_embed"], x["title_embed"])
    
    return cos
    # cos_sim = lambda row: 1 - cosine(row["query_embed"], row["title_embed"]) if not \
    # np.isnan(cosine(row['query_embed'], row['title_embed'])) else 0


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