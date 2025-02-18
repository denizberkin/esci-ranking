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


def levenshtein(x):
    a, b = x["query"], x["product_title"]
    return lev(a, b)


def lev(a: str, b: str):
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
            print(a, " ", b)
            print(d)
    return d[-1, -1]