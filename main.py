import os

import numpy as np
import pandas as pd
import re


from abydos.distance import Levenshtein
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, ndcg_score
import lightgbm as lgbm

from utils.load import load_df
from utils.preprocess import preprocess_text, levenshtein, levenshtein_norm
from utils.variables import SCORE_MAP, ROOT_FOLDER, COLUMNS_TO_PROCESS

pd.options.mode.chained_assignment = None


def process_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ fill na with '' and apply `preprocess_text` function to columns"""
    for col in COLUMNS_TO_PROCESS:
        df[col] = df[col].fillna("").apply(lambda x: preprocess_text(x))
    return df


def combined_column(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """ combine columns into one"""
    df["combined"] = df[columns].apply(lambda r: " ".join(r), axis=1)
    return df



if __name__ == "__main__":

    train_filenames = [f for f in os.listdir(ROOT_FOLDER) if f.startswith("train")]
    test_filenames = [f for f in os.listdir(ROOT_FOLDER) if f.startswith("test")]

    df = load_df(train_filenames[: 1])
    df_test = load_df(test_filenames[: 1])

    print("DF COLUMNS: ", df.columns)
    print("DF TEST COLUMNS: ", df_test.columns)
    print("DF SHAPE: ", df.shape)
    print("DF TEST SHAPE: ", df_test.shape)

    df = process_columns(df)
    df_test = process_columns(df_test)

    # df = combined_column(df, COLUMNS_TO_PROCESS)
    # df_test = combined_column(df_test, COLUMNS_TO_PROCESS)

    df["labels"] = df["esci_label"].apply(lambda x: SCORE_MAP[x])  # map scores
    print("LABEL DIST:\n", df["labels"].value_counts())

    df = df[: 1000]  # for testing purposes

    # calculate levenshtein dist between query and features
    # q - product_title
    # q - product_description
    # q - product_bullet_points
    # q - product_brand
    # q - product_color
    # q - combined (gonna be less for sure)
    df["levenshtein_title"] = df[["query", "product_title"]].apply(lambda x: levenshtein_norm(x["query"], x["product_title"]), axis=1)
    df["levenshtein_desc"] = df[["query", "product_description"]].apply(lambda x: levenshtein_norm(x["query"], x["product_description"]), axis=1)
    # df["levenshtein_bullet"] = df[["query", "product_bullet_points"]].apply(lambda x: levenshtein_norm(x["query"], x["product_bullet_points"]), axis=1)
    # df["levenshtein_brand"] = df[["query", "product_brand"]].apply(lambda x: levenshtein_norm(x["query"], x["product_brand"]), axis=1)
    # df["levenshtein_color"] = df[["query", "product_color"]].apply(lambda x: levenshtein_norm(x["query"], x["product_color"]), axis=1)
    # df["levenshtein_combined"] = df[["query", "combined"]].apply(lambda x: levenshtein_norm(x["query"], x["combined"]), axis=1)  # baddddd

    # take columns that start with "levenshtein"
    levenshtein_cols = [col for col in df.columns if col.startswith("levenshtein")]

    print(df["levenshtein_title"].describe())
    print("\n", df["levenshtein_title"].median())
    print(df["levenshtein_title"].min(), df["levenshtein_title"].max())



    """
        LABEL DIST:
    labels
    3    100820
    2     49034
    0     30071
    1      4428
    QUERY V TITLE
    Name: count, dtype: int64
    count    1000.000000
    mean        0.804282
    std         0.071537
    min         0.425532
    25%         0.759100
    50%         0.800000
    75%         0.855852
    max         0.982249
    Name: levenshtein_title, dtype: float64

    0.8
    0.425531914893617 0.9822485207100592
    """