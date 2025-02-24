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
from utils.preprocess import preprocess_text, scoring_function, cosine_sim, levenshtein
from utils.variables import SCORE_MAP, ROOT_FOLDER, COLUMNS_TO_PROCESS

pd.options.mode.chained_assignment = None


def process_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ fill na with '' and apply `preprocess_text` function to columns"""
    for col in COLUMNS_TO_PROCESS:
        df[col] = df[col].fillna("").apply(lambda x: preprocess_text(x))
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
