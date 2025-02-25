import os

import numpy as np
import pandas as pd
import re
from rapidfuzz.distance import Levenshtein as rapid_levenshtein

from train import train
from utils.load import load_df
from utils.preprocess import preprocess_text, levenshtein, levenshtein_norm, additional_features, \
    qf_overlap_ratio, tfidf_cosine_sim, sentence_transformer_cosine_sim, \
    process_columns, combined_column
from utils.process_color import colour_normalize, save_colours_list, colour_match
from utils.variables import SCORE_MAP, ROOT_FOLDER, COLUMNS_TO_PROCESS

pd.options.mode.chained_assignment = None


if __name__ == "__main__":

    train_filenames = [f for f in os.listdir(ROOT_FOLDER) if f.startswith("train")]
    test_filenames = [f for f in os.listdir(ROOT_FOLDER) if f.startswith("test")]

    df = load_df(train_filenames[:1])
    df_test = load_df(test_filenames[: 1])

    print("DF COLUMNS: ", df.columns)
    print("DF TEST COLUMNS: ", df_test.columns)
    print("DF SHAPE: ", df.shape)
    print("DF TEST SHAPE: ", df_test.shape)

    # df = df[: 1000]  # for testing purposes
    
    df = process_columns(df)
    df_test = process_columns(df_test)

    df, names_combined_column = combined_column(df, COLUMNS_TO_PROCESS)
    df_test, _ = combined_column(df_test, COLUMNS_TO_PROCESS)

    df["labels"] = df["esci_label"].apply(lambda x: SCORE_MAP[x])  # map scores
    print("LABEL DIST:\n", df["labels"].value_counts())


    # calculate levenshtein dist between query and features
    # q - product_title
    # q - product_description
    # q - product_bullet_points
    # q - product_brand
    # q - product_color
    # q - combined (gonna be less for sure)
    df[["query", "product_title"]].apply(lambda x: rapid_levenshtein.normalized_distance(x["query"], x["product_title"]), axis=1)
    df["levenshtein_title"] = df[["query", "product_title"]].apply(lambda x: rapid_levenshtein.normalized_distance(x["query"], x["product_title"]), axis=1)
    # df["levenshtein_desc"] = df[["query", "product_description"]].apply(lambda x: levenshtein_norm(x["query"], x["product_description"]), axis=1)
    # df["levenshtein_bullet"] = df[["query", "product_bullet_points"]].apply(lambda x: levenshtein_norm(x["query"], x["product_bullet_points"]), axis=1)
    # df["levenshtein_brand"] = df[["query", "product_brand"]].apply(lambda x: levenshtein_norm(x["query"], x["product_brand"]), axis=1)
    # df["levenshtein_color"] = df[["query", "product_color"]].apply(lambda x: levenshtein_norm(x["query"], x["product_color"]), axis=1)
    # df["levenshtein_combined"] = df[["query", "combined"]].apply(lambda x: levenshtein_norm(x["query"], x["combined"]), axis=1)  # baddddd

    print(df["product_color"].describe())
    unique_colour_list = df["product_color"].value_counts()
    print("LEN: ", len(unique_colour_list))
    # save 100 colours with highest freq as known colours
    # corrected_unique_colour_list = save_colours_list(unique_colour_list.keys())

    # normalize colours
    df["product_color"] = df["product_color"].apply(lambda x: colour_normalize(x))
    print(df["product_color"].value_counts())

    # normalized colours to a certain top-k colour set,
    # can now obtain a feature if a colour matches with query or not
    df["colour_match"] = df[["query", "product_color"]].apply(lambda r: colour_match(r["query"], r["product_color"]), axis=1)

    print(df["colour_match"].value_counts())

    # take columns that start with "levenshtein"
    levenshtein_cols = [col for col in df.columns if col.startswith("levenshtein")]

    """
    print(df["levenshtein_title"].describe())
    print("\n", df["levenshtein_title"].median())
    print(df["levenshtein_title"].min(), df["levenshtein_title"].max())
    """

    df, names_additional_feature = additional_features(df)
    
    names_tfidf_column, names_st_column = [], []
    print("starting cos sim calcs!!")
    # df, names_tfidf_column = tfidf_cosine_sim(df)
    df, names_st_column = sentence_transformer_cosine_sim(df)

    
    feature_columns = []
    feature_columns.extend(names_combined_column);feature_columns.extend(levenshtein_cols)
    feature_columns.extend(names_additional_feature);feature_columns.extend(names_tfidf_column)
    feature_columns.extend(names_st_column);feature_columns.extend(["colour_match"])
    # extend columns created in main.py by hand!
    
    print(feature_columns)    
    train(df, feature_columns)