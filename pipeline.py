import os

import pandas as pd
from rapidfuzz.distance import Levenshtein as rapid_levenshtein

from utils.preprocess import process_columns, combined_column, tfidf_cosine_sim, \
    prefix_match, postfix_match, additional_features, sentence_transformer_cosine_sim
from utils.process_color import colour_match, colour_normalize
from utils.variables import COLUMNS_TO_PROCESS, SCORE_MAP, ROOT_FOLDER

from utils.logger import log_time


@log_time
def time_process_columns(df):
    return process_columns(df)

@log_time
def time_combined_column(df, columns_to_process):
    return combined_column(df, columns_to_process)

@log_time
def time_levenshtein_title(df) -> tuple[pd.DataFrame, list[str]]:
    df["levenshtein_title"] = df[["query", "product_title"]].apply(lambda x: rapid_levenshtein.normalized_distance(x["query"], x["product_title"]), axis=1)
    return df, ["levenshtein_title"]

@log_time
def time_normalize_colours(df) -> tuple[pd.DataFrame, list[str]]:
    df["product_color"] = df["product_color"].apply(lambda x: colour_normalize(x))
    return df, ["product_color"]

@log_time
def time_colour_match(df) -> tuple[pd.DataFrame, list[str]]:
    df["colour_match"] = df[["query", "product_color"]].apply(lambda r: colour_match(r["query"], r["product_color"]), axis=1)
    print(df["colour_match"].value_counts(), "\n")
    return df, ["colour_match"]

@log_time
def time_prefix_match(df) -> tuple[pd.DataFrame, list[str]]:
    df["prefix_match"] = df[["query", "product_title"]].apply(lambda x: prefix_match(x["query"], x["product_title"]), axis=1)
    return df, ["prefix_match"]

@log_time
def time_postfix_match(df) -> tuple[pd.DataFrame, list[str]]:
    df["postfix_match"] = df[["query", "product_title"]].apply(lambda x: postfix_match(x["query"], x["product_title"]), axis=1)
    return df, ["postfix_match"]

@log_time
def time_additional_features(df) -> tuple[pd.DataFrame, list[str]]:
    df, names_additional_feature = additional_features(df)
    return df, names_additional_feature

@log_time
def time_tfidf_cosine_sim(df) -> tuple[pd.DataFrame, list[str]]:
    df, names_tfidf_column = tfidf_cosine_sim(df, save_embeddings=True)
    return df, names_tfidf_column

@log_time
def time_sentence_transformer_cosine_sim(df):
    df, names_st_column = sentence_transformer_cosine_sim(df, save_embeddings=True)
    return df, names_st_column


def preprocess_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # in the case we comment out columns, for it to not throw errors
    levenshtein_title_column, colour_match_column = [], []
    prefix_match_column, names_st_column = [], []
    postfix_match_column, names_additional_feature = [], []
    names_tfidf_column = []

    print("Start preprocessing pipeline")
    save_current_features_to = os.path.join(ROOT_FOLDER, "df_without_cosinesims.csv")
    if os.path.exists(save_current_features_to):
        # only load the calculated features
        df = pd.read_csv(save_current_features_to)
        # preprocess, remove punc, lower case etc.
        df = time_process_columns(df)

    else:
        # preprocess, remove punc, lower case etc.
        df = time_process_columns(df)

        # add combined column
        df, names_combined_column = time_combined_column(df, COLUMNS_TO_PROCESS)
        
        # map labels
        df["labels"] = df["esci_label"].apply(lambda x: SCORE_MAP[x])  # map scores

        # calculate levenshtein dist between query and features
        df, levenshtein_title_column = time_levenshtein_title(df)

        # normalize colours
        # df, _ = time_normalize_colours(df)

        # normalized colours to a certain top-k colour set,
        # can now obtain a feature if a colour matches with query or not
        # df, colour_match_column = time_colour_match(df)

        # prefix, postfix match
        # df, prefix_match_column = time_prefix_match(df)
        # df, postfix_match_column = time_postfix_match(df)

        # additional length-related features
        df, names_additional_feature = time_additional_features(df)

        print("Saving upto cosine sim calculations")
        df.to_csv(save_current_features_to, index=False)

    df, names_tfidf_column = time_tfidf_cosine_sim(df)
    print("TFIDF DONE!!!")
    df, names_st_column = time_sentence_transformer_cosine_sim(df)

    feature_columns = []
    feature_columns.extend(levenshtein_title_column)
    # feature_columns.extend(colour_match_column)
    # feature_columns.extend(prefix_match_column)
    # feature_columns.extend(postfix_match_column)
    feature_columns.extend(names_additional_feature)
    feature_columns.extend(names_tfidf_column)
    feature_columns.extend(names_st_column)
    feature_columns.extend(["levenshtein_title"])
    feature_columns.extend(["longest_common_substring_ratio", "longest_common_subsequence_ratio",
                "token_overlap", "query_length", "product_title_length", "length_ratio"])
    

    return df, feature_columns