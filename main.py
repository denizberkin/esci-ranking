import os

import numpy as np
import pandas as pd
import re
from rapidfuzz.distance import Levenshtein as rapid_levenshtein


from inference import test
from train import train

from utils.logger import log_time
from utils.save import load_df
from utils.plot import plot_df_corr, plot_column_skewness
from pipeline import preprocess_pipeline

from utils.variables import SCORE_MAP, ROOT_FOLDER, COLUMNS_TO_PROCESS


pd.options.mode.chained_assignment = None



if __name__ == "__main__":

    train_filenames = [f for f in os.listdir(ROOT_FOLDER) if f.startswith("train")]
    test_filenames = [f for f in os.listdir(ROOT_FOLDER) if f.startswith("test")]

    df = load_df(train_filenames[:1])  # no effect, saved the processed df and loading in "pipeline"
    df_test = load_df(test_filenames[:1])

    print("DF COLUMNS: ", df.columns)
    print("DF TEST COLUMNS: ", df_test.columns)
    print("DF SHAPE: ", df.shape)
    print("DF TEST SHAPE: ", df_test.shape)

    # df = df[: 1000]  # for testing purposes

    df, feature_columns = preprocess_pipeline(df)
    # df_test = preprocess_pipeline(df_test)

    plot_df_corr(df[feature_columns])
    plot_column_skewness(df[feature_columns])

    print("NUM FEATURES: ", len(feature_columns))
    best_model = train(df, feature_columns)
    
    test(best_model
        test_df=df_test, 
        feature_columns=feature_columns)