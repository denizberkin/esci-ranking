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
    # Parse command line arguments if needed
    import argparse
    parser = argparse.ArgumentParser(description='train and/or test the model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--train_limit', type=int, default=None, help='Limit training data for testing')
    args = parser.parse_args()

    if not args.train and not args.test:
        args.train = True
        args.test = True

    if args.train:
        train_filenames = [f for f in os.listdir(ROOT_FOLDER) if f.startswith("train")]
        df = load_df(train_filenames[:])  # no effect, saved the processed df and loading in "pipeline"
        
        print("DF COLUMNS: ", df.columns)
        print("DF SHAPE: ", df.shape)

        # df = df[: 1000]
        # print(f"Limited training data to {args.train_limit} samples")

        # Preprocess training data
        df, feature_columns = preprocess_pipeline(df)

        # Plot data characteristics if needed
        plot_df_corr(df[feature_columns + ["labels"]])
        # plot_column_skewness(df[feature_columns])

        print("NUM FEATURES: ", len(feature_columns))
        best_model = train(df, feature_columns)
    else:
        best_model = None
        feature_columns = None

    # Test model if testing
    if args.test:
        test_filenames = [f for f in os.listdir(ROOT_FOLDER) if f.startswith("test")]
        df_test = load_df(test_filenames[:1])
        
        print("DF TEST COLUMNS: ", df_test.columns)
        print("DF TEST SHAPE: ", df_test.shape)
        
        # Process test data through the same pipeline
        df_test, test_feature_columns = preprocess_pipeline(df_test, is_test=True)
        
        # Make sure we use the same feature columns as training if available
        if feature_columns is None:
            print("Using feature columns from test preprocessing")
            feature_columns = test_feature_columns
        
        # Run inference
        test_results = test(
            model=best_model,
            test_df=df_test,
            feature_columns=feature_columns
        )