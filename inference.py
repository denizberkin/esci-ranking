import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import lightgbm as lgbm

from utils.save import load_model
from utils.variables import MODEL_SAVE_FOLDER
from utils.plot import plot_importances


def test(model: lgbm.Booster = None, 
          test_df: pd.DataFrame = None,
          feature_columns: list[str] = None):
    if model is None:
        model: lgbm.Booster = load_model(os.path.join(MODEL_SAVE_FOLDER, "model_best.pkl"))
    
    plot_importances(model)

    # TODO: testing logic, metrics etc...
    """
    NUM FEATURES:  2 - only cossim (tfidf, sentence transformer)
    Processing Fold 1
    Validation Fold 1 - NDCG: 0.9105, Kendall Tau: nan, Weighted Tau: nan
    Processing Fold 2
    Validation Fold 2 - NDCG: 0.9110, Kendall Tau: nan, Weighted Tau: nan
    Processing Fold 3
    Validation Fold 3 - NDCG: 0.9072, Kendall Tau: nan, Weighted Tau: nan
    Processing Fold 4
    Validation Fold 4 - NDCG: 0.9091, Kendall Tau: nan, Weighted Tau: nan
    Processing Fold 5
    Validation Fold 5 - NDCG: 0.9080, Kendall Tau: nan, Weighted Tau: nan

    Metrics of Best Model - Fold 2: NDCG: 0.9110, Kendall Tau: nan, Weighted Tau: nan
    """

    """
    
    """




if __name__ == "__main__":
    test()