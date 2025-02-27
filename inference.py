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




if __name__ == "__main__":
    test()