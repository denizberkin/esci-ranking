import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import lightgbm as lgbm

from utils.save import load_model
from utils.variables import MODEL_SAVE_FOLDER
from utils.plot import plot_importances


def test():
    model: lgbm.Booster = load_model(os.path.join(MODEL_SAVE_FOLDER, "model_best.pkl"))

    # plot feature importances
    plot_importances(model)





if __name__ == "__main__":
    test()