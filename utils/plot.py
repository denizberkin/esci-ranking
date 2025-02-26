import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgbm

from utils.variables import PLOT_FOLDER


def plot_df_corr(df: pd.DataFrame, fn: str = "df_corr.png") -> bool:
    fig = plt.figure(figsize=(16, 12))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    success = plt.savefig(os.path.join(PLOT_FOLDER, fn))
    plt.close(fig)
    return success

def plot_column_skewness(df: pd.DataFrame, fn: str = "skewness.png") -> bool:
    fig = plt.figure(figsize=(16, 12))
    sns.histplot(df.skew(), bins=30, kde=True)
    success = plt.savefig(os.path.join(PLOT_FOLDER, fn))
    plt.close(fig)
    return success


def plot_importances(model: lgbm.Booster,
                     fn: str = "feature_importances.png") -> bool:
    fig = plt.figure(figsize=(16, 12))
    lgbm.plot_importance(model, importance_type="split")
    plt.tight_layout()
    success = plt.savefig(os.path.join(PLOT_FOLDER, fn))
    plt.close(fig)
    return success