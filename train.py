import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau, weightedtau
import lightgbm as lgbm

from utils.save import save_model
from utils.variables import MODEL_PARAMS, MODEL_SAVE_FOLDER

def train(df: pd.DataFrame, feature_columns: list):
    x = df[feature_columns]
    y = df["labels"]
    
    # track best model
    best_ndcg = -np.inf
    best_model = None
    best_fold = None
    
    gkf = GroupKFold(n_splits=5)
    
    for fold, (train_id, valid_id) in enumerate(tqdm(gkf.split(x, y, groups=df["query_id"]), total=5, desc="Folds")):
        tqdm.write(f"Processing Fold {fold + 1}")
        x_train, y_train = x.iloc[train_id], y.iloc[train_id]
        x_valid, y_valid = x.iloc[valid_id], y.iloc[valid_id]
        
        train_group = df.iloc[train_id].groupby("query_id").size().tolist()
        valid_group = df.iloc[valid_id].groupby("query_id").size().tolist()
        
        lgbm_train = lgbm.Dataset(x_train, y_train, group=train_group)
        lgbm_valid = lgbm.Dataset(x_valid, y_valid, group=valid_group)
        
        model = lgbm.train(params=MODEL_PARAMS,   # specified Ranking in params
                   train_set=lgbm_train, 
                   valid_sets=[lgbm_valid], 
                   num_boost_round=800)
        
        ndcg_scores = []
        for qid, group in tqdm(df.iloc[valid_id].groupby("query_id"), desc="Evaluating queries", leave=False):
            if group.shape[0] < 2:
                continue
            y_true = group["labels"].values.reshape(1, -1)
            x_query = group[feature_columns]
            y_pred = model.predict(x_query, num_iteration=model.best_iteration).reshape(1, -1)
            ndcg = ndcg_score(y_true, y_pred, k=3)
            ndcg_scores.append(ndcg)
        
        avg_ndcg = np.mean(ndcg_scores)
        tqdm.write("Validation NDCG Score for Fold {}: {:.4f}".format(fold + 1, np.mean(ndcg_scores)))
        
        if avg_ndcg > best_ndcg:
            best_ndcg = avg_ndcg
            best_model = model
            best_fold = fold + 1
            
    if best_model is not None:
        if not os.path.exists(MODEL_SAVE_FOLDER):
            os.makedirs(MODEL_SAVE_FOLDER)
        best_model_fn = os.path.join(MODEL_SAVE_FOLDER, f"model_fold_{best_fold}.pkl")
        save_model(model, best_model_fn, as_txt=False)
        tqdm.write(f"Best Model saved at {best_model_fn}")
    else:
        tqdm.write("??? No best model found ???")
        
    return best_model