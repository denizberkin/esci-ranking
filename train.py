import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau, weightedtau
import lightgbm as lgbm

from utils.save import save_model
from utils.variables import MODEL_PARAMS, MODEL_SAVE_FOLDER, NDCG_AT_K

def train(df: pd.DataFrame, feature_columns: list):
    x = df[feature_columns]
    y = df["labels"]
    
    # track best model
    best_ndcg = -np.inf
    best_model = None
    best_fold = None
    best_kendall = None
    best_weighted_tau = None
    
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
        kendall_scores = []
        weightedtau_scores = []
        
        for qid, group in tqdm(df.iloc[valid_id].groupby("query_id"), desc="Evaluating queries", leave=False):
            if group.shape[0] < 2:
                continue
            y_true = group["labels"].values.reshape(1, -1)
            x_query = group[feature_columns]
            y_pred = model.predict(x_query, num_iteration=model.best_iteration).reshape(1, -1)
            
            # scoring
            ndcg = ndcg_score(y_true, y_pred, k=NDCG_AT_K)

            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            
            tau_kendall, _ = kendalltau(y_true_flat, y_pred_flat)
            tau_weighted, _ = weightedtau(y_true_flat, y_pred_flat)
            if np.isnan(tau_kendall):
                tau_kendall = np.float64(0.0)
            if np.isnan(tau_weighted):
                tau_weighted = np.float64(0.0)
            
            ndcg_scores.append(ndcg)
            kendall_scores.append(tau_kendall)
            weightedtau_scores.append(tau_weighted)

        
        avg_ndcg = np.mean(ndcg_scores)
        avg_kendall = np.mean(kendall_scores)
        avg_weightedtau = np.mean(weightedtau_scores)
        tqdm.write(f"Validation Fold {fold + 1} - NDCG: {avg_ndcg:.4f}, Kendall Tau: {avg_kendall:.4f}, Weighted Tau: {avg_weightedtau:.4f}")
        
        if avg_ndcg > best_ndcg:
            best_ndcg = avg_ndcg
            best_model = model
            best_fold = fold + 1
            
            # this actually is not the best kendall and weighted tau, 
            # TODO: update lists to dicts and keep track of all, this way we can choose a "best_metric" to compare from variables.py
            best_kendall = avg_kendall
            best_weighted_tau = avg_weightedtau
            
    if best_model is not None:
        if not os.path.exists(MODEL_SAVE_FOLDER):
            os.makedirs(MODEL_SAVE_FOLDER)
        best_model_fn = os.path.join(MODEL_SAVE_FOLDER, f"model_fold_{best_fold}.pkl")
        save_model(model, best_model_fn, as_txt=False)
        tqdm.write(f"Best Model saved at {best_model_fn}")
        tqdm.write(f"Metrics of Best Model - Fold {best_fold}: NDCG: {best_ndcg:.4f}, Kendall Tau: {best_kendall:.4f}, Weighted Tau: {best_weighted_tau:.4f}")
    else:
        tqdm.write("??? No best model found ???")
        
    return best_model