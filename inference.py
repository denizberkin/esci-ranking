import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import lightgbm as lgbm

import lightgbm as lgbm
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau, weightedtau

from utils.save import load_model
from utils.variables import MODEL_SAVE_FOLDER, NDCG_AT_K, SCORE_MAP
from utils.plot import plot_importances
from pipeline import preprocess_pipeline


def test(model: lgbm.Booster = None, 
          test_df: pd.DataFrame = None,
          feature_columns: list[str] = None,
          plot_feature_importance: bool = True):
    if model is None:
        model: lgbm.Booster = load_model(os.path.join(MODEL_SAVE_FOLDER, "model_best.pkl"))
        
    if plot_feature_importance:
        plot_importances(model)
   
    if test_df is None:
        print("No test DataFrame provided, returning...")
        return None

    print(f"Testing on {len(test_df)} samples")
    
    ndcg_scores = []
    kendall_scores = []
    weightedtau_scores = []
    
    results_by_query = {}
    
    has_labels = "esci_label" in test_df.columns or "labels" in test_df.columns
    
    if has_labels and "labels" not in test_df.columns:
        test_df["labels"] = test_df["esci_label"].apply(lambda x: SCORE_MAP[x])
        
    for qid, group in tqdm(test_df.groupby("query_id"), desc="evaluating test queries"):
        x_query = group[feature_columns]
        y_pred = model.predict(x_query, num_iteration=model.best_iteration).reshape(1, -1)
        
        # store preds
        results_by_query[qid] = {
            "predictions": y_pred.flatten(),
            "example_ids": group["example_id"].values,
            "product_ids": group["product_id"].values if "product_id" in group.columns else None
        }
        
        if has_labels:
            y_true = group["labels"].values.reshape(1, -1)
            results_by_query[qid]["true_labels"] = y_true.flatten()
            
            if group.shape[0] < 2:
                continue
                
            # Calculate metrics
            ndcg = ndcg_score(y_true, y_pred, k=NDCG_AT_K)
            
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            
            tau_kendall, _ = kendalltau(y_true_flat, y_pred_flat)
            tau_weighted, _ = weightedtau(y_true_flat, y_pred_flat)
            
            # Handle NaN values
            if np.isnan(tau_kendall):
                tau_kendall = np.float64(0.0)
            if np.isnan(tau_weighted):
                tau_weighted = np.float64(0.0)
            
            ndcg_scores.append(ndcg)
            kendall_scores.append(tau_kendall)
            weightedtau_scores.append(tau_weighted)
            
    results = {"results_by_query": results_by_query}
    
    # Add metrics if ground truth was available
    if has_labels and ndcg_scores:
        metrics = {
            "avg_ndcg": np.mean(ndcg_scores),
            "avg_kendall_tau": np.mean(kendall_scores),
            "avg_weighted_tau": np.mean(weightedtau_scores)
        }
    
        results["metrics"] = metrics
        
        print(f"test Metrics - NDCG at {NDCG_AT_K}: {metrics['avg_ndcg']:.4f}, "
              f"Kendall Tau: {metrics['avg_kendall_tau']:.4f}, "
              f"Weighted Tau: {metrics['avg_weighted_tau']:.4f}")
    
    # Save predictions
    results_df = []
    for qid, data in results_by_query.items():
        for i, (pred, ex_id) in enumerate(zip(data["predictions"], data["example_ids"])):
            row = {
                "query_id": qid,
                "example_id": ex_id,
                "product_id": data["product_ids"][i] if data["product_ids"] is not None else None,
                "predicted_score": pred,
            }
            if has_labels:
                row["true_label"] = data["true_labels"][i]
            results_df.append(row)
    
    results_df = pd.DataFrame(results_df)
    
    # Save predictions to csv
    output_dir = os.path.join(MODEL_SAVE_FOLDER, "predictions")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    print(f"Saved predictions to {os.path.join(output_dir, 'test_predictions.csv')}")
    
    return results


if __name__ == "__main__":
    test()