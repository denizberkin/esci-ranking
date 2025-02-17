import os
from copy import deepcopy
import joblib

import datasets

import numpy as np
import pandas as pd
import lightgbm as lgbm 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score

from utils.load import load_df, read_combine_parquets, p2csv
from utils.preprocess import scoring_function, cosine_sim

pd.options.mode.chained_assignment = None


# XXX: use smoothed labels?, use scaled scoring?
SCORE_MAP = {"Exact": 1.0,  # exact
           "Substitute": 0.1,  # substitute
           "Complement": 0.01,  # complementary
           "Irrelevant": 0.0}  # irrelevant


def pipeline(df: pd.DataFrame
             )-> lgbm.LGBMRanker:
    df["relevance"] = df["esci_label"].apply(scoring_function,
                                                    args=(SCORE_MAP,))

    vectorizer = TfidfVectorizer(max_features=100)
    df["query_embed"] = list(vectorizer.fit_transform(df['query']).toarray())
    df["title_embed"] = list(vectorizer.transform(df["product_title"]).toarray())

    
    df["query_title_sim"] = df.apply(
        cosine_sim, axis=1
    )
    
    df["query_len"] = df["query"].map(lambda x: len(x.split()))
    df["title_len"] = df["product_title"].map(lambda x: len(x.split()))
    df["brand_labelenc"] = LabelEncoder().fit_transform(df["product_brand"].fillna("unknown"))

    features = ['query_title_sim', 'query_len', 'title_len', 'brand_labelenc']
    x = df[features]
    y = df['relevance']

    train_mask = df['split'] == 'train'
    val_mask = df['split'] == 'val'
    test_mask = df['split'] == 'test'

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    train_groups = df[train_mask].groupby("query_id").size().to_list()
    val_groups = df[val_mask].groupby("query_id").size().to_list()

    lgbm_ranker = lgbm.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="gbdt",
    n_estimators=200,
    learning_rate=0.05
    )
    
    # lgbm_ranker.fit(x_train, y_train,
    #                 eval_metric="ndcg",
    #                 eval_set=[(x_val, y_val)],
    #                 group=train_groups,
    #                 eval_group=[val_groups])
    
    # y_test_pred = lgbm_ranker.predict(x_test)
    # ndcg_score_test = ndcg_score(y_test, y_test_pred)
    # print("NDCG on test: ", ndcg_score_test)

    joblib.dump(lgbm_ranker, "ranker_model.pkl")
    return lgbm_ranker



def main(**kwargs):
    """ input: paths to  dfs """
    df_examples = load_df(kwargs["ex"], "parquet")
    df_products = load_df(kwargs["pr"], "parquet")
    # df_sources = load_df(kwargs["src"], "csv")

    # print(df_examples.head())
    # print(df_products.head())
    # print(df_sources.head())
    
    # join examples and products on product_id

    # pipeline(df_joined_small[:5000])




if __name__ == "__main__":
    ROOT = os.path.join(os.getcwd(), "formatted_esci")
    CSV_ROOT = os.path.join(os.getcwd(), "formatted_csv")

    # examples = os.path.join(ROOT, "dataset_examples.parquet")
    # products = os.path.join(ROOT, "dataset_products.parquet")
    # sources = os.path.join(ROOT, "dataset_sources.csv")
    
    df = pd.read_csv("formatted_csv/train-00000_sample1000.csv")

    # convert relevance to score 
    df["relevance"] = df["esci_label"].apply(scoring_function,
                                                    args=(SCORE_MAP,))
    
    print(df["product_id"].value_counts())
    
    vectorizer = TfidfVectorizer(max_features=300)
    df["query_embed"] = list(vectorizer.fit_transform(df['query']).toarray())
    df["title_embed"] = list(vectorizer.transform(df["product_title"]).toarray())

    print(df["query_embed"][0])

    
    """
    filenames = os.listdir(ROOT)
    train_filenames = [os.path.join(ROOT, file) for file in filenames if file.startswith("train")]
    test_filenames = [os.path.join(ROOT, file) for file in filenames if file.startswith("test")]

    dataset = read_combine_parquets(train_filenames[:2], t_return="list")

    col2drop = ["product_bullet_point"]
    p2csv(dataset[0], os.path.join(CSV_ROOT, "train-00000_sample1000.csv"), columns_to_drop=col2drop, get_small_us=True, idx=1000)
    """
    # main(ex=examples, pr=products, src=sources)

    # ds = datasets.load_dataset("tasksource/esci")
    # print(ds)
    print()  # 