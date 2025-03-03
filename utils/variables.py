import os

STREAMLIT = True
PLOT_FOLDER = "plots/"
MODEL_SAVE_FOLDER = "models/"
ROOT_FOLDER = "formatted_esci/"
EMBEDDING_FOLDER = "embeddings/"

ST_MODEL_NAME = "all-MiniLM-L6-v2"
ST_COS_SIM_FN = "st_cosine_sim.parquet"
TFIDF_COS_SIM_FN = "tfidf_cosine_sim.parquet"

NDCG_AT_K = 3

NUM_THREADS = 8  # if no saved embeddings and cuda not available


COLUMNS_TO_PROCESS =  [  # "query",    # data leak? 
                       "product_title", 
                       "product_description",
                       "product_brand",
                       "product_color"]  # product_text is already a combined version


SCORE_MAP = {
    "Exact": 3,  # exact
    "Substitute": 2,  # substitute
    "Complement": 1,  # complementary
    "Irrelevant": 0   # irrelevant
    }


# lgbm params
MODEL_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "verbose": -1
    }


COLOURS_TXT_FN = "unique_colours.txt"
with open(COLOURS_TXT_FN, "r+") as f:
    COLOURS = list(set(f.read().splitlines()))
    