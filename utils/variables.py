import os

MODEL_SAVE_FOLDER = "models/"
ROOT_FOLDER = "formatted_esci/"
EMBEDDING_FOLDER = "embeddings/"
NUM_THREADS = 6  # if no saved embeddings and cuda not available

COLUMNS_TO_PROCESS =  [  # "query",    # data leak? 
                       "product_title", 
                       "product_description",
                       "product_brand",
                       "product_color"]  # product_text is already a combined version


# XXX: use smoothed labels?, use scaled scoring?
SCORE_MAP = {
    "Exact": 3,  # exact
    "Substitute": 2,  # substitute
    "Complement": 1,  # complementary
    "Irrelevant": 0   # irrelevant
    }


ST_MODEL_NAME = "all-MiniLM-L6-v2"

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
    