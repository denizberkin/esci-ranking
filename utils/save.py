import numpy as np
import pickle
import joblib



def save_embeddings(embeddings, fn: str):
    with open(fn, "wb") as f:
        pickle.dump(embeddings, f)
    
def save_model(model, fn: str, as_txt: bool = False):
    """ fn: .pkl extension """
    if as_txt:
        with open(fn, "w") as f:
            f.write(str(model))
    else:
        joblib.dump(model, fn)
        
def save_st_model(st_model, fn: str):
    st_model.save_pretrained(fn)  # fn has no extension, saves model to folder
    # load into SentenceTransformer instance
    