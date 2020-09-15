import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import affinity_propagation
import logging
from numba import jit

def compute_normalize_similarity(X):
    S = cosine_similarity(X)
    S = 1/2 + 1/2*S
    return S


logging.basicConfig(format = "%(asctime)s %(message)s", level = logging.DEBUG)

df = pd.read_csv("timeline17.csv")
embedder = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")
logging.info("start embedding...")
X = embedder.encode(df.text)
X = X.astype(np.float32)
print(X.shape)
logging.info("start computing similarity...")
S = compute_normalize_similarity(X)
S = S.astype(np.float32)
del X
logging.info("start ap clustering...")
event_centers,labels = affinity_propagation(S)
print(len(labels))
