
import numpy as np
import os
import soundfile as sf
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import jensenshannon


def normalize_embeddings(embeddings):
    mean = embeddings.mean(axis=1, keepdims=True)
    std = embeddings.std(axis=1, keepdims=True)
    return (embeddings - mean) / std

def softmax(x, axis=None):
    """
    Compute softmax values for each set of scores in x along the specified axis.
    """
    e_x = np.exp((x - np.max(x, axis=axis, keepdims=True))/1)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def js_divergence(embedding_A, embedding_B):
    """
    Calculate the Jensen-Shannon divergence between two embeddings' probability distributions.
    """
    js_divergence_score =jensenshannon(embedding_A, embedding_B)**2 #js distance -> js divergence
    assert js_divergence_score >= 0, "improper probability distribution in embeddings detected!"
    
    return js_divergence_score