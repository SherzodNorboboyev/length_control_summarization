import json
import random
import numpy as np
import torch

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def length_bucket(n_words: int, short_thr: int = 60, medium_thr: int = 100):
    if n_words < short_thr:
        return "<SHORT>"
    elif n_words < medium_thr:
        return "<MEDIUM>"
    return "<LONG>"

def batch_iter(iterable, n: int):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]