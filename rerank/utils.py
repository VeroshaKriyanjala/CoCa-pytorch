import numpy as np

def length_normalize(logprob: float, length: int, length_penalty: float = 0.7) -> float:
    lp = ((5 + length) ** length_penalty) / ((5 + 1) ** length_penalty)
    return logprob / lp

def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    m, s = x.mean(), x.std()
    return (x - m) / (s + 1e-6)
