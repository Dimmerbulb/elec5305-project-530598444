import numpy as np
from pystoi.stoi import stoi
from config import SAMPLE_RATE, EPS


def si_sdr(s, s_hat, eps: float = EPS) -> float:
    s = np.asarray(s, dtype=np.float64)
    s_hat = np.asarray(s_hat, dtype=np.float64)
    L = min(len(s), len(s_hat))
    if L == 0:
        return float("-inf")
    s = s[:L]
    s_hat = s_hat[:L]
    s = s - np.mean(s)
    s_hat = s_hat - np.mean(s_hat)
    alpha = np.dot(s_hat, s) / (np.dot(s, s) + eps)
    s_target = alpha * s
    e = s_hat - s_target
    num = np.sum(s_target ** 2)
    den = np.sum(e ** 2) + eps
    return float(10 * np.log10(num / den + eps))


def stoi_score(clean, enhanced) -> float:
    clean = np.asarray(clean, dtype=np.float32)
    enhanced = np.asarray(enhanced, dtype=np.float32)
    L = min(len(clean), len(enhanced))
    if L <= 0:
        return 0.0
    return float(stoi(clean[:L], enhanced[:L], SAMPLE_RATE, extended=False))
