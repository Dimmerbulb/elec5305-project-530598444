import soundfile as sf
import numpy as np
import math
from scipy.signal import resample_poly
from config import SAMPLE_RATE


def load_wav(path, target_sr: int = SAMPLE_RATE):
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        audio = resample_poly(audio, up, down)
        sr = target_sr
    audio = audio.astype(np.float32)
    return audio, sr


def save_wav(path, audio, sr: int = SAMPLE_RATE):
    audio = np.asarray(audio, dtype=np.float32)
    sf.write(path, audio, sr)
