import numpy as np
from config import EPS
from utils.stft import stft, istft


def spectral_subtraction(y, beta: float = 1.0, gamma: float = 0.01, noise_frames: int = 6):
    Y, _ = stft(y)
    mag = np.abs(Y)
    phase = np.angle(Y)

    noise_mag = np.mean(mag[:, : max(1, noise_frames)], axis=1, keepdims=True)

    enhanced_mag = np.maximum(mag - beta * noise_mag, gamma * noise_mag)
    Z = enhanced_mag * np.exp(1j * phase)
    z = istft(Z)
    if len(z) > len(y):
        z = z[: len(y)]
    return z.astype(np.float32)
