import numpy as np
from config import EPS
from utils.stft import stft, istft


def wiener_filter(y, alpha: float = 0.98, noise_frames: int = 6):
    Y, _ = stft(y)
    mag = np.abs(Y)
    phase = np.angle(Y)

    noise_psd = np.mean(mag[:, : max(1, noise_frames)] ** 2, axis=1, keepdims=True)

    S_psd = np.zeros_like(mag)
    S_psd[:, 0:1] = mag[:, 0:1] ** 2
    for t in range(1, mag.shape[1]):
        S_psd[:, t:t + 1] = alpha * S_psd[:, t - 1:t] + (1.0 - alpha) * mag[:, t:t + 1] ** 2

    G = S_psd / (S_psd + noise_psd + EPS)
    enhanced_mag = G * mag
    Z = enhanced_mag * np.exp(1j * phase)
    z = istft(Z)
    if len(z) > len(y):
        z = z[: len(y)]
    return z.astype(np.float32)
