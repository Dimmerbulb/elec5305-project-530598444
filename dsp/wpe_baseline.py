import numpy as np

try:
    from nara_wpe import np_wpe as wpe
    from nara_wpe.utils import stft as wpe_stft, istft as wpe_istft
    HAS_WPE = True
except ImportError:
    wpe = None
    wpe_stft = None
    wpe_istft = None
    HAS_WPE = False


def apply_wpe(y, taps: int = 10, delay: int = 3, iterations: int = 3):
    if not HAS_WPE:
        raise RuntimeError(
            " "
        )

    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[None, :]

    Y_tf = wpe_stft(y)
    Y = np.transpose(Y_tf, (2, 0, 1))

    Z = wpe.wpe(Y, taps=taps, delay=delay, iterations=iterations)

    Z_tf = np.transpose(Z, (1, 2, 0))
    z = wpe_istft(Z_tf)
    if z.ndim > 1:
        z = z[0]
    return z.astype(np.float32)
