from scipy.signal import stft as sp_stft, istft as sp_istft
from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH


def stft(x, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, win_length: int = WIN_LENGTH):
    _, _, Zxx = sp_stft(
        x,
        fs=SAMPLE_RATE,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        window="hann",
        boundary=None,
        padded=False,
    )
    return Zxx, None


def istft(Zxx, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, win_length: int = WIN_LENGTH):
    _, x = sp_istft(
        Zxx,
        fs=SAMPLE_RATE,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        window="hann",
        input_onesided=True,
        boundary=None,
    )
    return x
