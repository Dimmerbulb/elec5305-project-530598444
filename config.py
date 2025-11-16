import os
import random
import numpy as np

SAMPLE_RATE = 16000
EPS = 1e-8

N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

SEED = 42

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(ROOT_DIR, "raw_data")
CLEAN_RAW_DIR = os.path.join(RAW_DATA_DIR, "clean")
RIR_DIR = os.path.join(ROOT_DIR, "rir")

DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
CLEAN_DIR = os.path.join(DATASET_DIR, "clean")
REVERB_DIR = os.path.join(DATASET_DIR, "reverb")

BASELINE_DIR = os.path.join(ROOT_DIR, "baselines")
SS_DIR = os.path.join(BASELINE_DIR, "ss")
WIENER_DIR = os.path.join(BASELINE_DIR, "wiener")
WPE_DIR = os.path.join(BASELINE_DIR, "wpe")
TINY_UNET_DIR = os.path.join(BASELINE_DIR, "tiny_unet")

RESULT_DIR = os.path.join(ROOT_DIR, "results")
METRIC_DIR = os.path.join(RESULT_DIR, "metrics")
DEMO_DIR = os.path.join(RESULT_DIR, "demos")

MODEL_DIR = os.path.join(ROOT_DIR, "model_ckpt")
TINY_UNET_CKPT = os.path.join(MODEL_DIR, "tiny_unet.pth")

MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")

for d in [
    RAW_DATA_DIR,
    CLEAN_RAW_DIR,
    RIR_DIR,
    DATASET_DIR,
    CLEAN_DIR,
    REVERB_DIR,
    BASELINE_DIR,
    SS_DIR,
    WIENER_DIR,
    WPE_DIR,
    TINY_UNET_DIR,
    RESULT_DIR,
    METRIC_DIR,
    DEMO_DIR,
    MODEL_DIR,
]:
    os.makedirs(d, exist_ok=True)


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
