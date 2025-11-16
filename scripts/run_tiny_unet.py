import os
import csv
import numpy as np

import torch

from config import ROOT_DIR, MANIFEST_PATH, TINY_UNET_CKPT, TINY_UNET_DIR
from utils.audio_io import load_wav, save_wav
from utils.stft import stft, istft
from models import TinyUNet


def _load_rows(split: str):
    rows = []
    if not os.path.exists(MANIFEST_PATH):
        return rows
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split", "") == split:
                rows.append(row)
    return rows


def main():
    if not os.path.exists(TINY_UNET_CKPT):
        print(f"[run_tiny_unet] No checkpoint at {TINY_UNET_CKPT}; run train_tiny_unet first.")
        return

    test_rows = _load_rows("test")
    if not test_rows:
        print("[run_tiny_unet] No test rows in manifest; falling back to all rows.")
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            test_rows = list(reader)
        if not test_rows:
            print("[run_tiny_unet] Manifest is empty; nothing to run.")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_tiny_unet] Using device: {device}")

    model = TinyUNet()
    model.load_state_dict(torch.load(TINY_UNET_CKPT, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(TINY_UNET_DIR, exist_ok=True)

    for row in test_rows:
        rev_path = row["rev_path"]
        if not os.path.isabs(rev_path):
            rev_path = os.path.join(ROOT_DIR, rev_path)
        wav, sr = load_wav(rev_path)

        Y, _ = stft(wav)
        mag = np.abs(Y)
        phase = np.angle(Y)

        F_bins, T_frames = mag.shape

        F_pad = (4 - (F_bins % 4)) % 4
        T_pad = (4 - (T_frames % 4)) % 4
        mag_pad = np.pad(mag, ((0, F_pad), (0, T_pad)), mode="constant", constant_values=0.0)

        inp = torch.from_numpy(mag_pad.astype(np.float32))[None, None, :, :].to(device)
        with torch.no_grad():
            mask = model(inp)
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()
        mask = mask[:F_bins, :T_frames]

        enh_mag = mask * mag
        Z_enh = enh_mag * np.exp(1j * phase)
        enh = istft(Z_enh)
        enh = enh[: len(wav)]

        fname = row["utt_id"] + ".wav"
        out_path = os.path.join(TINY_UNET_DIR, fname)
        save_wav(out_path, enh, sr)
        print(f"[run_tiny_unet] Saved {out_path}")

    print("[run_tiny_unet] Done.")


if __name__ == "__main__":
    main()
