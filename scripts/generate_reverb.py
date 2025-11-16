import os
import csv
import numpy as np
from glob import glob
from scipy.signal import fftconvolve

from config import (
    ROOT_DIR,
    CLEAN_DIR,
    REVERB_DIR,
    RIR_DIR,
    MANIFEST_PATH,
    SAMPLE_RATE,
    SEED,
    set_global_seed,
)
from utils.audio_io import load_wav, save_wav


def _assign_splits(rir_paths):
    rir_ids = [os.path.splitext(os.path.basename(p))[0] for p in rir_paths]
    n = len(rir_ids)
    if n == 0:
        return {}

    order = list(range(n))
    rng = np.random.RandomState(SEED)
    rng.shuffle(order)

    n_train = max(1, int(round(0.6 * n)))
    n_valid = max(1, int(round(0.2 * n))) if n >= 3 else max(0, n - n_train)

    split_map = {}
    for rank, idx in enumerate(order):
        rid = rir_ids[idx]
        if rank < n_train:
            split = "train"
        elif rank < n_train + n_valid:
            split = "valid"
        else:
            split = "test"
        split_map[rid] = split
    return split_map


def _parse_speaker_id(utt_id: str) -> str:
    base = utt_id
    if "_" in base:
        return base.split("_")[0]
    return "spk0"


def main():
    set_global_seed(SEED)

    clean_wavs = sorted(glob(os.path.join(CLEAN_DIR, "*.wav")))
    if not clean_wavs:
        print(f"[generate_reverb] No clean wavs in {CLEAN_DIR}. Nothing to do.")
        return

    rir_wavs = sorted(glob(os.path.join(RIR_DIR, "*.wav")))
    if not rir_wavs:
        print(f"[generate_reverb] No RIR wavs found in {RIR_DIR}. Please add some first.")
        return

    os.makedirs(REVERB_DIR, exist_ok=True)

    split_map = _assign_splits(rir_wavs)
    print("[generate_reverb] RIR split counts:")
    counts = {"train": 0, "valid": 0, "test": 0}
    for rid, split in split_map.items():
        counts[split] += 1
    for split, c in counts.items():
        print(f"  {split}: {c} rooms")

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "split", "clean_path", "rev_path", "rir_id", "speaker_id"])

        for i, clean_path in enumerate(clean_wavs):
            rir_path = rir_wavs[i % len(rir_wavs)]
            rir_id = os.path.splitext(os.path.basename(rir_path))[0]
            split = split_map[rir_id]

            clean, _ = load_wav(clean_path, target_sr=SAMPLE_RATE)
            rir, _ = load_wav(rir_path, target_sr=SAMPLE_RATE)

            if rir.ndim > 1:
                rir = rir.mean(axis=1)

            rev = fftconvolve(clean, rir, mode="full")
            rev = rev[: len(clean)]
            max_abs = np.max(np.abs(rev)) + 1e-8
            rev = (rev / max_abs).astype(np.float32)

            fname = os.path.basename(clean_path)
            rev_path = os.path.join(REVERB_DIR, fname)
            save_wav(rev_path, rev, SAMPLE_RATE)

            utt_id = os.path.splitext(fname)[0]
            speaker_id = _parse_speaker_id(utt_id)

            writer.writerow(
                [
                    utt_id,
                    split,
                    os.path.relpath(clean_path, ROOT_DIR),
                    os.path.relpath(rev_path, ROOT_DIR),
                    rir_id,
                    speaker_id,
                ]
            )

            print(f"[generate_reverb] {fname} <- RIR {rir_id} ({split})")

    print(f"[generate_reverb] Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
