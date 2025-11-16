import os
from glob import glob
from config import CLEAN_RAW_DIR, CLEAN_DIR
from utils.audio_io import load_wav, save_wav


def main():
    wavs = sorted(glob(os.path.join(CLEAN_RAW_DIR, "*.wav")))
    if not wavs:
        print(
            f"[prepare_clean] No wavs found in {CLEAN_RAW_DIR}. If you already have "
            f"dataset/clean ready, you can skip this step."
        )
        return

    os.makedirs(CLEAN_DIR, exist_ok=True)
    for src in wavs:
        audio, sr = load_wav(src)
        fname = os.path.basename(src)
        dst = os.path.join(CLEAN_DIR, fname)
        save_wav(dst, audio, sr)
        print(f"[prepare_clean] Saved clean: {dst}")

    print("[prepare_clean] Done.")


if __name__ == "__main__":
    main()
