import os
from glob import glob

from config import REVERB_DIR, SS_DIR, WIENER_DIR, WPE_DIR
from utils.audio_io import load_wav, save_wav
from dsp import spectral_subtraction, wiener_filter, apply_wpe, HAS_WPE


def main():
    noisy_files = sorted(glob(os.path.join(REVERB_DIR, "*.wav")))
    if not noisy_files:
        print(f"[run_baselines] No reverberant wavs in {REVERB_DIR}. Run generate_reverb first.")
        return

    os.makedirs(SS_DIR, exist_ok=True)
    os.makedirs(WIENER_DIR, exist_ok=True)
    os.makedirs(WPE_DIR, exist_ok=True)

    printed_wpe_warning = False

    for path in noisy_files:
        wav, sr = load_wav(path)
        fname = os.path.basename(path)

        enh_ss = spectral_subtraction(wav)
        save_wav(os.path.join(SS_DIR, fname), enh_ss, sr)

        enh_wf = wiener_filter(wav)
        save_wav(os.path.join(WIENER_DIR, fname), enh_wf, sr)

        if HAS_WPE:
            try:
                enh_wpe = apply_wpe(wav)
                save_wav(os.path.join(WPE_DIR, fname), enh_wpe, sr)
            except Exception as e:
                if not printed_wpe_warning:
                    print(f"[run_baselines] WPE failed on some files: {e}. Skipping WPE.")
                    printed_wpe_warning = True
        else:
            if not printed_wpe_warning:
                print(
                    "[run_baselines] nara_wpe not installed; skipping WPE baseline. "
                    "Install with 'pip install nara_wpe' if you want WPE."
                )
                printed_wpe_warning = True

        print(f"[run_baselines] Processed {fname}")

    print("[run_baselines] Done.")


if __name__ == "__main__":
    main()
