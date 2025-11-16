import os
import csv
import shutil
import random

from config import (
    ROOT_DIR,
    MANIFEST_PATH,
    DEMO_DIR,
    REVERB_DIR,
    SS_DIR,
    WIENER_DIR,
    WPE_DIR,
    TINY_UNET_DIR,
    SEED,
)


def _load_test_rows():
    rows = []
    if not os.path.exists(MANIFEST_PATH):
        return rows
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split", "") == "test":
                rows.append(row)
    return rows


def main(num_examples: int = 10):
    rows = _load_test_rows()
    if not rows:
        print("[export_demos] No test rows in manifest; nothing to export.")
        return

    random.Random(SEED).shuffle(rows)
    rows = rows[:num_examples]

    os.makedirs(DEMO_DIR, exist_ok=True)

    for row in rows:
        utt_id = row["utt_id"]
        fname = utt_id + ".wav"
        subdir = os.path.join(DEMO_DIR, utt_id)
        os.makedirs(subdir, exist_ok=True)

        clean_path = os.path.join(ROOT_DIR, row["clean_path"])
        rev_path = os.path.join(ROOT_DIR, row["rev_path"])

        def safe_copy(src, tag):
            if os.path.exists(src):
                dst = os.path.join(subdir, f"{tag}.wav")
                shutil.copy2(src, dst)

        safe_copy(clean_path, "clean")
        safe_copy(rev_path, "rev")
        safe_copy(os.path.join(SS_DIR, fname), "ss")
        safe_copy(os.path.join(WIENER_DIR, fname), "wiener")
        safe_copy(os.path.join(WPE_DIR, fname), "wpe")
        safe_copy(os.path.join(TINY_UNET_DIR, fname), "tiny_unet")

        print(f"[export_demos] Exported demo for {utt_id} -> {subdir}")

    print(f"[export_demos] Demos ready under {DEMO_DIR}.")


if __name__ == "__main__":
    main()
