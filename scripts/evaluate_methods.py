import os
import csv
from glob import glob
from collections import defaultdict

from config import (
    CLEAN_DIR,
    REVERB_DIR,
    SS_DIR,
    WIENER_DIR,
    WPE_DIR,
    TINY_UNET_DIR,
    METRIC_DIR,
    MANIFEST_PATH,
)
from utils.audio_io import load_wav
from utils.metrics import si_sdr, stoi_score


def _load_split_map():
    if not os.path.exists(MANIFEST_PATH):
        return None
    mapping = {}
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["utt_id"]] = row.get("split", "unknown")
    return mapping


def eval_one_method(clean_dir, method_dir, method_name, split_map):
    clean_files = sorted(glob(os.path.join(clean_dir, "*.wav")))
    rows = []
    sdrs = []
    stois = []

    for cpath in clean_files:
        fname = os.path.basename(cpath)
        utt_id = os.path.splitext(fname)[0]
        mpath = os.path.join(method_dir, fname)
        if not os.path.exists(mpath):
            continue

        clean, _ = load_wav(cpath)
        enh, _ = load_wav(mpath)

        sdr = si_sdr(clean, enh)
        st = stoi_score(clean, enh)

        split = split_map.get(utt_id, "unknown") if split_map else "unknown"

        rows.append(
            {
                "utt": fname,
                "utt_id": utt_id,
                "split": split,
                "method": method_name,
                "si_sdr": sdr,
                "stoi": st,
            }
        )
        sdrs.append(sdr)
        stois.append(st)

    avg_sdr = sum(sdrs) / max(1, len(sdrs))
    avg_stoi = sum(stois) / max(1, len(stois))
    return avg_sdr, avg_stoi, rows


def main():
    os.makedirs(METRIC_DIR, exist_ok=True)
    split_map = _load_split_map()

    methods = {
        "noisy": REVERB_DIR,
        "ss": SS_DIR,
        "wiener": WIENER_DIR,
    }
    if os.path.isdir(WPE_DIR) and os.listdir(WPE_DIR):
        methods["wpe"] = WPE_DIR
    if os.path.isdir(TINY_UNET_DIR) and os.listdir(TINY_UNET_DIR):
        methods["tiny_unet"] = TINY_UNET_DIR

    all_rows = []

    print("=== 平均指标（SI-SDR / STOI） ===")
    for name, mdir in methods.items():
        avg_sdr, avg_stoi, rows = eval_one_method(CLEAN_DIR, mdir, name, split_map)
        all_rows.extend(rows)
        print(f"{name:<8}| SI-SDR: {avg_sdr:7.2f} dB | STOI: {avg_stoi:.3f}")

    if split_map and all_rows:
        stats = defaultdict(lambda: defaultdict(lambda: {"sdr": [], "stoi": []}))
        for row in all_rows:
            split = row["split"]
            method = row["method"]
            stats[split][method]["sdr"].append(row["si_sdr"])
            stats[split][method]["stoi"].append(row["stoi"])

        print("\n=== 按数据划分 (train/valid/test) 的平均指标 ===")
        for split in ["train", "valid", "test", "unknown"]:
            if split not in stats:
                continue
            print(f"[{split}]")
            for method, d in stats[split].items():
                if not d["sdr"]:
                    continue
                sdr = sum(d["sdr"]) / len(d["sdr"])
                st = sum(d["stoi"]) / len(d["stoi"])
                print(f"  {method:<8}| SI-SDR: {sdr:7.2f} dB | STOI: {st:.3f}")
            print()

    out_csv = os.path.join(METRIC_DIR, "metrics_sisdr_stoi.csv")
    fieldnames = ["utt", "utt_id", "split", "method", "si_sdr", "stoi"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\n详细结果已保存到 {out_csv}")


if __name__ == "__main__":
    main()
