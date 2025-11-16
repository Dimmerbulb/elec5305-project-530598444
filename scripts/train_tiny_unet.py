import os
import csv
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import (
    ROOT_DIR,
    MANIFEST_PATH,
    SAMPLE_RATE,
    SEED,
    TINY_UNET_CKPT,
    set_global_seed,
)
from utils.audio_io import load_wav
from utils.stft import stft
from models import TinyUNet


class MagPairDataset(Dataset):
    def __init__(self, rows, window_frames: int = 256):
        self.rows = list(rows)
        self.window_frames = window_frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        clean_path = row["clean_path"]
        rev_path = row["rev_path"]
        if not os.path.isabs(clean_path):
            clean_path = os.path.join(ROOT_DIR, clean_path)
        if not os.path.isabs(rev_path):
            rev_path = os.path.join(ROOT_DIR, rev_path)

        clean, _ = load_wav(clean_path, target_sr=SAMPLE_RATE)
        rev, _ = load_wav(rev_path, target_sr=SAMPLE_RATE)

        Y_rev, _ = stft(rev)
        Y_clean, _ = stft(clean)

        mag_rev = np.abs(Y_rev)
        mag_clean = np.abs(Y_clean)

        T = min(mag_rev.shape[1], mag_clean.shape[1])
        mag_rev = mag_rev[:, :T]
        mag_clean = mag_clean[:, :T]

        F_bins = mag_rev.shape[0]
        F_pad = (4 - (F_bins % 4)) % 4
        if F_pad:
            mag_rev = np.pad(mag_rev, ((0, F_pad), (0, 0)), mode="constant")
            mag_clean = np.pad(mag_clean, ((0, F_pad), (0, 0)), mode="constant")

        T_win = self.window_frames
        if T < T_win:
            pad_T = T_win - T
            mag_rev = np.pad(mag_rev, ((0, 0), (0, pad_T)), mode="constant")
            mag_clean = np.pad(mag_clean, ((0, 0), (0, pad_T)), mode="constant")
        else:
            start = np.random.randint(0, T - T_win + 1)
            mag_rev = mag_rev[:, start : start + T_win]
            mag_clean = mag_clean[:, start : start + T_win]

        mag_rev = mag_rev.astype(np.float32)
        mag_clean = mag_clean.astype(np.float32)

        noisy_mag = torch.from_numpy(mag_rev)[None, :, :]
        clean_mag = torch.from_numpy(mag_clean)[None, :, :]
        return noisy_mag, clean_mag


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
    set_global_seed(SEED)

    train_rows = _load_rows("train")
    valid_rows = _load_rows("valid")

    if not train_rows:
        print("[train_tiny_unet] No train rows in manifest; skip neural training.")
        return
    if not valid_rows:
        valid_rows = train_rows

    train_ds = MagPairDataset(train_rows)
    valid_ds = MagPairDataset(valid_rows)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=4, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_tiny_unet] Using device: {device}")

    model = TinyUNet()
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val = float("inf")
    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for noisy_mag, clean_mag in train_loader:
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)

            mask = model(noisy_mag)
            enh_mag = mask * noisy_mag
            loss = F.l1_loss(enh_mag, clean_mag)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)

        model.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for noisy_mag, clean_mag in valid_loader:
                noisy_mag = noisy_mag.to(device)
                clean_mag = clean_mag.to(device)
                mask = model(noisy_mag)
                enh_mag = mask * noisy_mag
                loss = F.l1_loss(enh_mag, clean_mag)
                val_total += loss.item()
                val_batches += 1

        val_loss = val_total / max(1, val_batches)
        print(
            f"[train_tiny_unet] Epoch {epoch}/{num_epochs} - "
            f"train L1: {train_loss:.4f}, val L1: {val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(TINY_UNET_CKPT), exist_ok=True)
            torch.save(model.state_dict(), TINY_UNET_CKPT)
            print(f"[train_tiny_unet] Saved best model to {TINY_UNET_CKPT}")

    print("[train_tiny_unet] Training finished.")


if __name__ == "__main__":
    main()
