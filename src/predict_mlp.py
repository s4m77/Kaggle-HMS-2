"""Generate Kaggle submission from a trained MLP checkpoint.

Usage:
  python -m src.predict_mlp --config configs/training_mlp.yaml --checkpoint path/to/ckpt.ckpt --output submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from src.models.lightning_model import HMSLightningModule


VOTE_KEYS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict submission from MLP checkpoint")
    p.add_argument("--config", default="configs/training_mlp.yaml", help="Training config path")
    p.add_argument("--checkpoint", required=True, help="Path to Lightning checkpoint")
    p.add_argument("--output", default=None, help="Output CSV path (optional)")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    
    # Load model from checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    model = HMSLightningModule.load_from_checkpoint(args.checkpoint, config=cfg, strict=False)
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Build simple test inputs directly from test.csv and test_eegs
    print("Building test inputs from data/raw/test.csv and data/raw/test_eegs ...")
    test_csv_path = Path("data/raw/test.csv")
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    if "eeg_id" not in test_df.columns:
        raise KeyError("test.csv missing required column 'eeg_id'")

    eeg_root = Path("data/raw/test_eegs")
    # Pull preprocessing parameters from config
    sr = int(getattr(cfg.data.raw_eeg, "sampling_rate", 200))
    label_len = int(round(float(getattr(cfg.data.raw_eeg, "label_window_sec", 10.0)) * sr))
    context_len = int(round(float(getattr(cfg.data.raw_eeg, "context_window_sec", 50.0)) * sr))
    do_norm = bool(getattr(cfg.data.raw_eeg, "normalize", True))
    have_ids: list[int] = []
    have_tensors: list[torch.Tensor] = []
    missing_ids: list[int] = []

    def load_eeg_parquet(eeg_id: int) -> torch.Tensor:
        path = eeg_root / f"{eeg_id}.parquet"
        df = pd.read_parquet(path)
        if "EKG" in df.columns:
            df = df.drop(columns=["EKG"])
        arr = df.to_numpy(copy=False)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        arr = arr.T  # channels x time

        # Center-crop to context_len then to label_len to match training shapes
        T = arr.shape[1]
        # If shorter than label window, pad to label_len
        if T < label_len:
            pad = label_len - T
            arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
            T = arr.shape[1]

        # Determine context segment
        if T >= context_len:
            c_start = max(0, (T - context_len) // 2)
            c_end = c_start + context_len
        else:
            c_start, c_end = 0, T
        # Now choose centered label window within context
        c_len = c_end - c_start
        l_start = c_start + max(0, (c_len - label_len) // 2)
        l_end = l_start + label_len
        if l_end > c_end:
            l_start = max(c_start, c_end - label_len)
            l_end = c_end
        arr = arr[:, l_start:l_end]

        # Per-channel normalization (match training RawEEGDataset)
        if do_norm and arr.size > 0:
            mean = arr.mean(axis=-1, keepdims=True)
            std = arr.std(axis=-1, keepdims=True)
            std[std < 1e-6] = 1.0
            arr = (arr - mean) / std

        return torch.from_numpy(arr)

    for eeg_id in test_df["eeg_id"].astype(int).tolist():
        try:
            ten = load_eeg_parquet(eeg_id)
            have_ids.append(eeg_id)
            have_tensors.append(ten)
        except Exception as e:
            print(f"Warning: could not load EEG {eeg_id}: {e}")
            missing_ids.append(eeg_id)

    print(f"Loaded {len(have_tensors)} / {len(test_df)} EEG files")

    # Batched inference
    all_eeg_ids: list[int] = []
    all_probs: list[np.ndarray] = []

    def chunks(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    with torch.no_grad():
        for bidx, (ids_chunk, ten_chunk) in enumerate(zip(chunks(have_ids, args.batch_size), chunks(have_tensors, args.batch_size))):
            x = torch.stack(ten_chunk, dim=0).to(device)
            outputs = model.predict_step({"eeg_signal": x, "target": None, "confidence": torch.ones(x.size(0), device=x.device)}, bidx)
            probs = outputs["probs"].detach().cpu().numpy()
            all_eeg_ids.extend(ids_chunk)
            all_probs.append(probs)
            if (bidx + 1) % 10 == 0:
                print(f"Processed {bidx + 1} batches")
    
    # Concatenate all predictions
    all_probs = np.vstack(all_probs) if all_probs else np.zeros((0, len(VOTE_KEYS)), dtype=np.float32)
    print(f"Generated predictions for {len(all_eeg_ids)} available samples; {len(missing_ids)} missing")

    # Build submission in test.csv order, fill missing with uniform
    preds_df = pd.DataFrame({"eeg_id": all_eeg_ids})
    for i, vote_key in enumerate(VOTE_KEYS):
        preds_df[vote_key] = all_probs[:, i] if len(all_probs) else []

    sub = test_df[["eeg_id"]].merge(preds_df, on="eeg_id", how="left")
    if len(sub) > 0:
        sub[VOTE_KEYS] = sub[VOTE_KEYS].fillna(1.0 / len(VOTE_KEYS))

    print("\nSubmission preview:")
    print(sub.head())
    print(f"\nShape: {sub.shape}")
    print(f"Columns: {list(sub.columns)}")
    
    # Save or return
    if args.output:
        sub.to_csv(args.output, index=False)
        print(f"\nSaved submission to {args.output}")
        return None
    else:
        print("\nReturning submission DataFrame (not writing to file)")
        return sub


if __name__ == "__main__":
    main()
