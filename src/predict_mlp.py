"""Generate Kaggle submission from a trained EEG MLP checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.lightning_trainer.mlp_lightning_module import EEGMLPLightningModule

DEFAULT_VOTE_KEYS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict submission from an EEG MLP checkpoint.")
    p.add_argument("--config", default="configs/training_mlp.yaml", help="Config path used for training/inference.")
    p.add_argument("--checkpoint", required=True, help="Path to the Lightning checkpoint (.ckpt).")
    p.add_argument("--output", default=None, help="Output CSV path (defaults to stdout preview only).")
    p.add_argument("--batch_size", type=int, default=128, help="Batch size for batched inference.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    vote_keys = list(getattr(cfg.data, "vote_keys", DEFAULT_VOTE_KEYS))
    if not vote_keys:
        raise ValueError("Configuration must define data.vote_keys for submission columns.")

    print(f"Loading checkpoint from {args.checkpoint}")
    model = EEGMLPLightningModule.load_from_checkpoint(args.checkpoint, config=cfg, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    test_csv_path = Path(getattr(cfg.data, "metadata_csv", "data/raw/test.csv"))
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test metadata CSV not found at {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    required_cols = {"label_id", "eeg_id", "eeg_label_offset_seconds"}
    missing = required_cols - set(test_df.columns)
    if missing:
        raise KeyError(f"{test_csv_path} missing required columns: {missing}")

    test_df = test_df.copy()
    test_df["label_id"] = test_df["label_id"].astype(str)
    test_df["eeg_id"] = test_df["eeg_id"].astype(int)
    test_df["eeg_label_offset_seconds"] = test_df["eeg_label_offset_seconds"].fillna(0.0).astype(float)

    eeg_root = Path(getattr(cfg.data.raw_eeg, "base_dir", "data/raw")) / f"{getattr(cfg.data.raw_eeg, 'split', 'test')}_eegs"
    if not eeg_root.exists():
        raise FileNotFoundError(f"EEG directory not found at {eeg_root}")

    sr = int(getattr(cfg.data.raw_eeg, "sampling_rate", 200))
    label_len = int(round(float(getattr(cfg.data.raw_eeg, "label_window_sec", 10.0)) * sr))
    context_len = int(round(float(getattr(cfg.data.raw_eeg, "context_window_sec", 50.0)) * sr))
    normalize = bool(getattr(cfg.data.raw_eeg, "normalize", True))

    eeg_cache: Dict[int, np.ndarray] = {}

    def load_full_eeg(eeg_id: int) -> np.ndarray:
        cached = eeg_cache.get(eeg_id)
        if cached is not None:
            return cached
        path = eeg_root / f"{eeg_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"EEG parquet not found at {path}")
        df = pd.read_parquet(path)
        if "EKG" in df.columns:
            df = df.drop(columns=["EKG"])
        arr = df.to_numpy(copy=False)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        eeg_cache[eeg_id] = arr
        return arr

    def extract_window(eeg_id: int, offset_seconds: float) -> torch.Tensor:
        eeg_array = load_full_eeg(eeg_id)
        total_samples = eeg_array.shape[0]
        if total_samples == 0:
            raise ValueError(f"EEG {eeg_id} is empty.")

        offset = int(round(offset_seconds * sr))
        offset = max(0, min(offset, total_samples - 1))

        context_start = max(0, offset)
        context_end = min(context_start + context_len, total_samples)
        if context_end - context_start < label_len:
            context_start = max(0, context_end - label_len)
        if context_end <= context_start:
            context_start = max(0, total_samples - label_len)
            context_end = total_samples

        actual_context = context_end - context_start
        label_start = context_start + max(0, (actual_context - label_len) // 2)
        label_end = label_start + label_len
        if label_end > context_end:
            label_start = max(context_start, context_end - label_len)
            label_end = context_end

        window = eeg_array[label_start:label_end]
        if window.shape[0] < label_len:
            pad = label_len - window.shape[0]
            window = np.pad(window, ((0, pad), (0, 0)), mode="constant", constant_values=0.0)

        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        window = window.T  # channels x time

        if normalize and window.size > 0:
            mean = window.mean(axis=-1, keepdims=True)
            std = window.std(axis=-1, keepdims=True)
            std[std < 1e-6] = 1.0
            window = (window - mean) / std

        return torch.from_numpy(window.astype(np.float32, copy=False))

    processed_label_ids: List[str] = []
    all_probs: List[np.ndarray] = []
    missing_label_ids: List[str] = []

    rows = test_df[["label_id", "eeg_id", "eeg_label_offset_seconds"]].to_dict("records")
    total_rows = len(rows)
    print(f"Running inference for {total_rows} label windows ...")

    with torch.no_grad():
        for start in range(0, total_rows, args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            tensors: List[torch.Tensor] = []
            label_ids: List[str] = []

            for row in batch_rows:
                label_id = row["label_id"]
                eeg_id = int(row["eeg_id"])
                offset = float(row["eeg_label_offset_seconds"])
                try:
                    tensor = extract_window(eeg_id, offset)
                except Exception as exc:
                    print(f"Warning: skipping label_id={label_id} (eeg_id={eeg_id}): {exc}")
                    missing_label_ids.append(label_id)
                    continue
                tensors.append(tensor)
                label_ids.append(label_id)

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors, dim=0).to(device)
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

            processed_label_ids.extend(label_ids)
            all_probs.append(probs)

            if (start // args.batch_size + 1) % 10 == 0:
                processed = len(processed_label_ids) + len(missing_label_ids)
                print(f"Processed {processed}/{total_rows} samples")

    if all_probs:
        concatenated = np.vstack(all_probs)
    else:
        concatenated = np.zeros((0, len(vote_keys)), dtype=np.float32)

    print(f"Generated predictions for {len(processed_label_ids)} samples; {len(missing_label_ids)} missing.")

    preds_df = pd.DataFrame({"label_id": processed_label_ids})
    for i, vote_key in enumerate(vote_keys):
        preds_df[vote_key] = concatenated[:, i] if len(concatenated) else []

    sub = test_df[["label_id"]].merge(preds_df, on="label_id", how="left")
    if len(sub):
        sub[vote_keys] = sub[vote_keys].fillna(1.0 / len(vote_keys))

    print("\nSubmission preview:")
    print(sub.head())
    print(f"\nShape: {sub.shape}")
    print(f"Columns: {list(sub.columns)}")

    if args.output:
        sub.to_csv(args.output, index=False)
        print(f"\nSaved submission to {args.output}")
        return None

    print("\nNo output path provided; returning DataFrame.")
    return sub


if __name__ == "__main__":
    main()
