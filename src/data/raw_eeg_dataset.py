"""Dataset utilities for raw EEG signal windows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

EPS = 1e-6


class RawEEGDataset(Dataset):
    """Dataset loading raw EEG label windows on demand."""

    def __init__(
        self,
        records: Sequence[Dict[str, object]],
        *,
        raw_base_dir: Path,
        split: str,
        sampling_rate: int,
        label_window_sec: float,
        context_window_sec: float,
        normalize: bool,
        label_metadata: Dict[str, Dict[str, object]],
        vote_keys: Sequence[str],
        target_mode: Optional[str],
        label_to_index: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()

        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive.")
        if label_window_sec <= 0:
            raise ValueError("label_window_sec must be positive.")

        self.records = list(records)
        self.raw_base_dir = Path(raw_base_dir)
        self.split = split
        self.sampling_rate = int(sampling_rate)
        self.label_len = int(round(label_window_sec * self.sampling_rate))
        self.context_len = int(round(context_window_sec * self.sampling_rate))
        if self.label_len <= 0:
            raise ValueError("Computed label length must be positive.")
        if self.context_len <= 0:
            self.context_len = self.label_len
        self.normalize = normalize
        self.label_metadata = label_metadata
        self.vote_keys = [str(k) for k in vote_keys]
        self.target_mode = target_mode
        self.label_to_index = label_to_index

        self._eeg_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]

        patient_id = str(record["patient_id"])
        label_id = str(record["label_id"])
        eeg_id = int(record["eeg_id"])
        offset_seconds = float(record["offset_seconds"])

        signal = self._extract_signal(eeg_id=eeg_id, offset_seconds=offset_seconds)
        target = self._extract_target(label_id)

        meta = self.label_metadata.get(label_id, {}) if self.label_metadata else {}
        confidence = float(meta.get("confidence", 1.0)) if meta else 1.0

        example: Dict[str, object] = {
            "patient_id": patient_id,
            "label_id": label_id,
            "eeg_signal": signal,
            "target": target,
            "confidence": confidence,
        }
        return example

    def _load_eeg_array(self, eeg_id: int) -> np.ndarray:
        cached = self._eeg_cache.get(eeg_id)
        if cached is not None:
            return cached

        eeg_dir = self.raw_base_dir / f"{self.split}_eegs"
        path = eeg_dir / f"{eeg_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"EEG parquet not found at {path}")

        df = pd.read_parquet(path)
        if "EKG" in df.columns:
            df = df.drop(columns=["EKG"])

        array = df.to_numpy(copy=False)
        if array.dtype != np.float32:
            array = array.astype(np.float32, copy=False)

        self._eeg_cache[eeg_id] = array
        return array

    def _extract_signal(self, eeg_id: int, offset_seconds: float) -> torch.Tensor:
        eeg_array = self._load_eeg_array(eeg_id)
        total_samples, _ = eeg_array.shape
        if total_samples == 0:
            raise ValueError(f"EEG {eeg_id} contains no samples.")

        offset_samples = int(round(offset_seconds * self.sampling_rate))
        offset_samples = max(0, min(offset_samples, total_samples - 1))

        context_start = max(0, offset_samples)
        context_end = min(context_start + self.context_len, total_samples)
        if context_end - context_start < self.label_len:
            context_start = max(0, context_end - self.label_len)
        if context_end <= context_start:
            context_start = max(0, total_samples - self.label_len)
            context_end = total_samples

        actual_context_len = context_end - context_start
        label_start = context_start + max(0, (actual_context_len - self.label_len) // 2)
        label_end = label_start + self.label_len
        if label_end > context_end:
            label_start = max(context_start, context_end - self.label_len)
            label_end = context_end

        window = eeg_array[label_start:label_end]
        if window.shape[0] < self.label_len:
            pad = self.label_len - window.shape[0]
            window = np.pad(window, ((0, pad), (0, 0)), mode="constant", constant_values=0.0)

        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        window = window.T  # channels x time

        if self.normalize:
            mean = window.mean(axis=-1, keepdims=True)
            std = window.std(axis=-1, keepdims=True)
            std[std < EPS] = 1.0
            window = (window - mean) / std

        tensor = torch.from_numpy(window.astype(np.float32, copy=False))
        tensor = tensor.contiguous()
        return tensor

    def _extract_target(self, label_id: str) -> Optional[torch.Tensor]:
        if self.target_mode is None:
            return None

        meta = self.label_metadata.get(label_id)
        if meta is None:
            return None

        if self.target_mode == "votes":
            votes = meta.get("votes")
            if votes is None:
                return None
            vote_tensor = torch.tensor(
                [float(votes.get(k, 0.0)) for k in self.vote_keys],
                dtype=torch.float32,
            )
            total = vote_tensor.sum()
            if total > 0:
                vote_tensor = vote_tensor / total
            return vote_tensor

        if self.target_mode == "consensus":
            consensus = meta.get("expert_consensus")
            if consensus is None or self.label_to_index is None:
                return None
            idx = self.label_to_index.get(str(consensus))
            if idx is None:
                raise KeyError(f"Consensus label '{consensus}' missing in label_to_index mapping.")
            return torch.tensor(idx, dtype=torch.long)

        raise ValueError(f"Unsupported target_mode '{self.target_mode}'.")


def raw_eeg_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Collate raw EEG examples into a batch."""

    patient_ids = [str(item["patient_id"]) for item in batch]
    label_ids = [str(item["label_id"]) for item in batch]
    eeg_signals = torch.stack([item["eeg_signal"] for item in batch])
    confidences = torch.tensor([float(item.get("confidence", 1.0)) for item in batch], dtype=torch.float32)

    targets = [item["target"] for item in batch]
    stacked_targets = None
    if all(target is not None for target in targets):
        stacked_targets = torch.stack(targets)  # type: ignore[arg-type]

    return {
        "patient_ids": patient_ids,
        "label_ids": label_ids,
        "eeg_signal": eeg_signals,
        "eeg_graph": None,
        "spectrogram_graph": None,
        "target": stacked_targets,
        "confidence": confidences,
    }


__all__ = ["RawEEGDataset", "raw_eeg_collate_fn"]
