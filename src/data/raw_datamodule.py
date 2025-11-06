"""Lightning DataModule for raw EEG signal training."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .raw_eeg_dataset import RawEEGDataset, raw_eeg_collate_fn


class RawEEGDataModule(pl.LightningDataModule):
    """DataModule providing raw EEG label windows for baseline models."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        data_cfg = cfg.data

        self.metadata_csv = Path(data_cfg.metadata_csv)
        self.target_mode = data_cfg.target_mode
        self.vote_keys = [str(k) for k in data_cfg.vote_keys]

        loader_cfg = data_cfg.loader
        self.batch_size = int(loader_cfg.batch_size)
        self.shuffle = bool(loader_cfg.shuffle)
        self.num_workers = int(loader_cfg.num_workers)
        self.pin_memory = bool(loader_cfg.pin_memory)
        self.persistent_workers = bool(getattr(loader_cfg, "persistent_workers", False) and self.num_workers > 0)

        splits_cfg = data_cfg.splits
        self.split_ratios = list(splits_cfg.split_ratios)
        self.split_shuffle = bool(splits_cfg.shuffle)

        raw_cfg = getattr(data_cfg, "raw_eeg", None)
        if raw_cfg is None:
            raise KeyError("Configuration requires 'data.raw_eeg' section for raw EEG input.")

        self.raw_base_dir = Path(raw_cfg.get("base_dir", "data/raw"))
        self.raw_split = str(raw_cfg.get("split", "train"))
        self.sampling_rate = int(raw_cfg.get("sampling_rate"))
        self.label_window_sec = float(raw_cfg.get("label_window_sec"))
        self.context_window_sec = float(raw_cfg.get("context_window_sec", self.label_window_sec))
        self.normalize = bool(raw_cfg.get("normalize", True))

        label_to_index_cfg = raw_cfg.get("label_to_index", None)
        if label_to_index_cfg is not None:
            mapping = OmegaConf.to_container(label_to_index_cfg, resolve=True)  # type: ignore[arg-type]
            self.label_to_index = {str(k): int(v) for k, v in mapping.items()}
        else:
            self.label_to_index = None

        self.metadata_df: Optional[pd.DataFrame] = None
        self.label_metadata: Optional[Dict[str, Dict[str, object]]] = None

        self.train_dataset: Optional[RawEEGDataset] = None
        self.val_dataset: Optional[RawEEGDataset] = None
        self.test_dataset: Optional[RawEEGDataset] = None

    def prepare_data(self) -> None:
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found at {self.metadata_csv}")

        eeg_dir = self.raw_base_dir / f"{self.raw_split}_eegs"
        if not eeg_dir.exists():
            raise FileNotFoundError(f"Raw EEG directory not found at {eeg_dir}")

    def setup(self, stage: Optional[str] = None) -> None:
        self._ensure_metadata_ready()

        if stage in (None, "fit") and self.train_dataset is None:
            self._initialise_splits()

        if stage in (None, "validate") and self.val_dataset is None:
            self._initialise_splits(include_train=False)

        if stage in ("test", "predict") and self.test_dataset is None:
            self._initialise_splits(include_train=False)

    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.metadata_csv)
        df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
        df["patient_id"] = df["patient_id"].astype(str)
        df["label_id"] = df["label_id"].astype(str)
        return df

    def _ensure_metadata_ready(self) -> None:
        if self.metadata_df is None:
            self.metadata_df = self._load_metadata()
        if self.label_metadata is None:
            assert self.metadata_df is not None
            self.label_metadata = self._build_label_metadata(self.metadata_df)

    def patient_ids(self) -> List[str]:
        self._ensure_metadata_ready()
        assert self.metadata_df is not None
        return self.metadata_df["patient_id"].unique().tolist()

    def dataset_for_patients(self, patient_ids: Sequence[str]) -> RawEEGDataset:
        self._ensure_metadata_ready()
        return self._build_dataset(patient_ids)

    def dataloader(self, dataset: RawEEGDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=raw_eeg_collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def _build_label_metadata(self, df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
        metadata: Dict[str, Dict[str, object]] = {}
        required_columns = {"label_id", *self.vote_keys}
        missing = required_columns - set(df.columns)
        if missing:
            raise KeyError(f"Metadata missing columns required for votes: {missing}")

        for _, row in df.iterrows():
            label_key = str(row["label_id"])
            votes = {k: float(row[k]) for k in self.vote_keys}
            metadata[label_key] = {
                "votes": votes,
                "expert_consensus": row.get("expert_consensus"),
                "confidence": float(row.get("confidence", 1.0)),
            }
        return metadata

    def _initialise_splits(self, include_train: bool = True) -> None:
        assert self.metadata_df is not None
        assert self.label_metadata is not None

        patient_ids = self.metadata_df["patient_id"].unique().tolist()
        if self.split_shuffle:
            random.Random().shuffle(patient_ids)

        ratios = self._normalise_ratios(self.split_ratios)
        n_patients = len(patient_ids)
        train_end = int(round(ratios[0] * n_patients))
        val_end = train_end + int(round(ratios[1] * n_patients))

        train_ids = patient_ids[:train_end]
        val_ids = patient_ids[train_end:val_end]
        test_ids = patient_ids[val_end:]

        if not val_ids:
            val_ids = test_ids[:1]
            test_ids = test_ids[1:]
        if not test_ids:
            test_ids = val_ids[-1:]
            val_ids = val_ids[:-1]

        if include_train:
            self.train_dataset = self.dataset_for_patients(train_ids)
        self.val_dataset = self.dataset_for_patients(val_ids)
        self.test_dataset = self.dataset_for_patients(test_ids)

    def _build_dataset(self, patient_ids: Sequence[str]) -> RawEEGDataset:
        assert self.metadata_df is not None
        assert self.label_metadata is not None

        filter_mask = self.metadata_df["patient_id"].isin(patient_ids)
        subset = self.metadata_df.loc[filter_mask].copy()

        records: List[Dict[str, object]] = []
        for _, row in subset.iterrows():
            if pd.isna(row.get("eeg_id")) or pd.isna(row.get("eeg_label_offset_seconds")):
                continue
            records.append(
                {
                    "patient_id": row["patient_id"],
                    "label_id": row["label_id"],
                    "eeg_id": int(row["eeg_id"]),
                    "offset_seconds": float(row["eeg_label_offset_seconds"]),
                }
            )

        return RawEEGDataset(
            records=records,
            raw_base_dir=self.raw_base_dir,
            split=self.raw_split,
            sampling_rate=self.sampling_rate,
            label_window_sec=self.label_window_sec,
            context_window_sec=self.context_window_sec,
            normalize=self.normalize,
            label_metadata=self.label_metadata,
            vote_keys=self.vote_keys,
            target_mode=self.target_mode,
            label_to_index=self.label_to_index,
        )

    @staticmethod
    def _normalise_ratios(ratios: Sequence[float]) -> List[float]:
        total = float(sum(ratios))
        if total <= 0:
            raise ValueError("Split ratios must sum to a positive value.")
        return [float(r) / total for r in ratios]

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return self.dataloader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return self.dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return self.dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


__all__ = ["RawEEGDataModule"]
