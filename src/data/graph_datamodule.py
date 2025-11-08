"""PyTorch Lightning DataModule for HMS dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import multiprocessing as mp
from sklearn.model_selection import StratifiedGroupKFold

from src.data.graph_dataset import HMSDataset, collate_graphs


class HMSDataModule(LightningDataModule):
    """Lightning DataModule for HMS brain activity classification.
    
    Implements StratifiedGroupKFold cross-validation to:
    - Split by patient_id (no patient in both train and val)
    - Stratify by total_evaluators bins (balanced quality distribution)
    - Support K-fold training for robust evaluation
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing preprocessed patient files
    train_csv : str or Path
        Path to train_unique.csv with metadata
    batch_size : int
        Batch size for training
    n_folds : int
        Number of folds for cross-validation
    current_fold : int
        Which fold to use for validation (0 to n_folds-1)
    stratify_by_class : bool
        Whether to stratify by consensus class (recommended for vote prediction)
    stratify_by_evaluators : bool
        Whether to also stratify by evaluator quality bins (dual stratification)
    evaluator_bins : List[int]
        Bin edges for total_evaluators stratification (e.g., [0, 5, 10, 15, 20, 999])
    min_evaluators : int
        Minimum total_evaluators to include (for quality filtering)
    num_workers : int
        Number of workers for DataLoader
    pin_memory : bool
        Whether to pin memory in DataLoader
    shuffle_seed : int
        Random seed for fold splitting
    """
    
    def __init__(
        self,
        data_dir: str | Path = "data/processed",
        train_csv: str | Path = "data/raw/train_unique.csv",
        batch_size: int = 32,
        n_folds: int = 5,
        current_fold: int = 0,
        stratify_by_class: bool = True,
        stratify_by_evaluators: bool = False,
        evaluator_bins: List[int] = [0, 5, 10, 15, 20, 999],
        min_evaluators: int = 0,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        shuffle_seed: int = 42,
        preload_patients: bool = False,
        use_cache_server: bool = False,
        cache_host: str = "127.0.0.1",
        cache_port: int = 50000,
        cache_authkey: str = "hms-cache",
    ) -> None:
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.train_csv = Path(train_csv)
        self.batch_size = batch_size
        self.n_folds = n_folds
        self.current_fold = current_fold
        self.stratify_by_class = stratify_by_class
        self.stratify_by_evaluators = stratify_by_evaluators
        self.evaluator_bins = evaluator_bins
        self.min_evaluators = min_evaluators
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.shuffle_seed = shuffle_seed
        self.preload_patients = preload_patients
        self.use_cache_server = use_cache_server
        self.cache_host = cache_host
        self.cache_port = cache_port
        self.cache_authkey = cache_authkey
        # Track setup stages to avoid double-initialization when Trainer also calls setup
        self._setup_done: set[str] = set()
        
        # Validate inputs
        assert 0 <= current_fold < n_folds, \
            f"current_fold must be in [0, {n_folds-1}], got {current_fold}"
        
        # Will be set in setup()
        self.train_dataset: Optional[HMSDataset] = None
        self.val_dataset: Optional[HMSDataset] = None
        self.class_weights: Optional[torch.Tensor] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage.
        
        Parameters
        ----------
        stage : str, optional
            Either 'fit', 'validate', 'test', or 'predict'
        """
        # Map stage to a key and make idempotent
        stage_key = 'fit' if stage in (None, 'fit') else stage
        if stage_key in self._setup_done:
            return

        # Load metadata CSV
        self.metadata_df = pd.read_csv(self.train_csv)
        
        # Strip whitespace from column names and string columns
        self.metadata_df.columns = self.metadata_df.columns.str.strip()
        if 'expert_consensus' in self.metadata_df.columns:
            self.metadata_df['expert_consensus'] = self.metadata_df['expert_consensus'].str.strip()
        
        # Calculate total_evaluators
        vote_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
        self.metadata_df['total_evaluators'] = self.metadata_df[vote_cols].sum(axis=1)
        
        # Filter to only include patients with processed graph files
        all_patients = set(self.metadata_df['patient_id'].unique())
        processed_files = list(self.data_dir.glob("patient_*.pt"))
        processed_patients = {int(f.stem.split('_')[1]) for f in processed_files}
        
        missing_patients = all_patients - processed_patients
        if missing_patients:
            n_before = len(self.metadata_df)
            self.metadata_df = self.metadata_df[
                self.metadata_df['patient_id'].isin(processed_patients)
            ].reset_index(drop=True)
            n_after = len(self.metadata_df)
            print(f"⚠ Filtered to only processed patients: {len(all_patients)} → {len(processed_patients)} patients")
            print(f"  Samples: {n_before} → {n_after}")
            print(f"  Missing {len(missing_patients)} patients (preprocessing in progress)")
        
        # Filter by minimum evaluators (quality control)
        if self.min_evaluators > 0:
            n_before = len(self.metadata_df)
            self.metadata_df = self.metadata_df[
                self.metadata_df['total_evaluators'] >= self.min_evaluators
            ].reset_index(drop=True)
            n_after = len(self.metadata_df)
            print(f"Filtered by min_evaluators={self.min_evaluators}: {n_before} → {n_after} samples")
        
        # Determine stratification variable
        if self.stratify_by_class and self.stratify_by_evaluators:
            # Dual stratification: combine class + evaluator bin into single stratification key
            # Create evaluator bins
            self.metadata_df['evaluator_bin'] = pd.cut(
                self.metadata_df['total_evaluators'],
                bins=self.evaluator_bins,
                labels=False,
                include_lowest=True
            )
            # Combine class and evaluator bin (e.g., "class_0_bin_2")
            stratify_var = (
                self.metadata_df['expert_consensus'].astype(str) + '_' + 
                self.metadata_df['evaluator_bin'].astype(str)
            )
            print(f"Stratifying by consensus class AND evaluator bins")
        elif self.stratify_by_class:
            # Stratify by consensus class only
            stratify_var = self.metadata_df['expert_consensus']
            print(f"Stratifying by consensus class only")
        elif self.stratify_by_evaluators:
            # Stratify by evaluator bins only
            self.metadata_df['evaluator_bin'] = pd.cut(
                self.metadata_df['total_evaluators'],
                bins=self.evaluator_bins,
                labels=False,
                include_lowest=True
            )
            stratify_var = self.metadata_df['evaluator_bin']
            print(f"Stratifying by evaluator bins only")
        else:
            # No stratification, use constant value
            stratify_var = pd.Series([0] * len(self.metadata_df))
            print(f"No stratification (patient-based GroupKFold only)")
        
        # Group by patient_id
        patient_groups = self.metadata_df.groupby('patient_id').ngroup()
        
        # Perform StratifiedGroupKFold split
        skf = StratifiedGroupKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.shuffle_seed
        )
        
        # Assign fold to each sample
        self.metadata_df['fold'] = -1
        for fold, (_, val_idx) in enumerate(skf.split(
            X=self.metadata_df,
            y=stratify_var,
            groups=patient_groups
        )):
            self.metadata_df.loc[val_idx, 'fold'] = fold
        
        # Create train and validation datasets
        # Optional remote cache connection
        remote_cache = None
        if self.use_cache_server:
            try:
                from multiprocessing.managers import BaseManager
                class _ClientManager(BaseManager):
                    pass
                _ClientManager.register('get_cache')
                mgr = _ClientManager(address=(self.cache_host, int(self.cache_port)), authkey=self.cache_authkey.encode('utf-8'))
                mgr.connect()
                remote_cache = mgr.get_cache()
                print(f"Connected to CacheServer at {self.cache_host}:{self.cache_port}")
            except Exception as e:
                print(f"Warning: Could not connect to CacheServer: {e}. Continuing without remote cache.")

        if stage == "fit" or stage is None:
            train_df = self.metadata_df[
                self.metadata_df['fold'] != self.current_fold
            ].reset_index(drop=True)
            
            val_df = self.metadata_df[
                self.metadata_df['fold'] == self.current_fold
            ].reset_index(drop=True)
            
            if self.train_dataset is None:
                self.train_dataset = HMSDataset(
                    data_dir=self.data_dir,
                    metadata_df=train_df,
                    is_train=True,
                    preload_patients=self.preload_patients,
                    remote_cache=remote_cache,
                )
            
            if self.val_dataset is None:
                self.val_dataset = HMSDataset(
                    data_dir=self.data_dir,
                    metadata_df=val_df,
                    is_train=False,
                    preload_patients=self.preload_patients,
                    remote_cache=remote_cache,
                )
            
            # Compute class weights from training set
            self.class_weights = self.train_dataset.get_class_weights()
            
            # Get unique patients
            train_patients = train_df['patient_id'].nunique()
            val_patients = val_df['patient_id'].nunique()
            
            print(f"\n{'='*60}")
            print(f"Dataset Setup - Fold {self.current_fold}/{self.n_folds-1}:")
            print(f"  Train: {train_patients} patients, {len(self.train_dataset)} samples")
            print(f"  Val:   {val_patients} patients, {len(self.val_dataset)} samples")
            
            # Show class distribution
            if self.stratify_by_class:
                print(f"\n  Stratification by consensus class:")
                for fold_df, name in [(train_df, 'Train'), (val_df, 'Val')]:
                    class_dist = fold_df['expert_consensus'].value_counts().sort_index()
                    print(f"    {name}: {dict(class_dist)}")
            
            # Show evaluator bin distribution
            if self.stratify_by_evaluators:
                print(f"\n  Stratification by evaluator bins:")
                for fold_df, name in [(train_df, 'Train'), (val_df, 'Val')]:
                    bin_dist = fold_df['evaluator_bin'].value_counts().sort_index()
                    print(f"    {name}: {dict(bin_dist)}")
            
            print(f"\n  Class weights: {self.class_weights.tolist()}")
            print(f"{'='*60}\n")

        # Mark this stage as done
        self._setup_done.add(stage_key)
    
    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")

        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_graphs,
            persistent_workers=self.num_workers > 0,
        )
        if self.num_workers > 0 and self.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(self.prefetch_factor)
        # Use forkserver when possible to reduce FD/mmap issues
        if self.num_workers > 0:
            try:
                loader_kwargs["multiprocessing_context"] = mp.get_context("forkserver")
            except Exception:
                pass
        return DataLoader(self.train_dataset, **loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")

        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_graphs,
            persistent_workers=self.num_workers > 0,
        )
        if self.num_workers > 0 and self.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(self.prefetch_factor)
        if self.num_workers > 0:
            try:
                loader_kwargs["multiprocessing_context"] = mp.get_context("forkserver")
            except Exception:
                pass
        return DataLoader(self.val_dataset, **loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader (uses validation split in this project)."""
        return self.val_dataloader()
    
    def get_num_classes(self) -> int:
        """Return number of classes."""
        return 6
    
    def get_class_weights(self) -> Optional[torch.Tensor]:
        """Return class weights for loss balancing."""
        return self.class_weights


__all__ = ["HMSDataModule"]
