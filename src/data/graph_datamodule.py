"""PyTorch Lightning DataModule for HMS dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
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
    stratify_by_evaluators : bool
        Whether to stratify by total_evaluators bins
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
        stratify_by_evaluators: bool = True,
        evaluator_bins: List[int] = [0, 5, 10, 15, 20, 999],
        min_evaluators: int = 0,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_seed: int = 42,
    ) -> None:
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.train_csv = Path(train_csv)
        self.batch_size = batch_size
        self.n_folds = n_folds
        self.current_fold = current_fold
        self.stratify_by_evaluators = stratify_by_evaluators
        self.evaluator_bins = evaluator_bins
        self.min_evaluators = min_evaluators
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_seed = shuffle_seed
        
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
        
        # Create stratification bins
        if self.stratify_by_evaluators:
            self.metadata_df['evaluator_bin'] = pd.cut(
                self.metadata_df['total_evaluators'],
                bins=self.evaluator_bins,
                labels=False,
                include_lowest=True
            )
        else:
            # No stratification, use constant value
            self.metadata_df['evaluator_bin'] = 0
        
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
            y=self.metadata_df['evaluator_bin'],
            groups=patient_groups
        )):
            self.metadata_df.loc[val_idx, 'fold'] = fold
        
        # Create train and validation datasets
        if stage == "fit" or stage is None:
            train_df = self.metadata_df[
                self.metadata_df['fold'] != self.current_fold
            ].reset_index(drop=True)
            
            val_df = self.metadata_df[
                self.metadata_df['fold'] == self.current_fold
            ].reset_index(drop=True)
            
            self.train_dataset = HMSDataset(
                data_dir=self.data_dir,
                metadata_df=train_df,
                is_train=True,
            )
            
            self.val_dataset = HMSDataset(
                data_dir=self.data_dir,
                metadata_df=val_df,
                is_train=False,
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
            if self.stratify_by_evaluators:
                print(f"\n  Stratification by evaluator bins:")
                for fold_df, name in [(train_df, 'Train'), (val_df, 'Val')]:
                    bin_dist = fold_df['evaluator_bin'].value_counts().sort_index()
                    print(f"    {name}: {dict(bin_dist)}")
            print(f"\n  Class weights: {self.class_weights.tolist()}")
            print(f"{'='*60}\n")
    
    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_graphs,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_graphs,
            persistent_workers=self.num_workers > 0,
        )
    
    def get_num_classes(self) -> int:
        """Return number of classes."""
        return 6
    
    def get_class_weights(self) -> Optional[torch.Tensor]:
        """Return class weights for loss balancing."""
        return self.class_weights


__all__ = ["HMSDataModule"]
