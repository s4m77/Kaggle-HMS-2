"""PyTorch Lightning DataModule for HMS dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import numpy as np

from src.data.graph_dataset import HMSDataset, collate_graphs


class HMSDataModule(LightningDataModule):
    """Lightning DataModule for HMS brain activity classification.
    
    Handles:
    - Train/val/test splitting by patient (ensures no patient overlap)
    - DataLoader creation with custom collate function
    - Class weight computation for imbalanced data
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing preprocessed patient files
    batch_size : int
        Batch size for training
    train_ratio : float
        Proportion of data for training (default: 0.8)
    val_ratio : float
        Proportion of data for validation (default: 0.1)
    test_ratio : float
        Proportion of data for testing (default: 0.1)
    num_workers : int
        Number of workers for DataLoader
    pin_memory : bool
        Whether to pin memory in DataLoader
    shuffle_seed : int
        Random seed for patient shuffling
    """
    
    def __init__(
        self,
        data_dir: str | Path = "data/processed",
        batch_size: int = 32,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_seed: int = 42,
    ) -> None:
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_seed = shuffle_seed
        
        # Check ratios sum to 1.0
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        
        # Will be set in setup()
        self.train_dataset: Optional[HMSDataset] = None
        self.val_dataset: Optional[HMSDataset] = None
        self.test_dataset: Optional[HMSDataset] = None
        self.class_weights: Optional[torch.Tensor] = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage.
        
        Parameters
        ----------
        stage : str, optional
            Either 'fit', 'validate', 'test', or 'predict'
        """
        # Get all patient IDs from directory
        patient_files = sorted(self.data_dir.glob("patient_*.pt"))
        patient_ids = [
            int(f.stem.split('_')[1]) for f in patient_files
        ]
        
        if len(patient_ids) == 0:
            raise ValueError(f"No patient files found in {self.data_dir}")
        
        # Shuffle patients with fixed seed for reproducibility
        rng = np.random.RandomState(self.shuffle_seed)
        patient_ids = list(patient_ids)
        rng.shuffle(patient_ids)
        
        # Split by patient to avoid data leakage
        n_patients = len(patient_ids)
        n_train = int(n_patients * self.train_ratio)
        n_val = int(n_patients * self.val_ratio)
        
        train_patients = patient_ids[:n_train]
        val_patients = patient_ids[n_train:n_train + n_val]
        test_patients = patient_ids[n_train + n_val:]
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = HMSDataset(
                data_dir=self.data_dir,
                patient_ids=train_patients,
            )
            self.val_dataset = HMSDataset(
                data_dir=self.data_dir,
                patient_ids=val_patients,
            )
            
            # Compute class weights from training set
            self.class_weights = self.train_dataset.get_class_weights()
            
            print(f"\n{'='*60}")
            print(f"Dataset Setup:")
            print(f"  Train: {len(train_patients)} patients, {len(self.train_dataset)} samples")
            print(f"  Val:   {len(val_patients)} patients, {len(self.val_dataset)} samples")
            print(f"  Class weights: {self.class_weights.tolist()}")
            print(f"{'='*60}\n")
        
        if stage == "test" or stage is None:
            self.test_dataset = HMSDataset(
                data_dir=self.data_dir,
                patient_ids=test_patients,
            )
            
            print(f"\n{'='*60}")
            print(f"Test Dataset Setup:")
            print(f"  Test:  {len(test_patients)} patients, {len(self.test_dataset)} samples")
            print(f"{'='*60}\n")
    
    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
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
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_graphs,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        return DataLoader(
            self.test_dataset,
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
