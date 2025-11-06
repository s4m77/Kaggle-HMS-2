"""PyTorch Lightning DataModule with on-the-fly graph generation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from omegaconf import DictConfig

from src.data.graph_dataset_online import HMSOnlineDataset, custom_collate_fn
from src.data.utils.eeg_process import EEGGraphBuilder
from src.data.utils.spectrogram_process import SpectrogramGraphBuilder


class HMSOnlineDataModule(LightningDataModule):
    """Lightning DataModule with on-the-fly graph generation.
    
    Instead of loading pre-computed graphs, this datamodule:
    1. Loads raw EEG/spectrogram data during training
    2. Generates graphs dynamically in each batch
    3. Caches raw data in memory for fast access
    
    Benefits:
    - No need to pre-compute and store graphs
    - Can experiment with different graph construction parameters
    - Saves disk space
    
    Parameters
    ----------
    raw_data_dir : str or Path
        Directory containing raw parquet files (train_eegs/, train_spectrograms/)
    train_csv : str or Path
        Path to train_unique.csv with metadata
    graph_config : DictConfig
        Configuration for graph construction (from graphs.yaml)
    batch_size : int
        Batch size for training
    n_folds : int
        Number of folds for cross-validation
    current_fold : int
        Which fold to use for validation (0 to n_folds-1)
    stratify_by_class : bool
        Whether to stratify by consensus class
    stratify_by_evaluators : bool
        Whether to stratify by evaluator quality bins
    evaluator_bins : List[int]
        Bin edges for total_evaluators stratification
    min_evaluators : int
        Minimum total_evaluators to include
    num_workers : int
        Number of workers for DataLoader
    pin_memory : bool
        Whether to pin memory in DataLoader
    cache_raw_data : bool
        Whether to cache raw data in memory (recommended)
    shuffle_seed : int
        Random seed for fold splitting
    """
    
    def __init__(
        self,
        raw_data_dir: str | Path = "data/raw",
        train_csv: str | Path = "data/raw/train_unique.csv",
        graph_config: Optional[DictConfig] = None,
        batch_size: int = 32,
        n_folds: int = 5,
        current_fold: int = 0,
        stratify_by_class: bool = False,
        stratify_by_evaluators: bool = True,
        evaluator_bins: List[int] = [0, 5, 10, 15, 20, 999],
        min_evaluators: int = 0,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        cache_raw_data: bool = True,
        shuffle_seed: int = 42,
    ) -> None:
        super().__init__()
        
        self.raw_data_dir = Path(raw_data_dir)
        self.train_csv = Path(train_csv)
        self.graph_config = graph_config
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
        self.cache_raw_data = cache_raw_data
        self.shuffle_seed = shuffle_seed
        
        # Initialize graph builders
        self._init_graph_builders()
        
        # Datasets will be created in setup()
        self.train_dataset: Optional[HMSOnlineDataset] = None
        self.val_dataset: Optional[HMSOnlineDataset] = None
    
    def _init_graph_builders(self) -> None:
        """Initialize EEG and spectrogram graph builders from config."""
        if self.graph_config is None:
            # Default configuration
            self.eeg_builder = EEGGraphBuilder()
            self.spec_builder = SpectrogramGraphBuilder()
        else:
            # Build from config
            eeg_cfg = self.graph_config.eeg
            spec_cfg = self.graph_config.spectrogram
            
            # EEG builder
            self.eeg_builder = EEGGraphBuilder(
                sampling_rate=eeg_cfg.sampling_rate,
                window_size=eeg_cfg.window_size,
                stride=eeg_cfg.stride,
                bands=dict(eeg_cfg.bands),
                nperseg_factor=eeg_cfg.psd.nperseg_factor,
                channels=list(eeg_cfg.channels),
                # Preprocessing
                apply_bandpass=eeg_cfg.preprocessing.bandpass_filter.enabled,
                bandpass_low=eeg_cfg.preprocessing.bandpass_filter.lowcut,
                bandpass_high=eeg_cfg.preprocessing.bandpass_filter.highcut,
                bandpass_order=eeg_cfg.preprocessing.bandpass_filter.order,
                apply_notch=eeg_cfg.preprocessing.notch_filter.enabled,
                notch_freq=eeg_cfg.preprocessing.notch_filter.frequency,
                notch_q=eeg_cfg.preprocessing.notch_filter.quality_factor,
                apply_normalize=eeg_cfg.preprocessing.normalize.enabled,
            )
            
            # Spectrogram builder
            self.spec_builder = SpectrogramGraphBuilder(
                window_size=spec_cfg.window_size,
                stride=spec_cfg.stride,
                regions=list(spec_cfg.regions),
                bands=dict(spec_cfg.bands),
                aggregation=spec_cfg.aggregation,
                spatial_edges=list(spec_cfg.spatial_edges) if 'spatial_edges' in spec_cfg else None,
                apply_preprocessing=spec_cfg.preprocessing.enabled if 'preprocessing' in spec_cfg else True,
                clip_min=spec_cfg.preprocessing.clip_min if 'preprocessing' in spec_cfg else 1e-7,
                clip_max=spec_cfg.preprocessing.clip_max if 'preprocessing' in spec_cfg else 1e-4,
            )
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training and validation.
        
        Creates train/val splits using StratifiedGroupKFold.
        """
        # Load metadata
        df = pd.read_parquet(self.train_csv) if self.train_csv.suffix == '.parquet' else pd.read_csv(self.train_csv)
        
        # Strip whitespace from column names and string columns
        df.columns = df.columns.str.strip()
        if 'expert_consensus' in df.columns:
            df['expert_consensus'] = df['expert_consensus'].str.strip()
        
        # Calculate total_evaluators from vote columns
        vote_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
        df['total_evaluators'] = df[vote_cols].sum(axis=1)
        
        # Filter by minimum evaluators
        if self.min_evaluators > 0:
            n_before = len(df)
            df = df[df['total_evaluators'] >= self.min_evaluators].copy()
            n_after = len(df)
            print(f"Filtered by min_evaluators={self.min_evaluators}: {n_before} → {n_after} samples")
        
        # Create stratification variable
        if self.stratify_by_evaluators and self.stratify_by_class:
            # Dual stratification: combine class + evaluator bin
            df['evaluator_bin'] = pd.cut(
                df['total_evaluators'], 
                bins=self.evaluator_bins, 
                labels=False,
                include_lowest=True
            )
            # Combine into single stratification variable
            df['stratify_var'] = df['expert_consensus'].astype(str) + '_' + df['evaluator_bin'].astype(str)
            stratify_labels = df['stratify_var'].values
        elif self.stratify_by_evaluators:
            # Stratify by evaluator bins only
            df['evaluator_bin'] = pd.cut(
                df['total_evaluators'],
                bins=self.evaluator_bins,
                labels=False,
                include_lowest=True
            )
            stratify_labels = df['evaluator_bin'].values
        elif self.stratify_by_class:
            # Stratify by class only
            stratify_labels = df['expert_consensus'].values
        else:
            # No stratification (just group by patient)
            stratify_labels = np.zeros(len(df))
        
        # StratifiedGroupKFold split
        sgkf = StratifiedGroupKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.shuffle_seed
        )
        
        # Get train/val indices for current fold
        groups = df['patient_id'].values
        splits = list(sgkf.split(df, stratify_labels, groups=groups))
        train_idx, val_idx = splits[self.current_fold]
        
        # Create train and validation dataframes
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"\n{'='*60}")
        print(f"Fold {self.current_fold}/{self.n_folds-1} Setup (Online Graph Generation)")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_df)}")
        print(f"Val samples:   {len(val_df)}")
        print(f"Train patients: {train_df['patient_id'].nunique()}")
        print(f"Val patients:   {val_df['patient_id'].nunique()}")
        
        # Print stratification distribution
        if self.stratify_by_evaluators:
            train_bins = train_df['evaluator_bin'].value_counts().sort_index()
            val_bins = val_df['evaluator_bin'].value_counts().sort_index()
            print(f"\nEvaluator bin distribution:")
            print(f"  Train: {train_bins.to_dict()}")
            print(f"  Val:   {val_bins.to_dict()}")
        
        if self.stratify_by_class:
            train_classes = train_df['expert_consensus'].value_counts()
            val_classes = val_df['expert_consensus'].value_counts()
            print(f"\nClass distribution:")
            print(f"  Train: {train_classes.to_dict()}")
            print(f"  Val:   {val_classes.to_dict()}")
        
        print(f"{'='*60}\n")
        
        # Create datasets
        self.train_dataset = HMSOnlineDataset(
            raw_data_dir=self.raw_data_dir,
            metadata_df=train_df,
            eeg_builder=self.eeg_builder,
            spec_builder=self.spec_builder,
            cache_raw_data=self.cache_raw_data,
        )
        
        self.val_dataset = HMSOnlineDataset(
            raw_data_dir=self.raw_data_dir,
            metadata_df=val_df,
            eeg_builder=self.eeg_builder,
            spec_builder=self.spec_builder,
            cache_raw_data=self.cache_raw_data,
        )
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with GPU optimizations."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # Configurable pinned memory
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            collate_fn=custom_collate_fn,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader with GPU optimizations."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # Configurable pinned memory
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            collate_fn=custom_collate_fn,
        )


__all__ = ['HMSOnlineDataModule']
