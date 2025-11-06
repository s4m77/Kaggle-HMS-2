"""PyTorch Dataset for HMS with on-the-fly graph generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from src.data.utils.eeg_process import EEGGraphBuilder, select_eeg_channels
from src.data.utils.spectrogram_process import SpectrogramGraphBuilder


class HMSOnlineDataset(Dataset):
    """Dataset that generates graphs on-the-fly from raw EEG/spectrogram data.
    
    Instead of loading pre-computed graphs, this dataset:
    1. Loads raw EEG/spectrogram data from parquet files
    2. Builds graphs dynamically during __getitem__
    3. Caches raw data in memory to speed up repeated access
    
    This is useful when:
    - You want to experiment with different graph construction parameters
    - Storage space is limited
    - Graph construction is fast enough for online generation
    
    Parameters
    ----------
    raw_data_dir : str or Path
        Directory containing raw EEG/spectrogram parquet files
    metadata_df : pd.DataFrame
        DataFrame with sample metadata (patient_id, label_id, expert_consensus, etc.)
    eeg_builder : EEGGraphBuilder
        Builder for constructing EEG graphs from raw signals
    spec_builder : SpectrogramGraphBuilder
        Builder for constructing spectrogram graphs
    cache_raw_data : bool
        Whether to cache raw EEG/spec data in memory (recommended)
    transform : callable, optional
        Optional transform to apply to samples
    """
    
    def __init__(
        self,
        raw_data_dir: str | Path,
        metadata_df: pd.DataFrame,
        eeg_builder: EEGGraphBuilder,
        spec_builder: SpectrogramGraphBuilder,
        cache_raw_data: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.eeg_builder = eeg_builder
        self.spec_builder = spec_builder
        self.cache_raw_data = cache_raw_data
        self.transform = transform
        
        # Build index
        self.sample_indices: List[int] = list(range(len(self.metadata_df)))
        
        # Label mapping
        self.label_map = {
            'Seizure': 0,
            'LPD': 1,
            'GPD': 2,
            'LRDA': 3,
            'GRDA': 4,
            'Other': 5,
        }
        
        # Cache for raw data (to avoid repeated parquet reads)
        self._eeg_cache: Dict[int, np.ndarray] = {}
        # Cache spectrogram as full DataFrame to match SpectrogramGraphBuilder API
        self._spec_cache: Dict[int, pd.DataFrame] = {}
        
        # Determine which files to cache
        if self.cache_raw_data:
            print(f"Pre-caching raw data for {len(metadata_df)} samples...")
            self._precache_raw_data()
    
    def _precache_raw_data(self) -> None:
        """Pre-load all raw EEG and spectrogram data into memory."""
        unique_eeg_ids = self.metadata_df['eeg_id'].unique()
        unique_spec_ids = self.metadata_df['spectrogram_id'].unique()
        
        print(f"  Loading {len(unique_eeg_ids)} unique EEG files...")
        for eeg_id in unique_eeg_ids:
            eeg_path = self.raw_data_dir / "train_eegs" / f"{eeg_id}.parquet"
            if eeg_path.exists():
                eeg_df = pd.read_parquet(eeg_path)
                # Select relevant channels (exclude EKG)
                eeg_data = select_eeg_channels(eeg_df, self.eeg_builder.channels)
                self._eeg_cache[eeg_id] = eeg_data
        
        print(f"  Loading {len(unique_spec_ids)} unique spectrogram files...")
        for spec_id in unique_spec_ids:
            spec_path = self.raw_data_dir / "train_spectrograms" / f"{spec_id}.parquet"
            if spec_path.exists():
                spec_df = pd.read_parquet(spec_path)
                # Store full DataFrame; builder handles preprocessing/selection
                self._spec_cache[spec_id] = spec_df
        
        print(f"  Cached {len(self._eeg_cache)} EEG and {len(self._spec_cache)} spectrogram files")
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.sample_indices)
    
    def _load_eeg(self, eeg_id: int) -> np.ndarray:
        """Load raw EEG data (from cache or disk)."""
        if eeg_id in self._eeg_cache:
            return self._eeg_cache[eeg_id]
        
        # Load from disk
        eeg_path = self.raw_data_dir / "train_eegs" / f"{eeg_id}.parquet"
        eeg_df = pd.read_parquet(eeg_path)
        eeg_data = select_eeg_channels(eeg_df, self.eeg_builder.channels)
        
        # Cache if enabled
        if self.cache_raw_data:
            self._eeg_cache[eeg_id] = eeg_data
        
        return eeg_data
    
    def _load_spectrogram(self, spec_id: int) -> pd.DataFrame:
        """Load raw spectrogram DataFrame (from cache or disk)."""
        if spec_id in self._spec_cache:
            return self._spec_cache[spec_id]
        
        # Load from disk
        spec_path = self.raw_data_dir / "train_spectrograms" / f"{spec_id}.parquet"
        spec_df = pd.read_parquet(spec_path)
        
        # Cache if enabled
        if self.cache_raw_data:
            self._spec_cache[spec_id] = spec_df
        
        return spec_df
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample with on-the-fly graph generation.
        
        Parameters
        ----------
        idx : int
            Sample index
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'eeg_graphs': List of PyG Data objects (dynamically generated)
            - 'spec_graphs': List of PyG Data objects (dynamically generated)
            - 'target': int (0-5) or float tensor (vote distribution)
            - 'patient_id': int
            - 'eeg_id': int
            - 'spectrogram_id': int
        """
        # Get metadata for this sample
        sample_idx = self.sample_indices[idx]
        row = self.metadata_df.iloc[sample_idx]
        
        patient_id = int(row['patient_id'])
        eeg_id = int(row['eeg_id'])
        spec_id = int(row['spectrogram_id'])
        
        # Load raw data
        eeg_data = self._load_eeg(eeg_id)
        spec_data = self._load_spectrogram(spec_id)
        
        # Build EEG graphs on-the-fly
        eeg_graphs = self.eeg_builder.process_eeg_signal(eeg_data)
        
        # Build spectrogram graphs on-the-fly (expects DataFrame)
        spec_graphs = self.spec_builder.process_spectrogram(spec_data)
        
        # Get target (support both class index and vote distribution)
        if 'expert_consensus' in row:
            target_label = row['expert_consensus']
            target = self.label_map.get(target_label, 5)  # Default to 'Other'
        else:
            # Fallback to class 5 if no label
            target = 5
        
        # Optionally load vote distribution for KL divergence loss
        # Check if vote columns exist
        vote_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
        if all(col in row for col in vote_cols):
            votes = torch.tensor([
                row['seizure_vote'],
                row['lpd_vote'],
                row['gpd_vote'],
                row['lrda_vote'],
                row['grda_vote'],
                row['other_vote']
            ], dtype=torch.float32)
            # Normalize to probability distribution
            if votes.sum() > 0:
                target = votes / votes.sum()
            else:
                # If no votes, use one-hot encoding of consensus
                target = torch.zeros(6, dtype=torch.float32)
                target[self.label_map.get(row.get('expert_consensus', 'Other'), 5)] = 1.0
        
        # Sanitize graphs (remove NaN/Inf)
        eeg_graphs = self._sanitize_graphs(eeg_graphs)
        spec_graphs = self._sanitize_graphs(spec_graphs)
        
        # Construct sample
        sample = {
            'eeg_graphs': eeg_graphs,
            'spec_graphs': spec_graphs,
            'target': target,
            'patient_id': patient_id,
            'eeg_id': eeg_id,
            'spectrogram_id': spec_id,
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _sanitize_graphs(self, graphs: List) -> List:
        """Remove NaN/Inf from graph features."""
        for g in graphs:
            if hasattr(g, 'x') and g.x is not None:
                g.x = torch.nan_to_num(g.x, nan=0.0, posinf=0.0, neginf=0.0)
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                g.edge_attr = torch.nan_to_num(g.edge_attr, nan=0.0, posinf=1.0, neginf=0.0)
        return graphs


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching graph data.
    
    Handles both PyG graphs and vote distributions.
    """
    from torch_geometric.data import Batch
    
    # Batch EEG graphs (list of lists -> list of batches)
    eeg_graphs_batched = []
    n_eeg_windows = len(batch[0]['eeg_graphs'])
    for i in range(n_eeg_windows):
        window_graphs = [sample['eeg_graphs'][i] for sample in batch]
        eeg_graphs_batched.append(Batch.from_data_list(window_graphs))
    
    # Batch spectrogram graphs
    spec_graphs_batched = []
    n_spec_windows = len(batch[0]['spec_graphs'])
    for i in range(n_spec_windows):
        window_graphs = [sample['spec_graphs'][i] for sample in batch]
        spec_graphs_batched.append(Batch.from_data_list(window_graphs))
    
    # Stack targets (handle both scalar and distribution targets)
    targets = [sample['target'] for sample in batch]
    if isinstance(targets[0], torch.Tensor) and targets[0].dim() > 0:
        # Vote distributions
        targets = torch.stack(targets)
    else:
        # Class indices
        targets = torch.tensor(targets, dtype=torch.long)
    
    # Stack metadata
    patient_ids = torch.tensor([sample['patient_id'] for sample in batch], dtype=torch.long)
    eeg_ids = torch.tensor([sample['eeg_id'] for sample in batch], dtype=torch.long)
    spec_ids = torch.tensor([sample['spectrogram_id'] for sample in batch], dtype=torch.long)
    
    return {
        'eeg_graphs': eeg_graphs_batched,
        'spec_graphs': spec_graphs_batched,
        'targets': targets,
        'patient_ids': patient_ids,
        'eeg_ids': eeg_ids,
        'spectrogram_ids': spec_ids,
    }


__all__ = ['HMSOnlineDataset', 'custom_collate_fn']
