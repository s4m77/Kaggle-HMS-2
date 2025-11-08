"""PyTorch Dataset for HMS preprocessed graph data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
try:
    from tqdm import tqdm
except Exception:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class HMSDataset(Dataset):
    """Dataset for loading preprocessed HMS brain activity graphs.
    
    Loads patient files from data/processed/ where each file contains:
    - patient_id → label_id → {eeg_graphs: List[9], spec_graphs: List[119], target: int}
    
    This dataset works with a metadata DataFrame (from train_unique.csv) to:
    - Filter samples based on fold assignments
    - Access sample metadata (patient_id, label_id, votes, etc.)
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing preprocessed patient files (patient_*.pt)
    metadata_df : pd.DataFrame
        DataFrame with columns: patient_id, label_id, expert_consensus, *_vote, etc.
        Must have 'fold' column if using for train/val split
    is_train : bool
        Whether this is training set (affects which samples to use based on fold)
    transform : callable, optional
        Optional transform to apply to samples
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        metadata_df: pd.DataFrame,
        is_train: bool = True,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        preload_patients: bool = False,
        remote_cache: Optional[Dict[int, Dict[int, Dict[str, Any]]]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.is_train = is_train
        self.transform = transform
        self.preload_patients = preload_patients
        self.remote_cache = remote_cache
        
        # Build index: list of indices into metadata_df
        self.sample_indices: List[int] = list(range(len(self.metadata_df)))
        
        # Map labels to indices
        self.label_map = {
            'Seizure': 0,
            'LPD': 1,
            'GPD': 2,
            'LRDA': 3,
            'GRDA': 4,
            'Other': 5,
        }

        # In-memory per-patient cache to avoid repeated disk I/O and serialization
        self._patient_cache: Dict[int, Dict[int, Dict[str, Any]]] = {}

        if self.remote_cache is None and self.preload_patients:
            # Preload and sanitize all referenced patients (uses RAM; ensure capacity)
            unique_pids = sorted(self.metadata_df['patient_id'].unique())
            for pid in tqdm(unique_pids, desc="Preloading patients", total=len(unique_pids)):
                path = self.data_dir / f"patient_{int(pid)}.pt"
                if path.exists():
                    data = torch.load(path, weights_only=False)
                    self._patient_cache[int(pid)] = self._sanitize_patient_data(data)
        
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample.
        
        Parameters
        ----------
        idx : int
            Sample index
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'eeg_graphs': List of 9 PyG Data objects
            - 'spec_graphs': List of 119 PyG Data objects
            - 'target': int (0-5)
            - 'patient_id': int
            - 'label_id': int
        """
        # Get metadata for this sample
        sample_idx = self.sample_indices[idx]
        row = self.metadata_df.iloc[sample_idx]
        
        patient_id = int(row['patient_id'])
        label_id = int(row['label_id'])
        
        # Load patient file (cache and sanitize once per patient)
        if self.remote_cache is not None:
            patient_data = self.remote_cache[patient_id]
        elif patient_id in self._patient_cache:
            patient_data = self._patient_cache[patient_id]
        else:
            patient_path = self.data_dir / f"patient_{patient_id}.pt"
            patient_data = torch.load(patient_path, weights_only=False)
            patient_data = self._sanitize_patient_data(patient_data)
            self._patient_cache[patient_id] = patient_data
        
        # Get specific label data
        sample_data = patient_data[label_id]
        
        # Already sanitized on load
        eeg_graphs = sample_data['eeg_graphs']
        spec_graphs = sample_data['spec_graphs']
        
        # Construct sample
        sample = {
            'eeg_graphs': eeg_graphs,
            'spec_graphs': spec_graphs,
            'target': sample_data['target'],  # Shape: (6,) vote distribution
            'consensus_label': sample_data.get('consensus_label', -1),  # Integer label for metrics
            'patient_id': patient_id,
            'label_id': label_id,
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def _sanitize_patient_data(self, patient_data: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Sanitize NaN/Inf in features once per patient and drop gamma band if present."""
        for lbl_id, item in patient_data.items():
            # EEG graphs
            for g in item.get('eeg_graphs', []):
                if hasattr(g, 'x') and g.x is not None:
                    g.x = torch.nan_to_num(g.x, nan=0.0, posinf=0.0, neginf=0.0)
                if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                    g.edge_attr = torch.nan_to_num(g.edge_attr, nan=0.0, posinf=1.0, neginf=0.0)
            # Spec graphs
            for g in item.get('spec_graphs', []):
                if hasattr(g, 'x') and g.x is not None:
                    x = torch.nan_to_num(g.x, nan=0.0, posinf=0.0, neginf=0.0)
                    # Ensure 4 features (drop gamma if accidentally present)
                    if x.dim() == 2 and x.size(1) == 5:
                        x = x[:, :4]
                    g.x = x
        return patient_data
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes in dataset.
        
        Returns
        -------
        dict
            Dictionary mapping class index to count
        """
        class_counts = {}
        
        for sample_idx in self.sample_indices:
            row = self.metadata_df.iloc[sample_idx]
            label = row['expert_consensus'].strip()
            target = self.label_map.get(label, -1)
            
            if target >= 0:
                class_counts[target] = class_counts.get(target, 0) + 1
        
        return class_counts
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights for loss balancing.
        
        Returns
        -------
        torch.Tensor
            Weights of shape (num_classes,)
        """
        class_counts = self.get_class_distribution()
        num_classes = max(class_counts.keys()) + 1
        
        # Count per class
        counts = torch.zeros(num_classes)
        for cls, count in class_counts.items():
            counts[cls] = count
        
        # Compute inverse frequency weights
        total = counts.sum()
        weights = total / (num_classes * counts)
        
        # Normalize so mean weight is 1.0
        weights = weights / weights.mean()
        
        return weights


def collate_graphs(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader to batch graph sequences.
    
    Parameters
    ----------
    batch : List[Dict]
        List of samples from HMSDataset
        
    Returns
    -------
    dict
        Batched data with keys:
        - 'eeg_graphs': List of 9 batched graphs (each is a PyG Batch)
        - 'spec_graphs': List of 119 batched graphs (each is a PyG Batch)
        - 'targets': LongTensor of shape (batch_size,)
        - 'patient_ids': List of patient IDs
        - 'label_ids': List of label IDs
    """
    batch_size = len(batch)
    
    # Extract components
    eeg_sequences = [sample['eeg_graphs'] for sample in batch]  # List[List[9 graphs]]
    spec_sequences = [sample['spec_graphs'] for sample in batch]  # List[List[119 graphs]]
    
    # Handle both old format (int) and new format (tensor) for targets
    target_list = []
    consensus_label_list = []
    old_format_detected = False
    
    for sample in batch:
        target = sample['target']
        if isinstance(target, torch.Tensor):
            # New format: vote distribution (shape: [6])
            target_list.append(target)
            # Get consensus label from tensor (argmax) or from explicit field
            consensus_label = sample.get('consensus_label', torch.argmax(target).item())
        else:
            # Old format: integer class label
            # Convert to one-hot distribution for compatibility
            old_format_detected = True
            target_tensor = torch.zeros(6, dtype=torch.float32)
            target_tensor[int(target)] = 1.0
            target_list.append(target_tensor)
            consensus_label = int(target)
        consensus_label_list.append(consensus_label)
    
    # Warn user about old format (only once per process)
    if old_format_detected and not hasattr(collate_graphs, '_old_format_warned'):
        import warnings
        warnings.warn(
            "Old data format detected (integer labels). Converting to one-hot encoding. "
            "For true vote distributions, reprocess data with: "
            "python src/data/make_graph_dataset.py --config configs/graphs.yaml",
            UserWarning
        )
        collate_graphs._old_format_warned = True
    
    targets = torch.stack(target_list)  # Shape: (batch_size, 6)
    consensus_labels = torch.tensor(consensus_label_list, dtype=torch.long)
    patient_ids = [sample['patient_id'] for sample in batch]
    label_ids = [sample['label_id'] for sample in batch]
    
    # Batch EEG graphs: transpose to get 9 lists, each with batch_size graphs
    num_eeg_timesteps = len(eeg_sequences[0])
    batched_eeg_graphs = []
    for t in range(num_eeg_timesteps):
        # Get all graphs at timestep t across batch
        graphs_at_t = [eeg_sequences[b][t] for b in range(batch_size)]
        # Batch them into single PyG Batch object
        batched_graph = Batch.from_data_list(graphs_at_t)
        batched_eeg_graphs.append(batched_graph)
    
    # Batch Spectrogram graphs: keep all timesteps to preserve temporal information
    num_spec_timesteps = len(spec_sequences[0])
    batched_spec_graphs = []
    for t in range(num_spec_timesteps):
        graphs_at_t = [spec_sequences[b][t] for b in range(batch_size)]
        # Safety: ensure 4 features
        for graph in graphs_at_t:
            if hasattr(graph, 'x') and graph.x is not None and graph.x.dim() == 2 and graph.x.size(1) == 5:
                graph.x = graph.x[:, :4]
        batched_graph = Batch.from_data_list(graphs_at_t)
        batched_spec_graphs.append(batched_graph)
    
    return {
        'eeg_graphs': batched_eeg_graphs,  # List[9] of Batch objects
        'spec_graphs': batched_spec_graphs,  # List[119] of Batch objects (with 4 features each)
        'targets': targets,  # (batch_size, 6) - vote probability distributions
        'consensus_labels': consensus_labels,  # (batch_size,) - integer labels for metrics
        'patient_ids': patient_ids,
        'label_ids': label_ids,
    }


__all__ = ["HMSDataset", "collate_graphs"]
