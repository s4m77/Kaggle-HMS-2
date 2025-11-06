"""PyTorch Dataset for HMS preprocessed graph data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

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
    ) -> None:
        self.data_dir = Path(data_dir)
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.is_train = is_train
        self.transform = transform
        
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
        
        # Load patient file (use weights_only=False for PyG Data objects)
        patient_path = self.data_dir / f"patient_{patient_id}.pt"
        patient_data = torch.load(patient_path, weights_only=False)
        
        # Get specific label data
        sample_data = patient_data[label_id]
        
        # Clean NaN/Inf from graphs (safety check for corrupted preprocessing)
        eeg_graphs = sample_data['eeg_graphs']
        spec_graphs = sample_data['spec_graphs']
        
        # Replace NaN/Inf in EEG graph features
        for graph in eeg_graphs:
            if hasattr(graph, 'x') and graph.x is not None:
                graph.x = torch.nan_to_num(graph.x, nan=0.0, posinf=0.0, neginf=0.0)
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                graph.edge_attr = torch.nan_to_num(graph.edge_attr, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Replace NaN/Inf in Spec graph features
        for graph in spec_graphs:
            if hasattr(graph, 'x') and graph.x is not None:
                graph.x = torch.nan_to_num(graph.x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Construct sample
        sample = {
            'eeg_graphs': eeg_graphs,
            'spec_graphs': spec_graphs,
            'target': sample_data['target'],
            'patient_id': patient_id,
            'label_id': label_id,
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
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
    targets = torch.tensor([sample['target'] for sample in batch], dtype=torch.long)
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
    
    # Batch Spectrogram graphs: drop the 5th feature (gamma band - always 0)
    num_spec_timesteps = len(spec_sequences[0])
    batched_spec_graphs = []
    for t in range(num_spec_timesteps):
        graphs_at_t = [spec_sequences[b][t] for b in range(batch_size)]

        if len(graphs_at_t) > 40:  # If more than 40 spec windows
            indices = np.linspace(0, len(graphs_at_t)-1, 40, dtype=int)
            graphs_at_t = [graphs_at_t[i] for i in indices]
        
        # Drop the 5th feature (index 4) from each graph's node features
        for graph in graphs_at_t:
            if graph.x.size(1) == 5:  # If has 5 features, drop the last one
                graph.x = graph.x[:, :4]  # Keep only first 4 features (delta, theta, alpha, beta)
        
        # Batch them into single PyG Batch object
        batched_graph = Batch.from_data_list(graphs_at_t)
        batched_spec_graphs.append(batched_graph)
    
    return {
        'eeg_graphs': batched_eeg_graphs,  # List[9] of Batch objects
        'spec_graphs': batched_spec_graphs,  # List[119] of Batch objects (with 4 features each)
        'targets': targets,  # (batch_size,)
        'patient_ids': patient_ids,
        'label_ids': label_ids,
    }


__all__ = ["HMSDataset", "collate_graphs"]
