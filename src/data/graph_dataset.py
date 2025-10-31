"""PyTorch Dataset for HMS preprocessed graph data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class HMSDataset(Dataset):
    """Dataset for loading preprocessed HMS brain activity graphs.
    
    Loads patient files from data/processed/ where each file contains:
    - patient_id → label_id → {eeg_graphs: List[9], spec_graphs: List[119], target: int}
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing preprocessed patient files (patient_*.pt)
    patient_ids : List[int]
        List of patient IDs to include in this dataset
    transform : callable, optional
        Optional transform to apply to samples
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        patient_ids: List[int],
        transform: Optional[callable] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.transform = transform
        
        # Build index: list of (patient_id, label_id) tuples
        self.samples: List[Tuple[int, int]] = []
        self._build_index()
        
    def _build_index(self) -> None:
        """Build index of all (patient_id, label_id) samples."""
        for patient_id in self.patient_ids:
            patient_path = self.data_dir / f"patient_{patient_id}.pt"
            
            if not patient_path.exists():
                print(f"Warning: Patient file not found: {patient_path}")
                continue
            
            # Load patient data
            try:
                # Use weights_only=False for PyTorch Geometric Data objects
                patient_data = torch.load(patient_path, weights_only=False)
                
                # Add all label_ids for this patient
                for label_id in patient_data.keys():
                    self.samples.append((patient_id, label_id))
                    
            except Exception as e:
                print(f"Error loading patient {patient_id}: {e}")
                continue
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)
    
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
        patient_id, label_id = self.samples[idx]
        
        # Load patient file (use weights_only=False for PyG Data objects)
        patient_path = self.data_dir / f"patient_{patient_id}.pt"
        patient_data = torch.load(patient_path, weights_only=False)
        
        # Get specific label data
        sample_data = patient_data[label_id]
        
        # Construct sample
        sample = {
            'eeg_graphs': sample_data['eeg_graphs'],
            'spec_graphs': sample_data['spec_graphs'],
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
        
        for patient_id, label_id in self.samples:
            patient_path = self.data_dir / f"patient_{patient_id}.pt"
            patient_data = torch.load(patient_path, weights_only=False)
            target = patient_data[label_id]['target']
            
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
    
    # Batch Spectrogram graphs: same process
    num_spec_timesteps = len(spec_sequences[0])
    batched_spec_graphs = []
    for t in range(num_spec_timesteps):
        graphs_at_t = [spec_sequences[b][t] for b in range(batch_size)]
        batched_graph = Batch.from_data_list(graphs_at_t)
        batched_spec_graphs.append(batched_graph)
    
    return {
        'eeg_graphs': batched_eeg_graphs,  # List[9] of Batch objects
        'spec_graphs': batched_spec_graphs,  # List[119] of Batch objects
        'targets': targets,  # (batch_size,)
        'patient_ids': patient_ids,
        'label_ids': label_ids,
    }


__all__ = ["HMSDataset", "collate_graphs"]
