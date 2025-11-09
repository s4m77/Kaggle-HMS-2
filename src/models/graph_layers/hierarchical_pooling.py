"""Hierarchical pooling for organizing node embeddings into regions."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn


# Channel to region mapping based on 10-20 system
# For EEG: 4 clinical regions (Frontal, Central, Parietal, Occipital)
EEG_CHANNEL_TO_REGION = {
    # FRONTAL (Fp, F, Fz)
    'Fp1': 0,  # Left Frontal
    'Fp2': 0,  # Right Frontal
    'F3': 0,   # Left Frontal
    'F4': 0,   # Right Frontal
    'F7': 0,   # Left Frontal
    'F8': 0,   # Right Frontal
    'Fz': 0,   # Central Frontal
    
    # CENTRAL (C, Cz, T3, T4)
    'C3': 1,   # Left Central
    'C4': 1,   # Right Central
    'Cz': 1,   # Central
    'T3': 1,   # Left Temporal (adjacent to central)
    'T4': 1,   # Right Temporal (adjacent to central)
    
    # PARIETAL (P, Pz, T5, T6)
    'P3': 2,   # Left Parietal
    'P4': 2,   # Right Parietal
    'Pz': 2,   # Central Parietal
    'T5': 2,   # Left Parietal-Temporal
    'T6': 2,   # Right Parietal-Temporal
    
    # OCCIPITAL (O)
    'O1': 3,   # Left Occipital
    'O2': 3,   # Right Occipital
}

# Region names for reference
REGION_NAMES = ['Frontal', 'Central', 'Parietal', 'Occipital']


class HierarchicalPoolingLayer(nn.Module):
    """Pool node embeddings by clinical regions.
    
    For EEG, organizes 19 channels into 4 clinical regions:
    - Region 0: Frontal (7 channels)
    - Region 1: Central (5 channels)
    - Region 2: Parietal (5 channels)
    - Region 3: Occipital (2 channels)
    
    For Spectrogram, keeps 4 nodes as-is (already regional).
    
    Parameters
    ----------
    channels : List[str], optional
        List of channel names (for EEG). If provided, creates channel-to-region mapping.
        If None, assumes input is already regional (e.g., spectrogram).
    num_regions : int
        Number of output regions (default: 4)
    pooling_method : str
        How to pool within regions: 'mean', 'max', or 'sum'
    """
    
    def __init__(
        self,
        channels: Optional[List[str]] = None,
        num_regions: int = 4,
        pooling_method: str = "mean",
    ) -> None:
        super().__init__()
        
        self.channels = channels
        self.num_regions = num_regions
        self.pooling_method = pooling_method.lower()
        
        if self.pooling_method not in ("mean", "max", "sum"):
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")
        
        # Build channel-to-region mapping if channels are provided
        if channels is not None:
            self.channel_to_region = self._build_channel_to_region_mapping(channels)
        else:
            # For spectrogram: identity mapping (4 nodes → 4 regions)
            self.channel_to_region = None
    
    def _build_channel_to_region_mapping(self, channels: List[str]) -> torch.Tensor:
        """Build a mapping from channel indices to region indices.
        
        Parameters
        ----------
        channels : List[str]
            Ordered list of channel names
        
        Returns
        -------
        torch.Tensor
            Tensor of shape (num_channels,) where element i is the region index for channel i
        """
        region_mapping = []
        for ch in channels:
            region_idx = EEG_CHANNEL_TO_REGION.get(ch, 0)  # Default to region 0 if not found
            region_mapping.append(region_idx)
        
        return torch.tensor(region_mapping, dtype=torch.long)
    
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Pool node embeddings by region.
        
        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings of shape (batch_size, num_nodes, embedding_dim)
            or (num_nodes, embedding_dim) for single sample
        
        Returns
        -------
        torch.Tensor
            Regional embeddings of shape (batch_size, num_regions, embedding_dim)
            or (num_regions, embedding_dim) for single sample
        """
        is_batched = node_embeddings.dim() == 3
        
        if not is_batched:
            # Add batch dimension if single sample
            node_embeddings = node_embeddings.unsqueeze(0)
        
        batch_size, num_nodes, embedding_dim = node_embeddings.shape
        device = node_embeddings.device
        
        # If no channel mapping, assume each node is already a region
        if self.channel_to_region is None:
            # For spectrogram: nodes are already regions
            regional_embeddings = node_embeddings
        else:
            # For EEG: pool channels by region
            region_mapping = self.channel_to_region.to(device)
            
            # Initialize regional embeddings
            regional_embeddings = torch.zeros(
                batch_size, self.num_regions, embedding_dim,
                dtype=node_embeddings.dtype,
                device=device
            )
            
            # Pool each region
            for region_idx in range(self.num_regions):
                # Get mask for channels in this region
                mask = region_mapping == region_idx
                
                if mask.sum() == 0:
                    # No channels for this region, keep zeros
                    continue
                
                # Get embeddings for this region
                region_nodes = node_embeddings[:, mask, :]  # (batch_size, num_nodes_in_region, embedding_dim)
                
                # Pool within region
                if self.pooling_method == "mean":
                    region_pooled = region_nodes.mean(dim=1)  # (batch_size, embedding_dim)
                elif self.pooling_method == "max":
                    region_pooled = region_nodes.max(dim=1)[0]  # (batch_size, embedding_dim)
                elif self.pooling_method == "sum":
                    region_pooled = region_nodes.sum(dim=1)  # (batch_size, embedding_dim)
                
                regional_embeddings[:, region_idx, :] = region_pooled
        
        if not is_batched:
            # Remove batch dimension if input was single sample
            regional_embeddings = regional_embeddings.squeeze(0)
        
        return regional_embeddings


__all__ = ["HierarchicalPoolingLayer", "EEG_CHANNEL_TO_REGION", "REGION_NAMES"]
