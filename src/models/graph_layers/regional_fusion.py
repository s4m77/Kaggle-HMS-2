"""Regional cross-modal attention fusion for EEG and Spectrogram features."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class RegionalCrossModalFusion(nn.Module):
    """Fuses EEG and Spectrogram regional features using cross-attention.
    
    Preserves regional structure by applying cross-attention between corresponding
    regions from each modality, then aggregating for final classification.
    
    Architecture:
    1. For each region pair: Apply bidirectional cross-attention
    2. Concatenate attended features from both modalities
    3. Optional: Apply attention pooling to aggregate regions into single vector
    
    This preserves anatomical structure and allows the model to learn which
    brain regions are most relevant for classification.
    
    Parameters
    ----------
    eeg_dim : int
        Dimension of EEG regional features (LSTM output dim)
    spec_dim : int
        Dimension of Spectrogram regional features (LSTM output dim)
    num_eeg_regions : int
        Number of EEG regions (default: 4 for Frontal, Central, Parietal, Occipital)
    num_spec_regions : int
        Number of Spectrogram regions (default: 4 for LL, RL, LP, RP)
    hidden_dim : int
        Hidden dimension for attention projection
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout probability
    use_attention_pooling : bool
        If True, use attention pooling to aggregate regions into single vector
        If False, return all regional features (for region-level analysis)
    """
    
    def __init__(
        self,
        eeg_dim: int,
        spec_dim: int,
        num_eeg_regions: int = 4,
        num_spec_regions: int = 4,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
        use_attention_pooling: bool = True,
    ) -> None:
        super().__init__()
        
        self.eeg_dim = eeg_dim
        self.spec_dim = spec_dim
        self.num_eeg_regions = num_eeg_regions
        self.num_spec_regions = num_spec_regions
        self.hidden_dim = hidden_dim
        self.use_attention_pooling = use_attention_pooling
        self.last_attention: Optional[Dict[str, torch.Tensor]] = None
        
        # Project to common dimension if needed
        self.eeg_proj = nn.Linear(eeg_dim, hidden_dim) if eeg_dim != hidden_dim else nn.Identity()
        self.spec_proj = nn.Linear(spec_dim, hidden_dim) if spec_dim != hidden_dim else nn.Identity()
        
        # Cross-attention: EEG regions attend to Spectrogram regions
        self.eeg_to_spec_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross-attention: Spectrogram regions attend to EEG regions
        self.spec_to_eeg_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer normalization for each modality
        self.eeg_norm = nn.LayerNorm(hidden_dim)
        self.spec_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Regional fusion: combine attended features from both modalities
        # Output per region: concat(eeg_attended, spec_attended)
        self.region_fusion_dim = hidden_dim * 2
        
        # Optional attention pooling to aggregate regions
        if use_attention_pooling:
            self.attention_pooling = nn.Sequential(
                nn.Linear(self.region_fusion_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            self.output_dim = self.region_fusion_dim
        else:
            self.attention_pooling = None
            # Output will be all regions concatenated
            # We'll use the minimum number of regions between the two modalities
            self.output_dim = self.region_fusion_dim * min(num_eeg_regions, num_spec_regions)
    
    def forward(
        self,
        eeg_features: torch.Tensor,
        spec_features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply regional cross-modal attention fusion.
        
        Parameters
        ----------
        eeg_features : torch.Tensor
            EEG regional features of shape (batch_size, num_eeg_regions, eeg_dim)
        spec_features : torch.Tensor
            Spectrogram regional features of shape (batch_size, num_spec_regions, spec_dim)
        
        Returns
        -------
        torch.Tensor
            If use_attention_pooling=True: Fused features of shape (batch_size, output_dim)
            If use_attention_pooling=False: Regional features of shape (batch_size, num_regions, output_dim)
        """
        batch_size = eeg_features.shape[0]
        
        # Validate input shapes
        if eeg_features.dim() != 3:
            raise ValueError(f"Expected eeg_features to be 3D (batch, regions, dim), got shape {eeg_features.shape}")
        if spec_features.dim() != 3:
            raise ValueError(f"Expected spec_features to be 3D (batch, regions, dim), got shape {spec_features.shape}")
        
        # Project to common dimension
        eeg_proj = self.eeg_proj(eeg_features)  # (batch, num_eeg_regions, hidden_dim)
        spec_proj = self.spec_proj(spec_features)  # (batch, num_spec_regions, hidden_dim)
        
        # EEG regions attend to Spectrogram regions
        # Query: EEG regions want information from Spec regions
        # Key/Value: Spec regions provide information
        self.last_attention = None

        eeg_attended, eeg_attn_weights = self.eeg_to_spec_attn(
            query=eeg_proj,
            key=spec_proj,
            value=spec_proj,
        )
        eeg_attended = self.eeg_norm(eeg_attended + eeg_proj)  # Residual connection
        eeg_attended = self.dropout(eeg_attended)
        # eeg_attended: (batch, num_eeg_regions, hidden_dim)
        
        # Spectrogram regions attend to EEG regions
        # Query: Spec regions want information from EEG regions
        # Key/Value: EEG regions provide information
        spec_attended, spec_attn_weights = self.spec_to_eeg_attn(
            query=spec_proj,
            key=eeg_proj,
            value=eeg_proj,
        )
        spec_attended = self.spec_norm(spec_attended + spec_proj)  # Residual connection
        spec_attended = self.dropout(spec_attended)
        # spec_attended: (batch, num_spec_regions, hidden_dim)
        
        # Handle region count mismatch by padding/trimming to match
        # Use the minimum number of regions
        num_regions = min(self.num_eeg_regions, self.num_spec_regions)
        
        # Trim to match region count (take first N regions)
        eeg_attended = eeg_attended[:, :num_regions, :]  # (batch, num_regions, hidden_dim)
        spec_attended = spec_attended[:, :num_regions, :]  # (batch, num_regions, hidden_dim)
        
        # Concatenate modalities for each region
        # Each region gets rich cross-modal representation
        regional_fused = torch.cat([eeg_attended, spec_attended], dim=-1)
        # regional_fused: (batch, num_regions, hidden_dim * 2)
        
        if self.use_attention_pooling:
            # Apply attention pooling to aggregate regions into single vector
            # Compute attention scores for each region
            attention_scores = self.attention_pooling(regional_fused)  # (batch, num_regions, 1)
            attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, num_regions, 1)
            
            # Weighted sum of regional features
            pooled = (regional_fused * attention_weights).sum(dim=1)  # (batch, region_fusion_dim)
            self.last_attention = {
                "eeg_to_spec": eeg_attn_weights.detach().cpu(),
                "spec_to_eeg": spec_attn_weights.detach().cpu(),
                "regional_pool": attention_weights.detach().cpu(),
            }
            return pooled
        else:
            # Return all regional features (flatten)
            # Useful for region-level analysis or interpretability
            fused_flat = regional_fused.reshape(batch_size, -1)  # (batch, num_regions * region_fusion_dim)
            self.last_attention = {
                "eeg_to_spec": eeg_attn_weights.detach().cpu(),
                "spec_to_eeg": spec_attn_weights.detach().cpu(),
            }
            return fused_flat

    def get_last_attention(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return the most recent set of attention weights."""
        return self.last_attention


__all__ = ["RegionalCrossModalFusion"]
