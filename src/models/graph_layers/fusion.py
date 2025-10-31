"""Cross-modal attention fusion for EEG and Spectrogram features."""

from __future__ import annotations

import torch
from torch import nn


class CrossModalFusion(nn.Module):
    """Fuses EEG and Spectrogram features using cross-attention.
    
    Uses bidirectional cross-attention to allow each modality to attend
    to relevant parts of the other modality.
    
    Parameters
    ----------
    eeg_dim : int
        Dimension of EEG features
    spec_dim : int
        Dimension of Spectrogram features
    hidden_dim : int
        Hidden dimension for attention projection
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        eeg_dim: int,
        spec_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.eeg_dim = eeg_dim
        self.spec_dim = spec_dim
        self.hidden_dim = hidden_dim
        
        # Project to common dimension if needed
        self.eeg_proj = nn.Linear(eeg_dim, hidden_dim) if eeg_dim != hidden_dim else nn.Identity()
        self.spec_proj = nn.Linear(spec_dim, hidden_dim) if spec_dim != hidden_dim else nn.Identity()
        
        # Cross-attention: EEG attends to Spectrogram
        self.eeg_to_spec_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross-attention: Spectrogram attends to EEG
        self.spec_to_eeg_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer normalization
        self.eeg_norm = nn.LayerNorm(hidden_dim)
        self.spec_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension (concatenated)
        self.output_dim = hidden_dim * 2
    
    def forward(
        self,
        eeg_features: torch.Tensor,
        spec_features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-modal attention fusion.
        
        Parameters
        ----------
        eeg_features : torch.Tensor
            EEG features of shape (batch_size, eeg_dim) or (batch_size, seq_len, eeg_dim)
        spec_features : torch.Tensor
            Spectrogram features of shape (batch_size, spec_dim) or (batch_size, seq_len, spec_dim)
        
        Returns
        -------
        torch.Tensor
            Fused features of shape (batch_size, output_dim)
        """
        # Ensure inputs are 3D for attention (add seq dimension if needed)
        if eeg_features.dim() == 2:
            eeg_features = eeg_features.unsqueeze(1)  # (batch, 1, eeg_dim)
        if spec_features.dim() == 2:
            spec_features = spec_features.unsqueeze(1)  # (batch, 1, spec_dim)
        
        # Project to common dimension
        eeg_proj = self.eeg_proj(eeg_features)  # (batch, seq_len, hidden_dim)
        spec_proj = self.spec_proj(spec_features)  # (batch, seq_len, hidden_dim)
        
        # EEG attends to Spectrogram
        eeg_attended, _ = self.eeg_to_spec_attn(
            query=eeg_proj,
            key=spec_proj,
            value=spec_proj,
        )
        eeg_attended = self.eeg_norm(eeg_attended + eeg_proj)  # Residual connection
        eeg_attended = self.dropout(eeg_attended)
        
        # Spectrogram attends to EEG
        spec_attended, _ = self.spec_to_eeg_attn(
            query=spec_proj,
            key=eeg_proj,
            value=eeg_proj,
        )
        spec_attended = self.spec_norm(spec_attended + spec_proj)  # Residual connection
        spec_attended = self.dropout(spec_attended)
        
        # Pool temporal dimension (mean pooling)
        eeg_pooled = eeg_attended.mean(dim=1)  # (batch, hidden_dim)
        spec_pooled = spec_attended.mean(dim=1)  # (batch, hidden_dim)
        
        # Concatenate modalities
        fused = torch.cat([eeg_pooled, spec_pooled], dim=1)  # (batch, 2*hidden_dim)
        
        return fused


__all__ = ["CrossModalFusion"]
