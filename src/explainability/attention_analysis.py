"""
Attention analysis utilities for cross-modal fusion explainability.

Provides functions to extract, analyze, and interpret attention weights from
the cross-modal fusion module.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from dataclasses import dataclass
from scipy.stats import entropy


@dataclass
class AttentionStatistics:
    """Container for attention weight statistics."""
    mean: float
    std: float
    max: float
    min: float
    entropy: float
    sparsity: float  # Percentage of near-zero weights
    
    def __str__(self) -> str:
        return (
            f"AttentionStats(mean={self.mean:.4f}, std={self.std:.4f}, "
            f"max={self.max:.4f}, entropy={self.entropy:.4f}, sparsity={self.sparsity:.2f}%)"
        )


class AttentionAnalyzer:
    """
    Analyzes and interprets cross-modal attention weights.
    
    Provides methods to understand how the model attends across modalities,
    compute statistical measures, and identify important attention patterns.
    """
    
    def __init__(self, num_heads: int = 8):
        """
        Parameters
        ----------
        num_heads : int
            Number of attention heads in the fusion module
        """
        self.num_heads = num_heads
    
    def analyze_attention_weights(
        self,
        attention_weights: torch.Tensor,
        name: str = "attention",
    ) -> AttentionStatistics:
        """
        Compute statistics from attention weight matrix.
        
        Parameters
        ----------
        attention_weights : torch.Tensor
            Attention weights of shape (batch, num_heads, query_len, key_len)
        name : str
            Name for logging
        
        Returns
        -------
        AttentionStatistics
            Statistical summary of attention weights
        """
        # Move to CPU and convert to numpy
        weights_np = attention_weights.detach().cpu().numpy()
        
        # Flatten all dimensions except keeping structure
        weights_flat = weights_np.flatten()
        
        # Compute statistics
        mean = float(np.mean(weights_flat))
        std = float(np.std(weights_flat))
        max_val = float(np.max(weights_flat))
        min_val = float(np.min(weights_flat))
        
        # Entropy (normalized to [0, 1])
        attn_entropy = entropy(weights_flat + 1e-8)
        max_entropy = np.log(len(weights_flat))
        normalized_entropy = attn_entropy / max_entropy if max_entropy > 0 else 0
        
        # Sparsity: percentage of weights < 0.1 (near-zero threshold)
        sparsity = float((weights_flat < 0.1).sum() / len(weights_flat) * 100)
        
        return AttentionStatistics(
            mean=mean,
            std=std,
            max=max_val,
            min=min_val,
            entropy=float(normalized_entropy),
            sparsity=sparsity,
        )
    
    def get_top_attention_pairs(
        self,
        attention_weights: torch.Tensor,
        top_k: int = 5,
    ) -> List[Tuple[int, int, float]]:
        """
        Get top-k strongest attention connections.
        
        Parameters
        ----------
        attention_weights : torch.Tensor
            Attention weights of shape (batch, num_heads, query_len, key_len)
        top_k : int
            Number of top pairs to return
        
        Returns
        -------
        List[Tuple[int, int, float]]
            List of (query_idx, key_idx, attention_weight) sorted by weight
        """
        # Average over batch and heads
        avg_weights = attention_weights.mean(dim=(0, 1))  # (query_len, key_len)
        
        # Get indices of top-k values
        top_values, top_indices = torch.topk(
            avg_weights.flatten(),
            k=min(top_k, avg_weights.numel()),
        )
        
        # Convert flat indices to 2D
        pairs = []
        for value, flat_idx in zip(top_values, top_indices):
            query_idx = flat_idx // avg_weights.shape[1]
            key_idx = flat_idx % avg_weights.shape[1]
            pairs.append((int(query_idx), int(key_idx), float(value)))
        
        return pairs
    
    def compute_attention_entropy(
        self,
        attention_weights: torch.Tensor,
        per_head: bool = False,
    ) -> float | Dict[int, float]:
        """
        Compute attention entropy (concentration/diffusion).
        
        High entropy = attention spread across many positions (diffuse)
        Low entropy = attention concentrated on few positions (sharp)
        
        Parameters
        ----------
        attention_weights : torch.Tensor
            Attention weights of shape (batch, num_heads, query_len, key_len)
        per_head : bool
            If True, return entropy per head; if False, return average
        
        Returns
        -------
        float or Dict[int, float]
            Average entropy or per-head entropy dictionary
        """
        weights_np = attention_weights.detach().cpu().numpy()
        
        if per_head:
            entropies = {}
            # Average over batch and query_len
            for head_idx in range(weights_np.shape[1]):
                head_weights = weights_np[:, head_idx, :, :]  # (batch, query, key)
                head_weights_avg = head_weights.mean(axis=(0, 1))  # (key,)
                head_entropy = entropy(head_weights_avg + 1e-8)
                entropies[head_idx] = head_entropy
            return entropies
        else:
            # Compute global entropy
            avg_weights = weights_np.mean(axis=(0, 1, 2))  # scalar
            return float(entropy(weights_np.flatten() + 1e-8))
    
    def compare_modality_agreement(
        self,
        eeg_to_spec_weights: torch.Tensor,
        spec_to_eeg_weights: torch.Tensor,
    ) -> float:
        """
        Measure agreement between cross-modal attentions.
        
        If EEG → Spec attention and Spec → EEG attention are similar,
        modalities are well-aligned. Returns correlation coefficient.
        
        Parameters
        ----------
        eeg_to_spec_weights : torch.Tensor
            EEG query to Spec key attention, shape (batch, num_heads, query_len, key_len)
        spec_to_eeg_weights : torch.Tensor
            Spec query to EEG key attention, shape (batch, num_heads, query_len, key_len)
        
        Returns
        -------
        float
            Correlation coefficient in range [-1, 1] where 1 = perfect agreement
        """
        # Average over batch and heads for simplicity
        eeg_avg = eeg_to_spec_weights.mean(dim=(0, 1)).flatten()
        spec_avg = spec_to_eeg_weights.mean(dim=(0, 1)).flatten()
        
        # Pad if different lengths
        max_len = max(len(eeg_avg), len(spec_avg))
        eeg_padded = torch.nn.functional.pad(eeg_avg, (0, max_len - len(eeg_avg)))
        spec_padded = torch.nn.functional.pad(spec_avg, (0, max_len - len(spec_avg)))
        
        # Compute Pearson correlation
        correlation = torch.corrcoef(torch.stack([eeg_padded, spec_padded]))[0, 1]
        return float(correlation)


def extract_attention_statistics(
    attention_dict: Dict,
    num_heads: int = 8,
) -> Dict[str, AttentionStatistics]:
    """
    Extract statistics from attention dictionary returned by fusion module.
    
    Parameters
    ----------
    attention_dict : Dict
        Dictionary with 'eeg_to_spec' and 'spec_to_eeg' attention weights
    num_heads : int
        Number of attention heads
    
    Returns
    -------
    Dict[str, AttentionStatistics]
        Statistics for each attention direction
    """
    analyzer = AttentionAnalyzer(num_heads=num_heads)
    
    stats = {
        'eeg_to_spec': analyzer.analyze_attention_weights(
            attention_dict['eeg_to_spec'],
            name='EEG→Spec',
        ),
        'spec_to_eeg': analyzer.analyze_attention_weights(
            attention_dict['spec_to_eeg'],
            name='Spec→EEG',
        ),
    }
    
    return stats


def compute_modality_alignment(
    attention_dict: Dict,
    num_heads: int = 8,
) -> Dict[str, float]:
    """
    Compute alignment metrics between modalities.
    
    Parameters
    ----------
    attention_dict : Dict
        Dictionary with attention weights
    num_heads : int
        Number of attention heads
    
    Returns
    -------
    Dict[str, float]
        Alignment metrics including correlation and entropy measures
    """
    analyzer = AttentionAnalyzer(num_heads=num_heads)
    
    eeg_to_spec = attention_dict['eeg_to_spec']
    spec_to_eeg = attention_dict['spec_to_eeg']
    
    # Compute agreement
    agreement = analyzer.compare_modality_agreement(eeg_to_spec, spec_to_eeg)
    
    # Compute entropies
    eeg_to_spec_entropy = analyzer.compute_attention_entropy(eeg_to_spec)
    spec_to_eeg_entropy = analyzer.compute_attention_entropy(spec_to_eeg)
    
    return {
        'modality_agreement': agreement,
        'eeg_to_spec_entropy': eeg_to_spec_entropy,
        'spec_to_eeg_entropy': spec_to_eeg_entropy,
        'entropy_difference': abs(eeg_to_spec_entropy - spec_to_eeg_entropy),
    }


def head_wise_attention_analysis(
    attention_dict: Dict,
    num_heads: int = 8,
) -> Dict[str, Dict[int, Dict]]:
    """
    Analyze attention patterns per head (specialization analysis).
    
    Different heads often specialize in different patterns. This shows
    which heads focus on which modality interactions.
    
    Parameters
    ----------
    attention_dict : Dict
        Dictionary with attention weights
    num_heads : int
        Number of attention heads
    
    Returns
    -------
    Dict[str, Dict[int, Dict]]
        Per-head statistics for each attention direction
    """
    analyzer = AttentionAnalyzer(num_heads=num_heads)
    
    eeg_to_spec = attention_dict['eeg_to_spec']
    spec_to_eeg = attention_dict['spec_to_eeg']
    
    results = {'eeg_to_spec': {}, 'spec_to_eeg': {}}
    
    # Analyze each head
    for direction, weights in [('eeg_to_spec', eeg_to_spec), ('spec_to_eeg', spec_to_eeg)]:
        weights_np = weights.detach().cpu().numpy()  # (batch, heads, query, key)
        
        for head_idx in range(num_heads):
            head_weights = torch.from_numpy(weights_np[:, head_idx, :, :])
            
            # Create dummy tensor for analysis (batch=1)
            head_weights_expanded = head_weights.unsqueeze(0).unsqueeze(1)
            
            stats = analyzer.analyze_attention_weights(head_weights_expanded)
            top_pairs = analyzer.get_top_attention_pairs(head_weights_expanded, top_k=3)
            
            results[direction][head_idx] = {
                'statistics': stats,
                'top_pairs': top_pairs,
                'specialization_score': stats.entropy,  # Lower entropy = more specialized
            }
    
    return results


__all__ = [
    'AttentionAnalyzer',
    'AttentionStatistics',
    'extract_attention_statistics',
    'compute_modality_alignment',
    'head_wise_attention_analysis',
]
