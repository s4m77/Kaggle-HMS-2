"""
Visualization utilities for cross-modal attention explainability.

Provides functions to visualize attention weights, heatmaps, and analysis results.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def plot_cross_modal_attention(
    attention_dict: Dict,
    sample_idx: int = 0,
    head_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot cross-modal attention as heatmaps.
    
    Shows how EEG attends to Spectrogram and vice versa.
    
    Parameters
    ----------
    attention_dict : Dict
        Dictionary with 'eeg_to_spec' and 'spec_to_eeg' attention weights
    sample_idx : int
        Which sample in batch to visualize
    head_idx : int, optional
        Which attention head to show. If None, averages all heads.
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure. If None, doesn't save.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    eeg_to_spec = attention_dict['eeg_to_spec'][sample_idx].detach().cpu().numpy()
    spec_to_eeg = attention_dict['spec_to_eeg'][sample_idx].detach().cpu().numpy()
    
    # (num_heads, query_len, key_len)
    if head_idx is not None:
        eeg_to_spec = eeg_to_spec[head_idx]
        spec_to_eeg = spec_to_eeg[head_idx]
    else:
        eeg_to_spec = eeg_to_spec.mean(axis=0)
        spec_to_eeg = spec_to_eeg.mean(axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # EEG → Spectrogram attention
    sns.heatmap(
        eeg_to_spec,
        ax=ax1,
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'},
        vmin=0,
        vmax=1,
    )
    ax1.set_title('EEG → Spectrogram Attention\n(EEG queries attending to Spec keys)')
    ax1.set_xlabel('Spectrogram Keys')
    ax1.set_ylabel('EEG Queries')
    
    # Spectrogram → EEG attention
    sns.heatmap(
        spec_to_eeg,
        ax=ax2,
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'},
        vmin=0,
        vmax=1,
    )
    ax2.set_title('Spectrogram → EEG Attention\n(Spec queries attending to EEG keys)')
    ax2.set_xlabel('EEG Keys')
    ax2.set_ylabel('Spectrogram Queries')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    title: str = "Attention Weights",
    sample_idx: int = 0,
    head_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a single attention weight matrix as a heatmap.
    
    Parameters
    ----------
    attention_weights : torch.Tensor
        Attention weights of shape (batch, num_heads, query_len, key_len)
    title : str
        Title for the plot
    sample_idx : int
        Which sample to visualize
    head_idx : int, optional
        Which head to visualize. If None, averages all heads.
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    weights_np = attention_weights[sample_idx].detach().cpu().numpy()
    
    if head_idx is not None:
        weights_np = weights_np[head_idx]
    else:
        weights_np = weights_np.mean(axis=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        weights_np,
        ax=ax,
        cmap='RdYlGn',
        cbar_kws={'label': 'Attention Weight'},
        vmin=0,
        vmax=1,
        annot=False,
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Positions')
    ax.set_ylabel('Query Positions')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_head_contributions(
    attention_dict: Dict,
    num_heads: int = 8,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot per-head attention statistics to show head specialization.
    
    Different heads often learn different patterns. This visualization
    shows which heads are more "focused" vs "diffuse" in their attention.
    
    Parameters
    ----------
    attention_dict : Dict
        Dictionary with attention weights
    num_heads : int
        Number of attention heads
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    eeg_to_spec = attention_dict['eeg_to_spec'].detach().cpu().numpy()
    spec_to_eeg = attention_dict['spec_to_eeg'].detach().cpu().numpy()
    
    # Compute per-head mean attention
    eeg_to_spec_means = []
    spec_to_eeg_means = []
    
    for head_idx in range(num_heads):
        eeg_to_spec_means.append(eeg_to_spec[:, head_idx, :, :].mean())
        spec_to_eeg_means.append(spec_to_eeg[:, head_idx, :, :].mean())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    heads = np.arange(num_heads)
    
    # EEG → Spec
    ax1.bar(heads, eeg_to_spec_means, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Head Index')
    ax1.set_ylabel('Mean Attention Weight')
    ax1.set_title('EEG → Spectrogram: Per-Head Strength')
    ax1.set_xticks(heads)
    ax1.grid(axis='y', alpha=0.3)
    
    # Spec → EEG
    ax2.bar(heads, spec_to_eeg_means, color='coral', alpha=0.7)
    ax2.set_xlabel('Head Index')
    ax2.set_ylabel('Mean Attention Weight')
    ax2.set_title('Spectrogram → EEG: Per-Head Strength')
    ax2.set_xticks(heads)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_modality_alignment(
    attention_dict: Dict,
    sample_idx: int = 0,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot modality alignment visualization.
    
    Shows:
    1. Similarity between forward and backward attention
    2. Per-head agreement scores
    3. Overall alignment metrics
    
    Parameters
    ----------
    attention_dict : Dict
        Dictionary with attention weights
    sample_idx : int
        Which sample to visualize
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    eeg_to_spec = attention_dict['eeg_to_spec'][sample_idx].detach().cpu().numpy()
    spec_to_eeg = attention_dict['spec_to_eeg'][sample_idx].detach().cpu().numpy()
    
    num_heads = eeg_to_spec.shape[0]
    
    # Compute per-head similarity (using flattened matrices)
    head_similarities = []
    for head_idx in range(num_heads):
        e2s = eeg_to_spec[head_idx].flatten()
        s2e = spec_to_eeg[head_idx].flatten()
        
        # Pad to same length
        max_len = max(len(e2s), len(s2e))
        e2s_pad = np.pad(e2s, (0, max_len - len(e2s)))
        s2e_pad = np.pad(s2e, (0, max_len - len(s2e)))
        
        # Compute correlation
        if len(e2s_pad) > 0 and len(s2e_pad) > 0:
            similarity = np.corrcoef(e2s_pad, s2e_pad)[0, 1]
        else:
            similarity = 0
        head_similarities.append(similarity)
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Per-head agreement
    ax1 = fig.add_subplot(gs[0, :])
    heads = np.arange(num_heads)
    colors = ['green' if s > 0.5 else 'orange' if s > 0 else 'red' for s in head_similarities]
    ax1.bar(heads, head_similarities, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axhline(y=0.5, color='g', linestyle='--', linewidth=0.5, alpha=0.5, label='High agreement')
    ax1.set_xlabel('Attention Head')
    ax1.set_ylabel('Forward-Backward Correlation')
    ax1.set_title('Per-Head Modality Alignment (EEG→Spec vs Spec→EEG)')
    ax1.set_xticks(heads)
    ax1.set_ylim([-1, 1])
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    
    # 2. Average EEG→Spec attention pattern
    ax2 = fig.add_subplot(gs[1, 0])
    eeg_to_spec_avg = eeg_to_spec.mean(axis=0)
    sns.heatmap(
        eeg_to_spec_avg,
        ax=ax2,
        cmap='Blues',
        cbar_kws={'label': 'Avg Weight'},
        vmin=0,
        vmax=1,
    )
    ax2.set_title('Averaged EEG→Spec Attention')
    
    # 3. Average Spec→EEG attention pattern
    ax3 = fig.add_subplot(gs[1, 1])
    spec_to_eeg_avg = spec_to_eeg.mean(axis=0)
    sns.heatmap(
        spec_to_eeg_avg,
        ax=ax3,
        cmap='Oranges',
        cbar_kws={'label': 'Avg Weight'},
        vmin=0,
        vmax=1,
    )
    ax3.set_title('Averaged Spec→EEG Attention')
    
    plt.suptitle('Modality Alignment Analysis', fontsize=14, fontweight='bold', y=1.00)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


__all__ = [
    'plot_cross_modal_attention',
    'plot_attention_heatmap',
    'plot_head_contributions',
    'plot_modality_alignment',
]
