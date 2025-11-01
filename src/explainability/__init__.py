"""
Explainability module for HMS Multi-Modal GNN.

Provides tools for interpreting cross-modal attention mechanisms and
visualizing how the model reconciles EEG and Spectrogram modalities.
"""

from .attention_analysis import (
    AttentionAnalyzer,
    extract_attention_statistics,
    compute_modality_alignment,
    head_wise_attention_analysis,
)
from .visualizations import (
    plot_cross_modal_attention,
    plot_attention_heatmap,
    plot_head_contributions,
    plot_modality_alignment,
)

__all__ = [
    "AttentionAnalyzer",
    "extract_attention_statistics",
    "compute_modality_alignment",
    "head_wise_attention_analysis",
    "plot_cross_modal_attention",
    "plot_attention_heatmap",
    "plot_head_contributions",
    "plot_modality_alignment",
]
