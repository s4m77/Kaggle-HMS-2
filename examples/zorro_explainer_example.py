"""Example usage of the ZORRO explainer for HMS model."""

from typing import List, Dict, Optional
import torch
from torch_geometric.data import Batch

from src.models import HMSMultiModalGNN
from src.models.zorro_explainer import ZORROExplainer, ZORROExplanation


def explain_hms_predictions(
    model: HMSMultiModalGNN,
    eeg_graphs: List[Batch],
    spec_graphs: List[Batch],
    sample_indices: Optional[List[int]] = None,
    top_k: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[int, Dict[str, ZORROExplanation]]:
    """Explain HMS model predictions using ZORRO.
    
    Parameters
    ----------
    model : HMSMultiModalGNN
        Trained multi-modal GNN model
    eeg_graphs : List[Batch]
        List of 9 batched EEG graphs
    spec_graphs : List[Batch]
        List of 119 batched spectrogram graphs
    sample_indices : List[int], optional
        Indices of samples to explain. If None, explains all samples in batch.
    top_k : int
        Number of most important nodes to return
    device : torch.device, optional
        Device to run computations on
    
    Returns
    -------
    Dict[int, Dict[str, ZORROExplanation]]
        Dictionary mapping sample_idx -> {"eeg": explanation, "spec": explanation}
    
    Example
    -------
    >>> model = HMSMultiModalGNN(...)
    >>> model.load_state_dict(torch.load("best_model.pt"))
    >>> model.to(device)
    >>> 
    >>> # Explain predictions for batch
    >>> explanations = explain_hms_predictions(
    ...     model=model,
    ...     eeg_graphs=eeg_graphs,
    ...     spec_graphs=spec_graphs,
    ...     sample_indices=[0, 1, 2],
    ...     top_k=15,
    ...     device=device,
    ... )
    >>> 
    >>> # Access EEG explanation for sample 0
    >>> eeg_exp = explanations[0]["eeg"]
    >>> print(f"Top EEG nodes: {eeg_exp.top_k_nodes[:5]}")
    >>> print(f"Feature importance: {eeg_exp.feature_importance}")
    >>> 
    >>> # Access spectrogram explanation for sample 0
    >>> spec_exp = explanations[0]["spec"]
    >>> print(f"Top spectrogram nodes: {spec_exp.top_k_nodes[:5]}")
    """
    
    if device is None:
        device = next(model.parameters()).device
    
    # Create explainer
    explainer = ZORROExplainer(
        model=model,
        target_class=None,  # Will use predicted class
        device=device,
        perturbation_mode="zero",  # Can also use "noise" or "mean"
        noise_std=0.1,
    )
    
    batch_size = eeg_graphs[0].num_graphs
    
    if sample_indices is None:
        sample_indices = list(range(batch_size))
    
    # Explain batch
    all_explanations = {}
    
    for sample_idx in sample_indices:
        eeg_exp = explainer.explain_sample(
            graphs=eeg_graphs,
            modality="eeg",
            sample_idx=sample_idx,
            top_k=top_k,
            n_samples=5,  # Monte Carlo samples
            pbar=True,
        )
        
        spec_exp = explainer.explain_sample(
            graphs=spec_graphs,
            modality="spec",
            sample_idx=sample_idx,
            top_k=top_k,
            n_samples=5,
            pbar=True,
        )
        
        all_explanations[sample_idx] = {
            "eeg": eeg_exp,
            "spec": spec_exp,
        }
    
    return all_explanations


def print_explanation(explanation: ZORROExplanation, modality_name: str = "Modality") -> None:
    """Pretty print a ZORRO explanation.
    
    Parameters
    ----------
    explanation : ZORROExplanation
        The explanation to print
    modality_name : str
        Name to use in output (e.g., "EEG" or "Spectrogram")
    """
    print(f"\n{'='*60}")
    print(f"{modality_name} Explanation Summary")
    print(f"{'='*60}")
    
    print(f"\nOriginal Prediction Shape: {explanation.prediction_original.shape}")
    print(f"Predicted Class: {explanation.prediction_original.argmax().item()}")
    print(f"Prediction Logits: {explanation.prediction_original}")
    
    print(f"\n--- Top {len(explanation.top_k_nodes)} Important Nodes ---")
    for rank, (node_idx, importance) in enumerate(explanation.top_k_nodes, 1):
        print(f"  {rank:2d}. Node {node_idx:3d}: importance = {importance:.4f}")
    
    print(f"\n--- Feature Importance (Top 10) ---")
    top_features = torch.topk(explanation.feature_importance, k=min(10, len(explanation.feature_importance)))
    for rank, (feat_idx, importance) in enumerate(zip(top_features.indices, top_features.values), 1):
        print(f"  {rank:2d}. Feature {feat_idx:2d}: importance = {importance:.4f}")
    
    print(f"\nNode Importance Shape: {explanation.node_importance.shape}")
    print(f"  ({explanation.node_importance.shape[0]} nodes × {explanation.node_importance.shape[1]} features)")


def compare_modalities(
    eeg_exp: ZORROExplanation,
    spec_exp: ZORROExplanation,
) -> None:
    """Compare explanations across modalities.
    
    Parameters
    ----------
    eeg_exp : ZORROExplanation
        EEG modality explanation
    spec_exp : ZORROExplanation
        Spectrogram modality explanation
    """
    print(f"\n{'='*60}")
    print("Modality Comparison")
    print(f"{'='*60}")
    
    eeg_total_importance = eeg_exp.node_importance.sum().item()
    spec_total_importance = spec_exp.node_importance.sum().item()
    
    print(f"\nTotal Node Importance:")
    print(f"  EEG:         {eeg_total_importance:.4f}")
    print(f"  Spectrogram: {spec_total_importance:.4f}")
    
    eeg_max_importance = eeg_exp.node_importance.max().item()
    spec_max_importance = spec_exp.node_importance.max().item()
    
    print(f"\nMax Node Importance:")
    print(f"  EEG:         {eeg_max_importance:.4f}")
    print(f"  Spectrogram: {spec_max_importance:.4f}")
    
    print(f"\nMean Feature Importance:")
    print(f"  EEG:         {eeg_exp.feature_importance.mean():.4f}")
    print(f"  Spectrogram: {spec_exp.feature_importance.mean():.4f}")
    
    print(f"\nNumber of Nodes:")
    print(f"  EEG:         {len(eeg_exp.node_indices)}")
    print(f"  Spectrogram: {len(spec_exp.node_indices)}")


if __name__ == "__main__":
    # Example usage
    print("ZORRO Explainer Example")
    print("This module provides utilities to explain HMS model predictions.")
    print("\nSee the docstrings for usage examples.")
