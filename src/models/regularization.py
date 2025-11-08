"""Graph regularization techniques for GNNs."""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Batch


def graph_laplacian_regularization(
    x: Tensor,
    edge_index: Tensor,
    batch: Tensor | None = None,
    normalize: bool = True,
) -> Tensor:
    """Compute Graph Laplacian regularization (smoothness prior).
    
    Encourages connected nodes to have similar feature representations.
    This is particularly useful for brain networks where spatially connected
    regions tend to have correlated activity.
    
    The regularization term is:
        L_smooth = (1/|E|) * Σ_(i,j)∈E ||x_i - x_j||²
    With normalize=True, we further divide by the feature dimension so the
    magnitude is roughly independent of the number of features.
    
    Where E is the set of edges and x_i, x_j are node features.
    
    Parameters
    ----------
    x : Tensor
        Node features of shape (num_nodes, num_features)
    edge_index : Tensor
        Edge indices of shape (2, num_edges)
    batch : Tensor, optional
        Batch assignment for each node. If provided, normalizes per graph.
    normalize : bool
        Whether to normalize by number of edges (default: True)
    
    Returns
    -------
    Tensor
        Scalar regularization loss
        
    Examples
    --------
    >>> x = torch.randn(100, 64)  # 100 nodes, 64 features
    >>> edge_index = torch.randint(0, 100, (2, 200))  # 200 edges
    >>> loss = graph_laplacian_regularization(x, edge_index)
    >>> total_loss = ce_loss + 0.001 * loss
    """
    if edge_index.size(1) == 0:
        # No edges, return zero loss
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    
    # Get source and target nodes
    row, col = edge_index[0], edge_index[1]
    
    # Compute feature differences between connected nodes
    diff = x[row] - x[col]
    
    # L2 norm of differences
    smoothness_loss = (diff ** 2).sum()
    
    if normalize:
        # Normalize by number of edges and feature dimension to keep scale stable
        num_edges = max(int(edge_index.size(1)), 1)
        num_feats = max(int(x.size(1)) if x.dim() > 1 else 1, 1)
        smoothness_loss = smoothness_loss / (num_edges * num_feats)
    
    return smoothness_loss


def edge_weight_regularization(
    edge_attr: Tensor,
    penalty_type: str = "l2",
    normalize: bool = True,
) -> Tensor:
    """Regularize edge attributes (weights).
    
    Prevents edge weights from becoming too large, which can happen
    with coherence-based edges in brain networks.
    
    Parameters
    ----------
    edge_attr : Tensor
        Edge attributes/weights of shape (num_edges, num_edge_features)
    penalty_type : str
        Type of penalty: 'l2' (Ridge) or 'l1' (Lasso)
    normalize : bool
        Whether to normalize by number of edges
    
    Returns
    -------
    Tensor
        Scalar regularization loss
    """
    if edge_attr is None or edge_attr.numel() == 0:
        return torch.tensor(0.0, device=edge_attr.device if edge_attr is not None else 'cpu')
    
    if penalty_type.lower() == "l2":
        penalty = (edge_attr ** 2).sum()
    elif penalty_type.lower() == "l1":
        penalty = edge_attr.abs().sum()
    else:
        raise ValueError(f"Unknown penalty_type: {penalty_type}. Use 'l2' or 'l1'.")
    
    if normalize:
        penalty = penalty / edge_attr.numel()
    
    return penalty


def compute_graph_regularization(
    batch_graphs: list[Batch],
    node_features_list: list[Tensor] | None = None,
    lambda_laplacian: float = 0.001,
    lambda_edge: float = 0.0,
) -> Tensor:
    """Compute combined graph regularization for a batch of graphs.
    
    This is a convenience function that computes both Laplacian and edge
    regularization for a list of batched graphs (e.g., temporal sequence).
    
    Parameters
    ----------
    batch_graphs : list[Batch]
        List of PyG Batch objects (e.g., temporal sequence)
    node_features_list : list[Tensor], optional
        List of node features for each graph. If None, uses batch_graphs[i].x
    lambda_laplacian : float
        Weight for Laplacian regularization (0 = disabled)
    lambda_edge : float
        Weight for edge regularization (0 = disabled)
    
    Returns
    -------
    Tensor
        Total regularization loss (scalar)
        
    Examples
    --------
    >>> # In training loop
    >>> eeg_features = model.eeg_encoder(eeg_graphs, return_sequence=True)
    >>> reg_loss = compute_graph_regularization(
    ...     eeg_graphs, 
    ...     eeg_features,
    ...     lambda_laplacian=0.001
    ... )
    >>> total_loss = ce_loss + reg_loss
    """
    total_loss = 0.0
    
    for i, batch in enumerate(batch_graphs):
        # Use provided features or fall back to batch.x
        if node_features_list is not None:
            x = node_features_list[i]
        else:
            x = batch.x
        
        # Laplacian regularization
        if lambda_laplacian > 0:
            laplacian_loss = graph_laplacian_regularization(
                x=x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                normalize=True,
            )
            total_loss = total_loss + lambda_laplacian * laplacian_loss
        
        # Edge weight regularization
        if lambda_edge > 0 and hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            edge_loss = edge_weight_regularization(
                edge_attr=batch.edge_attr,
                penalty_type="l2",
                normalize=True,
            )
            total_loss = total_loss + lambda_edge * edge_loss
    
    return total_loss


__all__ = [
    "graph_laplacian_regularization",
    "edge_weight_regularization", 
    "compute_graph_regularization",
]
