"""Tests for graph regularization functions."""

import torch
import pytest
from torch_geometric.data import Data, Batch

from src.models.regularization import (
    graph_laplacian_regularization,
    edge_weight_regularization,
    compute_graph_regularization,
)


def test_graph_laplacian_basic():
    """Test basic Graph Laplacian regularization."""
    # Create simple graph: 4 nodes, 4 edges
    x = torch.randn(4, 10)  # 4 nodes, 10 features
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    loss = graph_laplacian_regularization(x, edge_index)
    
    assert loss.ndim == 0  # Scalar
    assert loss.item() >= 0  # Non-negative
    print(f"✓ Graph Laplacian loss: {loss.item():.6f}")


def test_graph_laplacian_smooth_features():
    """Test that smooth features have lower regularization loss."""
    # Create smooth features (connected nodes are similar)
    x_smooth = torch.tensor([
        [1.0, 0.0],
        [1.1, 0.1],  # Similar to node 0
        [1.0, 0.0],
        [0.9, -0.1], # Similar to node 0
    ])
    
    # Create rough features (connected nodes are different)
    x_rough = torch.tensor([
        [1.0, 0.0],
        [-1.0, 1.0],  # Very different from node 0
        [1.0, 0.0],
        [0.0, -1.0],  # Very different from node 0
    ])
    
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    loss_smooth = graph_laplacian_regularization(x_smooth, edge_index)
    loss_rough = graph_laplacian_regularization(x_rough, edge_index)
    
    assert loss_smooth < loss_rough, "Smooth features should have lower loss"
    print(f"✓ Smooth loss: {loss_smooth.item():.6f} < Rough loss: {loss_rough.item():.6f}")


def test_graph_laplacian_empty_edges():
    """Test that empty edge set returns zero loss."""
    x = torch.randn(4, 10)
    edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
    
    loss = graph_laplacian_regularization(x, edge_index)
    
    assert loss.item() == 0.0
    print("✓ Empty edge set returns zero loss")


def test_edge_weight_regularization():
    """Test edge weight regularization."""
    # Create edge attributes
    edge_attr = torch.randn(10, 1)  # 10 edges, 1 feature
    
    # L2 penalty
    loss_l2 = edge_weight_regularization(edge_attr, penalty_type="l2")
    assert loss_l2.item() >= 0
    
    # L1 penalty
    loss_l1 = edge_weight_regularization(edge_attr, penalty_type="l1")
    assert loss_l1.item() >= 0
    
    print(f"✓ Edge L2 penalty: {loss_l2.item():.6f}")
    print(f"✓ Edge L1 penalty: {loss_l1.item():.6f}")


def test_compute_graph_regularization_batch():
    """Test combined regularization on batch of graphs."""
    # Create 3 graphs
    graphs = []
    for i in range(3):
        x = torch.randn(5, 8)  # 5 nodes, 8 features
        edge_index = torch.randint(0, 5, (2, 6))  # 6 edges
        edge_attr = torch.randn(6, 1)
        batch_vec = torch.zeros(5, dtype=torch.long)  # Single graph
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_vec)
        graphs.append(data)
    
    # Batch the graphs
    batch_graphs = [Batch.from_data_list([g]) for g in graphs]
    
    # Compute regularization
    reg_loss = compute_graph_regularization(
        batch_graphs=batch_graphs,
        lambda_laplacian=0.001,
        lambda_edge=0.0001,
    )
    
    assert reg_loss.ndim == 0
    assert reg_loss.item() >= 0
    print(f"✓ Combined regularization loss: {reg_loss.item():.6f}")


def test_regularization_values():
    """Test that regularization has reasonable magnitude."""
    # Simulate EEG-like graph: 19 nodes, 5 features
    x = torch.randn(19, 5)
    edge_index = torch.randint(0, 19, (2, 50))  # 50 edges
    
    loss = graph_laplacian_regularization(x, edge_index, normalize=True)
    
    # Normalized loss should be reasonable (typically < 1.0 for random features)
    assert 0 < loss.item() < 10.0, f"Loss {loss.item()} outside expected range"
    print(f"✓ Regularization magnitude reasonable: {loss.item():.6f}")


def test_backward_pass():
    """Test that gradients flow through regularization."""
    x = torch.randn(10, 5, requires_grad=True)
    edge_index = torch.randint(0, 10, (2, 20))
    
    loss = graph_laplacian_regularization(x, edge_index)
    loss.backward()
    
    assert x.grad is not None
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    print("✓ Gradients flow correctly through regularization")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Graph Regularization Functions")
    print("="*60 + "\n")
    
    test_graph_laplacian_basic()
    test_graph_laplacian_smooth_features()
    test_graph_laplacian_empty_edges()
    test_edge_weight_regularization()
    test_compute_graph_regularization_batch()
    test_regularization_values()
    test_backward_pass()
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60 + "\n")
