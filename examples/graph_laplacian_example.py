"""Example of how to use Graph Laplacian regularization in training loop."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import HMSMultiModalGNN, compute_graph_regularization


def training_step_with_regularization(
    model: HMSMultiModalGNN,
    batch: dict,
    criterion: nn.Module,
    lambda_laplacian: float = 0.001,
    lambda_edge: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Training step with Graph Laplacian regularization.
    
    Parameters
    ----------
    model : HMSMultiModalGNN
        The model to train
    batch : dict
        Batch from DataLoader with keys:
        - 'eeg_graphs': List[Batch] - 9 EEG graphs
        - 'spec_graphs': List[Batch] - 119 Spec graphs
        - 'targets': Tensor - Ground truth labels
    criterion : nn.Module
        Loss function (e.g., CrossEntropyLoss)
    lambda_laplacian : float
        Weight for Graph Laplacian regularization
    lambda_edge : float
        Weight for edge regularization (0 = disabled)
    
    Returns
    -------
    total_loss : Tensor
        Total loss (CE + regularization)
    loss_dict : dict
        Dictionary with individual loss components for logging
    """
    eeg_graphs = batch['eeg_graphs']
    spec_graphs = batch['spec_graphs']
    targets = batch['targets']
    
    # Forward pass with intermediate outputs
    logits, intermediate = model(
        eeg_graphs, 
        spec_graphs,
        return_intermediate=True
    )
    
    # Classification loss
    ce_loss = criterion(logits, targets)
    
    # Graph Laplacian regularization on EEG graphs
    eeg_reg_loss = compute_graph_regularization(
        batch_graphs=intermediate['eeg_graphs'],
        node_features_list=None,  # Use batch.x
        lambda_laplacian=lambda_laplacian,
        lambda_edge=lambda_edge,
    )
    
    # Graph Laplacian regularization on Spectrogram graphs
    spec_reg_loss = compute_graph_regularization(
        batch_graphs=intermediate['spec_graphs'],
        node_features_list=None,  # Use batch.x
        lambda_laplacian=lambda_laplacian,
        lambda_edge=lambda_edge,
    )
    
    # Total regularization
    reg_loss = eeg_reg_loss + spec_reg_loss
    
    # Total loss
    total_loss = ce_loss + reg_loss
    
    # Loss components for logging
    loss_dict = {
        'loss': total_loss.item(),
        'ce_loss': ce_loss.item(),
        'eeg_reg_loss': eeg_reg_loss.item(),
        'spec_reg_loss': spec_reg_loss.item(),
        'reg_loss': reg_loss.item(),
    }
    
    return total_loss, loss_dict


def example_training_loop():
    """Example training loop with Graph Laplacian regularization."""
    
    # Initialize model
    model = HMSMultiModalGNN()
    model.train()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001  # Standard L2 regularization
    )
    
    # Hyperparameters for Graph Laplacian
    lambda_laplacian = 0.001  # From config
    lambda_edge = 0.0  # Disabled by default
    
    # Training loop (pseudo-code)
    # dataloader = DataLoader(...)
    # for epoch in range(num_epochs):
    #     for batch in dataloader:
    #         optimizer.zero_grad()
    #         
    #         # Training step with regularization
    #         total_loss, loss_dict = training_step_with_regularization(
    #             model=model,
    #             batch=batch,
    #             criterion=criterion,
    #             lambda_laplacian=lambda_laplacian,
    #             lambda_edge=lambda_edge,
    #         )
    #         
    #         # Backward and optimize
    #         total_loss.backward()
    #         optimizer.step()
    #         
    #         # Log losses
    #         print(f"Epoch {epoch}, Loss: {loss_dict['loss']:.4f}, "
    #               f"CE: {loss_dict['ce_loss']:.4f}, "
    #               f"Reg: {loss_dict['reg_loss']:.4f}")
    
    print("Graph Laplacian regularization configured successfully!")
    print(f"  λ_laplacian = {lambda_laplacian}")
    print(f"  λ_edge = {lambda_edge}")


if __name__ == "__main__":
    example_training_loop()
