"""MLP classifier head for final predictions."""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Multi-layer perceptron classifier with dropout and activation.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dims : List[int]
        List of hidden layer dimensions
    num_classes : int
        Number of output classes
    dropout : float
        Dropout probability
    activation : str
        Activation function ('relu', 'elu', 'gelu')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int = 6,
        dropout: float = 0.3,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Activation function
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "elu":
            self.activation = nn.ELU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classifier.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim)
        
        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes)
        """
        return self.classifier(x)


__all__ = ["MLPClassifier"]
