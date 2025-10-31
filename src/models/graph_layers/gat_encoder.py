"""Graph Attention Network encoder for EEG/Spectrogram graphs."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """Multi-layer GAT encoder for graph data.
    
    Stacks multiple GAT layers with residual connections and dropout.
    
    Parameters
    ----------
    in_channels : int
        Input feature dimension (e.g., 5 for band powers)
    hidden_dim : int
        Hidden dimension for intermediate layers
    out_dim : int
        Output dimension for final embeddings
    num_layers : int
        Number of GAT layers
    heads : int
        Number of attention heads
    dropout_p : float
        Dropout probability
    use_edge_attr : bool
        Whether to use edge attributes (e.g., coherence values)
    """
    
    def __init__(
        self,
        in_channels: int = 5,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout_p: float = 0.3,
        use_edge_attr: bool = True,
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_p = dropout_p
        self.use_edge_attr = use_edge_attr
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer: in_channels -> hidden_dim * heads
        edge_dim = 1 if use_edge_attr else None
        self.convs.append(
            GATConv(
                in_channels=in_channels,
                out_channels=hidden_dim,
                heads=heads,
                dropout=dropout_p,
                edge_dim=edge_dim,
                concat=True,  # Concatenate heads
            )
        )
        self.norms.append(nn.LayerNorm(hidden_dim * heads))
        self.dropouts.append(nn.Dropout(dropout_p))
        
        # Middle layers: hidden_dim * heads -> hidden_dim * heads
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim * heads,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout_p,
                    edge_dim=edge_dim,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim * heads))
            self.dropouts.append(nn.Dropout(dropout_p))
        
        # Last layer: hidden_dim * heads -> out_dim (average heads)
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim * heads,
                    out_channels=out_dim,
                    heads=heads,
                    dropout=dropout_p,
                    edge_dim=edge_dim,
                    concat=False,  # Average heads for final layer
                )
            )
            self.norms.append(nn.LayerNorm(out_dim))
            self.dropouts.append(nn.Dropout(dropout_p))
        
        self.activation = nn.ELU()
    
    def forward(self, data: Batch) -> torch.Tensor:
        """Encode graph to node embeddings.
        
        Parameters
        ----------
        data : Batch
            Batched graph data with x, edge_index, and optionally edge_attr
        
        Returns
        -------
        torch.Tensor
            Node embeddings of shape (num_nodes, out_dim)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if self.use_edge_attr and hasattr(data, 'edge_attr') else None
        
        # Apply GAT layers
        for i, (conv, norm, dropout) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            x_new = conv(x, edge_index, edge_attr=edge_attr)
            x_new = norm(x_new)
            
            # Skip connection (if dimensions match)
            if i > 0 and x.shape[-1] == x_new.shape[-1]:
                x_new = x_new + x
            
            x = self.activation(x_new)
            x = dropout(x)
        
        return x


__all__ = ["GATEncoder"]
