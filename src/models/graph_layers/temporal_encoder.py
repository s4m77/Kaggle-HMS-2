"""RNN-based temporal encoder for sequences of EEG graphs."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from src.models.gat_encoder import GATEncoder


class TemporalGraphEncoder(nn.Module):
    """Encodes a sequence of temporal graphs using BiLSTM.
    
    For each time window:
    1. Individual graph → GATEncoder → node embeddings
    2. Pool nodes to get window embedding
    3. Feed sequence through BiLSTM
    4. Return final hidden state as sequence representation
    
    Parameters
    ----------
    in_channels : int
        Input node feature dimension (e.g., 5 for band powers)
    gat_hidden_dim : int
        Hidden dimension for GAT layers
    gat_out_dim : int
        Output dimension for GAT (embedding dimension per node)
    gat_num_layers : int
        Number of GAT layers
    gat_heads : int
        Number of attention heads in GAT
    gat_dropout : float
        Dropout probability in GAT
    use_edge_attr : bool
        Whether GAT uses edge attributes
    rnn_hidden_dim : int
        Hidden dimension for LSTM (output will be 2x this if bidirectional)
    rnn_num_layers : int
        Number of LSTM layers
    rnn_dropout : float
        Dropout probability in LSTM (only used if num_layers > 1)
    bidirectional : bool
        Whether to use bidirectional LSTM
    pooling_method : str
        How to pool node embeddings: 'mean', 'max', or 'sum'
    """

    def __init__(
        self,
        in_channels: int = 5,
        gat_hidden_dim: int = 64,
        gat_out_dim: int = 64,
        gat_num_layers: int = 2,
        gat_heads: int = 4,
        gat_dropout: float = 0.3,
        use_edge_attr: bool = True,
        rnn_hidden_dim: int = 128,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0.2,
        bidirectional: bool = True,
        pooling_method: str = "mean",
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.gat_out_dim = gat_out_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.bidirectional = bidirectional
        self.pooling_method = pooling_method.lower()
        
        if self.pooling_method not in ("mean", "max", "sum"):
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")
        
        # GAT encoder for individual graphs
        self.gat_encoder = GATEncoder(
            in_channels=in_channels,
            hidden_dim=gat_hidden_dim,
            out_dim=gat_out_dim,
            num_layers=gat_num_layers,
            heads=gat_heads,
            dropout_p=gat_dropout,
            use_edge_attr=use_edge_attr,
        )
        
        # LSTM for temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=gat_out_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(gat_out_dim)
        
        # Output dimension
        self.output_dim = rnn_hidden_dim * 2 if bidirectional else rnn_hidden_dim
    
    def forward(
        self,
        graphs: List[Batch],
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Process a list of temporal graphs and return encoded representation.
        
        Parameters
        ----------
        graphs : List[Batch]
            List of batched graphs, one per time window. Each batch contains
            multiple samples from the same window.
        return_sequence : bool
            If True, return full sequence output. If False, return only final hidden state.
        
        Returns
        -------
        torch.Tensor
            If return_sequence=False: Final LSTM hidden state of shape (batch_size, output_dim)
            If return_sequence=True: Full sequence of shape (batch_size, seq_len, output_dim)
        """
        if not graphs:
            raise ValueError("graphs list cannot be empty")
        
        # Get batch size from first graph
        batch_size = graphs[0].num_graphs
        
        # Process each temporal window through GAT
        window_embeddings: List[torch.Tensor] = []
        for graph in graphs:
            # Encode graph to get node embeddings
            node_embeddings = self.gat_encoder(graph)  # (num_nodes, gat_out_dim)
            
            # Pool nodes to get window representation
            window_emb = self._pool_nodes(node_embeddings, graph)  # (batch_size, gat_out_dim)
            window_embeddings.append(window_emb)
        
        # Stack windows into sequence: (batch_size, seq_len, gat_out_dim)
        sequence = torch.stack(window_embeddings, dim=1)
        
        # Apply layer normalization
        sequence = self.layer_norm(sequence)
        
        # Apply LSTM
        lstm_out, (h_n, c_n) = self.lstm(sequence)
        # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        # h_n: (num_layers * num_directions, batch_size, hidden_dim)
        
        if return_sequence:
            return lstm_out  # (batch_size, seq_len, output_dim)
        else:
            # Extract final hidden state
            if self.bidirectional:
                # Concatenate forward and backward from last layer
                final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch_size, 2*hidden_dim)
            else:
                final_hidden = h_n[-1]  # (batch_size, hidden_dim)
            
            return final_hidden  # (batch_size, output_dim)
    
    def _pool_nodes(self, node_embeddings: torch.Tensor, graph: Batch) -> torch.Tensor:
        """Pool node embeddings to graph-level embeddings using PyG pooling.
        
        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings of shape (num_nodes, gat_out_dim)
        graph : Batch
            Batched graph data containing batch assignment
        
        Returns
        -------
        torch.Tensor
            Pooled embeddings of shape (batch_size, gat_out_dim)
        """
        if self.pooling_method == "mean":
            return global_mean_pool(node_embeddings, graph.batch)
        elif self.pooling_method == "max":
            return global_max_pool(node_embeddings, graph.batch)
        elif self.pooling_method == "sum":
            return global_add_pool(node_embeddings, graph.batch)


__all__ = ["TemporalGraphEncoder"]
