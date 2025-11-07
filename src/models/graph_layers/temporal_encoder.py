"""RNN-based temporal encoder for sequences of EEG graphs."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
from torch_geometric.data import Batch

from src.models import GATEncoder
from src.models.graph_layers.hierarchical_pooling import HierarchicalPoolingLayer


class TemporalGraphEncoder(nn.Module):
    """Encodes a sequence of temporal graphs using BiLSTM on node sequences.
    
    For each time window:
    1. Individual graph → GATEncoder → node embeddings (keeps all nodes)
    2. Pad/aggregate node sequences across batch
    3. Feed node sequences through BiLSTM for temporal modeling
    4. Pool nodes after LSTM to get window representation
    5. Return final representation (optionally with spatial structure preserved)
    
    This preserves spatial-temporal structure by letting LSTM learn across node sequences,
    not just across pooled window-level features.
    
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
        How to pool node embeddings after LSTM: 'mean', 'max', or 'sum'
    max_nodes : int
        Expected number of nodes per graph (for consistency)
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
        max_nodes: int = None,
        channels: Optional[List[str]] = None,
        use_hierarchical_pooling: bool = True,
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.gat_out_dim = gat_out_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.bidirectional = bidirectional
        self.pooling_method = pooling_method.lower()
        self.max_nodes = max_nodes
        self.channels = channels
        self.use_hierarchical_pooling = use_hierarchical_pooling
        
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
        
        # LSTM for temporal sequence modeling on node embeddings
        # Input: (batch_size * num_nodes, seq_len, gat_out_dim)
        # This processes each node's temporal evolution independently across the sequence
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
        
        # Hierarchical pooling (after LSTM)
        if use_hierarchical_pooling:
            self.hierarchical_pooling = HierarchicalPoolingLayer(
                channels=channels,
                num_regions=4,
                pooling_method=pooling_method,
            )
        else:
            self.hierarchical_pooling = None
        
        # Center-focused attention on final representations
        # If using hierarchical pooling, attention operates on regional features
        attention_input_dim = rnn_hidden_dim * 2 if bidirectional else rnn_hidden_dim
        self.center_attention = nn.Linear(attention_input_dim, 1)
        
        # Output dimension
        # If hierarchical pooling is used, output is concatenated regional features
        if use_hierarchical_pooling and channels is not None:
            self.output_dim = (rnn_hidden_dim * 2 if bidirectional else rnn_hidden_dim) * 4  # 4 regions
        else:
            self.output_dim = rnn_hidden_dim * 2 if bidirectional else rnn_hidden_dim
    
    def forward(
        self,
        graphs: List[Batch],
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Process a sequence of temporal graphs preserving node structure.
        
        Strategy:
        1. For each graph in the sequence, get node embeddings from GAT
        2. Gather all nodes across timesteps for each sample
        3. Apply LSTM to track each node's temporal evolution
        4. Pool nodes to get final representation
        
        Parameters
        ----------
        graphs : List[Batch]
            List of batched graphs, one per time window. Each batch contains
            multiple samples from the same window. Each graph should have 'is_center' attribute.
        return_sequence : bool
            If True, return full sequence output. If False, return attention-weighted representation.
        
        Returns
        -------
        torch.Tensor
            If return_sequence=False: Attention-weighted output of shape (batch_size, output_dim)
            If return_sequence=True: Full sequence of shape (batch_size, seq_len, output_dim)
        """
        if not graphs:
            raise ValueError("graphs list cannot be empty")
        
        # Get batch size and sequence length from first graph
        batch_size = graphs[0].num_graphs
        seq_len = len(graphs)
        
        # Extract is_center flags for attention weighting
        is_center_mask = torch.stack([graph.is_center[0] for graph in graphs])  # (seq_len,)
        
        # Process each graph through GAT to get node embeddings
        # Collect node embeddings for each timestep: List[seq_len] of (total_nodes, gat_out_dim)
        all_node_embeddings: List[torch.Tensor] = []
        all_graph_batch_indices: List[torch.Tensor] = []
        
        for graph in graphs:
            # Encode graph to get node embeddings
            node_embeddings = self.gat_encoder(graph)  # (num_nodes_in_batch, gat_out_dim)
            all_node_embeddings.append(node_embeddings)
            all_graph_batch_indices.append(graph.batch)  # (num_nodes_in_batch,)
        
        # Now organize node embeddings for LSTM processing
        # Goal: For each sample in batch, collect its nodes across all timesteps
        # Create tensor of shape (batch_size * num_nodes, seq_len, gat_out_dim)
        
        # First, we need to figure out num_nodes per sample
        # For EEG: 19 nodes, for Spectrogram: 4 nodes
        # Get it from the first graph
        first_graph_num_nodes = graphs[0].x.shape[0] // batch_size if graphs[0].x is not None else 0
        
        # If nodes are consistent, create a matrix for LSTM
        # Shape: (batch_size, seq_len, gat_out_dim) or (batch_size, seq_len, num_nodes, gat_out_dim)
        # We'll do a simpler approach: pool to get (batch_size, seq_len, gat_out_dim) first
        # then apply LSTM on the sequence of pooled features, but keep more information
        
        # Apply layer normalization to node embeddings
        window_embeddings_per_node: List[torch.Tensor] = []
        for node_emb in all_node_embeddings:
            node_emb = self.layer_norm(node_emb)
            window_embeddings_per_node.append(node_emb)
        
        # Strategy depends on whether we use hierarchical pooling
        if self.use_hierarchical_pooling and self.hierarchical_pooling is not None:
            # Create a structured tensor for LSTM: (batch_size, seq_len, num_regions, gat_out_dim)
            # by extracting and organizing nodes by region for each sample per timestep
            sequence_per_sample = []
            for sample_idx in range(batch_size):
                sample_sequence = []
                for t in range(seq_len):
                    node_emb = window_embeddings_per_node[t]  # (num_nodes, gat_out_dim)
                    batch_idx = all_graph_batch_indices[t]  # (num_nodes,)
                    
                    # Get nodes belonging to this sample
                    mask = batch_idx == sample_idx
                    sample_nodes = node_emb[mask]  # (num_nodes_in_sample, gat_out_dim)
                    
                    sample_sequence.append(sample_nodes)
                
                # For this sample, collect all timesteps: List[seq_len] of (num_nodes, gat_out_dim)
                sequence_per_sample.append(sample_sequence)
            
            # Now apply LSTM and hierarchical pooling per timestep
            final_representations = []
            for t in range(seq_len):
                # Collect all sample nodes at timestep t
                timestep_nodes = []
                for sample_idx in range(batch_size):
                    timestep_nodes.append(sequence_per_sample[sample_idx][t])
                
                # Pad to same size for batching (needed if node count varies)
                max_nodes_t = max(n.shape[0] for n in timestep_nodes)
                padded_nodes = []
                for nodes in timestep_nodes:
                    if nodes.shape[0] < max_nodes_t:
                        padding = torch.zeros(
                            max_nodes_t - nodes.shape[0], nodes.shape[1],
                            dtype=nodes.dtype, device=nodes.device
                        )
                        nodes = torch.cat([nodes, padding], dim=0)
                    padded_nodes.append(nodes)
                
                # Stack: (batch_size, num_nodes, gat_out_dim)
                timestep_batch = torch.stack(padded_nodes, dim=0)
                
                # Apply hierarchical pooling to get regional features
                # Output: (batch_size, num_regions, gat_out_dim)
                regional_features = self.hierarchical_pooling(timestep_batch)
                
                final_representations.append(regional_features)
            
            # Stack across timesteps: (batch_size, seq_len, num_regions, gat_out_dim)
            sequence = torch.stack(final_representations, dim=1)
            
            # Reshape to 2D for LSTM: (batch_size, seq_len, num_regions * gat_out_dim)
            bs, seq_len_actual, num_regions, feat_dim = sequence.shape
            sequence = sequence.reshape(bs, seq_len_actual, num_regions * feat_dim)
        else:
            # Original approach: simple mean pooling
            sequence_per_sample = []
            for sample_idx in range(batch_size):
                sample_sequence = []
                for t in range(seq_len):
                    node_emb = window_embeddings_per_node[t]  # (num_nodes, gat_out_dim)
                    batch_idx = all_graph_batch_indices[t]  # (num_nodes,)
                    
                    # Get nodes belonging to this sample
                    mask = batch_idx == sample_idx
                    sample_nodes = node_emb[mask]  # (num_nodes_in_sample, gat_out_dim)
                    
                    # Pool to single vector per sample per timestep
                    sample_vec = sample_nodes.mean(dim=0, keepdim=True)  # (1, gat_out_dim)
                    sample_sequence.append(sample_vec)
                
                # Stack across timesteps: (seq_len, gat_out_dim)
                sample_seq_tensor = torch.cat(sample_sequence, dim=0)
                sequence_per_sample.append(sample_seq_tensor)
            
            # Stack all samples: (batch_size, seq_len, gat_out_dim)
            sequence = torch.stack(sequence_per_sample, dim=0)
        
        # Check for NaN/Inf in sequence (safety check)
        if torch.isnan(sequence).any() or torch.isinf(sequence).any():
            sequence = torch.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply LSTM on the sequence
        lstm_out, (h_n, c_n) = self.lstm(sequence)
        # lstm_out: (batch_size, seq_len, rnn_hidden_dim * num_directions)
        
        # After LSTM, apply hierarchical pooling if enabled and not already done
        if self.use_hierarchical_pooling and self.hierarchical_pooling is not None and not isinstance(sequence, torch.Tensor):
            # Already pooled before LSTM, just use lstm_out
            final_out = lstm_out
        else:
            final_out = lstm_out
        
        if return_sequence:
            return final_out  # (batch_size, seq_len, output_dim)
        else:
            # Apply center-focused attention
            attention_scores = self.center_attention(final_out)  # (batch_size, seq_len, 1)
            attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_len)
            
            # Boost attention for center window
            center_boost = is_center_mask.float() * 0.3
            center_boost = center_boost.unsqueeze(0).expand(batch_size, -1)
            attention_scores = attention_scores + center_boost
            
            # Apply softmax
            attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
            
            # Weighted sum of LSTM outputs
            attention_weights = attention_weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
            attended_output = (final_out * attention_weights).sum(dim=1)  # (batch_size, output_dim)
            
            return attended_output

__all__ = ["TemporalGraphEncoder"]
