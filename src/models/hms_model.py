"""Multi-modal temporal graph neural network for HMS brain activity classification."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
from torch_geometric.data import Batch

from src.models.graph_layers.temporal_encoder import TemporalGraphEncoder
from src.models.graph_layers.fusion import CrossModalFusion
from src.models.graph_layers.regional_fusion import RegionalCrossModalFusion
from src.models.graph_layers.classifier import MLPClassifier


class HMSMultiModalGNN(nn.Module):
    """Multi-modal GNN for EEG and Spectrogram classification.
    
    Architecture:
    1. EEG Branch: 9 graphs → GAT → BiLSTM → features
    2. Spectrogram Branch: 119 graphs → GAT → BiLSTM → features  
    3. Cross-Modal Fusion: Bidirectional attention between modalities
    4. Classifier: MLP → 6 class predictions
    
    This model processes temporal sequences of graphs from two modalities,
    learns temporal dependencies, fuses them with cross-attention, and
    predicts one of 6 brain activity classes.
    
    Parameters
    ----------
    eeg_config : dict
        Configuration for EEG encoder
    spec_config : dict
        Configuration for spectrogram encoder
    fusion_config : dict
        Configuration for fusion module
    classifier_config : dict
        Configuration for classifier
    num_classes : int
        Number of output classes (default: 6)
    use_regional_fusion : bool
        If True, use regional cross-modal fusion (preserves spatial structure)
        If False, use simple cross-modal fusion (collapses to single vector)
    """
    
    def __init__(
        self,
        eeg_config: Optional[dict] = None,
        spec_config: Optional[dict] = None,
        fusion_config: Optional[dict] = None,
        classifier_config: Optional[dict] = None,
        num_classes: int = 6,
        use_regional_fusion: bool = True,
    ) -> None:
        super().__init__()
        
        # Default configurations
        eeg_config = eeg_config or {}
        spec_config = spec_config or {}
        fusion_config = fusion_config or {}
        classifier_config = classifier_config or {}
        
        self.use_regional_fusion = use_regional_fusion
        
        # Determine number of regions for each modality
        num_eeg_regions = eeg_config.get("num_regions", 4)
        num_spec_regions = spec_config.get("num_regions", 4)
        
        # EEG Encoder (processes 9 temporal graphs)
        self.eeg_encoder = TemporalGraphEncoder(
            in_channels=eeg_config.get("in_channels", 5),
            gat_hidden_dim=eeg_config.get("gat_hidden_dim", 64),
            gat_out_dim=eeg_config.get("gat_out_dim", 64),
            gat_num_layers=eeg_config.get("gat_num_layers", 2),
            gat_heads=eeg_config.get("gat_heads", 4),
            gat_dropout=eeg_config.get("gat_dropout", 0.3),
            use_edge_attr=eeg_config.get("use_edge_attr", True),
            rnn_hidden_dim=eeg_config.get("rnn_hidden_dim", 128),
            rnn_num_layers=eeg_config.get("rnn_num_layers", 2),
            rnn_dropout=eeg_config.get("rnn_dropout", 0.2),
            bidirectional=eeg_config.get("bidirectional", True),
            pooling_method=eeg_config.get("pooling_method", "mean"),
            channels=eeg_config.get("channels", None),
            use_hierarchical_pooling=eeg_config.get("use_hierarchical_pooling", False),
            num_regions=num_eeg_regions,
            return_regional_features=use_regional_fusion,
        )
        
        # Spectrogram Encoder (processes 119 temporal graphs)
        self.spec_encoder = TemporalGraphEncoder(
            in_channels=spec_config.get("in_channels", 4),
            gat_hidden_dim=spec_config.get("gat_hidden_dim", 64),
            gat_out_dim=spec_config.get("gat_out_dim", 64),
            gat_num_layers=spec_config.get("gat_num_layers", 2),
            gat_heads=spec_config.get("gat_heads", 4),
            gat_dropout=spec_config.get("gat_dropout", 0.3),
            use_edge_attr=spec_config.get("use_edge_attr", False),  # Spec has fixed edges
            rnn_hidden_dim=spec_config.get("rnn_hidden_dim", 128),
            rnn_num_layers=spec_config.get("rnn_num_layers", 2),
            rnn_dropout=spec_config.get("rnn_dropout", 0.2),
            bidirectional=spec_config.get("bidirectional", True),
            pooling_method=spec_config.get("pooling_method", "mean"),
            channels=spec_config.get("channels", None),
            use_hierarchical_pooling=spec_config.get("use_hierarchical_pooling", False),
            num_regions=num_spec_regions,
            return_regional_features=use_regional_fusion,
        )
        
        # Cross-Modal Fusion
        eeg_output_dim = self.eeg_encoder.output_dim
        spec_output_dim = self.spec_encoder.output_dim
        
        if use_regional_fusion:
            # Regional fusion: preserves spatial structure
            self.fusion = RegionalCrossModalFusion(
                eeg_dim=eeg_output_dim,
                spec_dim=spec_output_dim,
                num_eeg_regions=num_eeg_regions,
                num_spec_regions=num_spec_regions,
                hidden_dim=fusion_config.get("hidden_dim", 256),
                num_heads=fusion_config.get("num_heads", 8),
                dropout=fusion_config.get("dropout", 0.2),
                use_attention_pooling=fusion_config.get("use_attention_pooling", True),
            )
        else:
            # Simple fusion: collapses to single vector
            self.fusion = CrossModalFusion(
                eeg_dim=eeg_output_dim,
                spec_dim=spec_output_dim,
                hidden_dim=fusion_config.get("hidden_dim", 256),
                num_heads=fusion_config.get("num_heads", 8),
                dropout=fusion_config.get("dropout", 0.2),
            )
        
        # Classifier
        fusion_output_dim = self.fusion.output_dim
        
        self.classifier = MLPClassifier(
            input_dim=fusion_output_dim,
            hidden_dims=classifier_config.get("hidden_dims", [256, 128]),
            num_classes=num_classes,
            dropout=classifier_config.get("dropout", 0.3),
            activation=classifier_config.get("activation", "elu"),
        )
        
        self.num_classes = num_classes
    
    def forward(
        self,
        eeg_graphs: List[Batch],
        spec_graphs: List[Batch],
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Forward pass through the multi-modal model.
        
        Parameters
        ----------
        eeg_graphs : List[Batch]
            List of 9 batched EEG graphs (one per temporal window)
        spec_graphs : List[Batch]
            List of 119 batched spectrogram graphs (one per temporal window)
        return_intermediate : bool
            If True, return intermediate features for regularization
        
        Returns
        -------
        torch.Tensor or tuple
            If return_intermediate=False: Logits of shape (batch_size, num_classes)
            If return_intermediate=True: (logits, intermediate_dict) where
                intermediate_dict contains:
                - 'eeg_graphs': Input EEG graphs (for Laplacian regularization)
                - 'spec_graphs': Input Spec graphs (for Laplacian regularization)
        """
        # Encode EEG sequence
        eeg_features = self.eeg_encoder(eeg_graphs, return_sequence=False)
        # eeg_features: (batch_size, eeg_output_dim)
        
        # Encode Spectrogram sequence
        spec_features = self.spec_encoder(spec_graphs, return_sequence=False)
        # spec_features: (batch_size, spec_output_dim)
        
        # Fuse modalities with cross-attention
        fused_features = self.fusion(eeg_features, spec_features)
        # fused_features: (batch_size, fusion_output_dim)
        
        # Classify
        logits = self.classifier(fused_features)
        # logits: (batch_size, num_classes)
        
        if return_intermediate:
            # Return graphs for regularization computation
            intermediate = {
                'eeg_graphs': eeg_graphs,
                'spec_graphs': spec_graphs,
            }
            return logits, intermediate
        
        return logits
    
    def get_model_info(self) -> dict:
        """Get model architecture information.
        
        Returns
        -------
        dict
            Dictionary containing model dimensions and parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "eeg_output_dim": self.eeg_encoder.output_dim,
            "spec_output_dim": self.spec_encoder.output_dim,
            "fusion_output_dim": self.fusion.output_dim,
            "num_classes": self.num_classes,
            "use_regional_fusion": self.use_regional_fusion,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }


__all__ = ["HMSMultiModalGNN"]
