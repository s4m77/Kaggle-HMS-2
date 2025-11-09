"""EEG-only temporal graph neural network for HMS brain activity classification."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
from torch_geometric.data import Batch

from src.models import TemporalGraphEncoder, MLPClassifier


class HMSEEGOnlyGNN(nn.Module):
    """EEG-only GNN that mirrors the EEG branch of the multi-modal model.

    Architecture:
    1. EEG Branch: temporal sequence of EEG graphs + GAT + BiLSTM
    2. Optional regional flattening (to mimic regional fusion behaviour)
    3. Classifier: MLP -> class predictions

    Parameters
    ----------
    eeg_config : dict, optional
        Configuration for EEG encoder
    classifier_config : dict, optional
        Configuration for classifier
    num_classes : int
        Number of output classes (default: 6)
    use_regional_fusion : bool
        If True, keep regional structure (flatten regions for classifier).
        If False, fall back to single pooled vector (legacy behaviour).
    """

    def __init__(
        self,
        eeg_config: Optional[dict] = None,
        classifier_config: Optional[dict] = None,
        num_classes: int = 6,
        use_regional_fusion: bool = True,
    ) -> None:
        super().__init__()

        eeg_config = eeg_config or {}
        classifier_config = classifier_config or {}

        self.use_regional_fusion = use_regional_fusion
        self.num_classes = num_classes

        num_eeg_regions = eeg_config.get("num_regions", 4)

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

        eeg_output_dim = self.eeg_encoder.output_dim
        if use_regional_fusion:
            classifier_input_dim = num_eeg_regions * eeg_output_dim
        else:
            classifier_input_dim = eeg_output_dim

        self.num_regions = num_eeg_regions
        self.feature_dim = classifier_input_dim

        self.classifier = MLPClassifier(
            input_dim=classifier_input_dim,
            hidden_dims=classifier_config.get("hidden_dims", [256, 128]),
            num_classes=num_classes,
            dropout=classifier_config.get("dropout", 0.3),
            activation=classifier_config.get("activation", "elu"),
        )

    def forward(
        self,
        eeg_graphs: List[Batch],
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Forward pass through the EEG-only model."""
        eeg_features = self.eeg_encoder(eeg_graphs, return_sequence=False)

        if self.use_regional_fusion:
            regional_features = eeg_features
            eeg_features = regional_features.reshape(regional_features.size(0), -1)
        else:
            regional_features = None

        logits = self.classifier(eeg_features)

        if return_intermediate:
            intermediate = {"eeg_graphs": eeg_graphs}
            if regional_features is not None:
                intermediate["regional_eeg_features"] = regional_features
            return logits, intermediate

        return logits

    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "eeg_output_dim": self.eeg_encoder.output_dim,
            "feature_dim": self.feature_dim,
            "num_eeg_regions": self.num_regions,
            "use_regional_fusion": self.use_regional_fusion,
            "num_classes": self.num_classes,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }


__all__ = ["HMSEEGOnlyGNN"]
