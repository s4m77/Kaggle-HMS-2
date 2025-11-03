"""EEG-only temporal graph neural network for HMS brain activity classification."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
from torch_geometric.data import Batch

from src.models import TemporalGraphEncoder, MLPClassifier


class HMSEEGOnlyGNN(nn.Module):
    """GNN that processes only EEG temporal graph sequences.

    Architecture:
    1. EEG Branch: temporal sequence of EEG graphs → GAT → BiLSTM → features
    2. Classifier: MLP → class predictions

    Parameters
    ----------
    eeg_config : dict, optional
        Configuration for EEG encoder
    classifier_config : dict, optional
        Configuration for classifier
    num_classes : int
        Number of output classes (default: 6)
    """

    def __init__(
        self,
        eeg_config: Optional[dict] = None,
        classifier_config: Optional[dict] = None,
        num_classes: int = 6,
    ) -> None:
        super().__init__()

        eeg_config = eeg_config or {}
        classifier_config = classifier_config or {}

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
        )

        eeg_output_dim = self.eeg_encoder.output_dim

        self.classifier = MLPClassifier(
            input_dim=eeg_output_dim,
            hidden_dims=classifier_config.get("hidden_dims", [256, 128]),
            num_classes=num_classes,
            dropout=classifier_config.get("dropout", 0.3),
            activation=classifier_config.get("activation", "elu"),
        )

        self.num_classes = num_classes

    def forward(
        self,
        eeg_graphs: List[Batch],
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Forward pass through the EEG-only model."""
        eeg_features = self.eeg_encoder(eeg_graphs, return_sequence=False)
        logits = self.classifier(eeg_features)

        if return_intermediate:
            intermediate = {"eeg_graphs": eeg_graphs}
            return logits, intermediate

        return logits

    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "eeg_output_dim": self.eeg_encoder.output_dim,
            "num_classes": self.num_classes,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }


__all__ = ["HMSEEGOnlyGNN"]
