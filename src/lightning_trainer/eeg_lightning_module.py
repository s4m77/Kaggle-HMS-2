"""PyTorch Lightning module for EEG-only HMS GNN."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MetricCollection

from src.models import HMSEEGOnlyGNN
from src.models.regularization import compute_graph_regularization


class HMSEEGOnlyLightningModule(LightningModule):
    """Lightning wrapper for the EEG-only HMS GNN.

    Handles:
    - Training, validation, and test steps
    - Loss computation with optional class weights (CrossEntropy) or KL divergence
    - Graph Laplacian regularization (EEG graphs only)
    - Metrics for vote prediction:
        * KL Divergence (loss) - Primary Kaggle metric
        * Accuracy - Consensus class correctness
        * MSE - Vote distribution error
        * Total Variation Distance - Interpretable vote error
        * Per-Class MSE - Class-specific error analysis
    - WandB logging
    - Learning rate scheduling
    - Optimizer configuration

    Parameters
    ----------
    model_config : dict
        Model configuration (eeg_encoder, classifier)
    num_classes : int
        Number of output classes (default: 6)
    learning_rate : float
        Learning rate for optimizer
    weight_decay : float
        Weight decay for optimizer (L2 regularization)
    class_weights : torch.Tensor, optional
        Class weights for loss balancing
    scheduler_config : dict, optional
        Learning rate scheduler configuration
    graph_laplacian_lambda : float
        Graph Laplacian regularization weight (0 = disabled)
    edge_weight_penalty : float
        Edge weight regularization penalty (0 = disabled)
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        num_classes: int = 6,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        graph_laplacian_lambda: float = 0.0,
        edge_weight_penalty: float = 0.0,
        loss_type: str = 'CrossEntropyLoss',
    ) -> None:
        super().__init__()

        # Save hyperparameters (will be logged to WandB)
        self.save_hyperparameters(ignore=['class_weights'])

        # Initialize model
        self.model = HMSEEGOnlyGNN(
            eeg_config=model_config.get('eeg_encoder'),
            classifier_config=model_config.get('classifier'),
            num_classes=num_classes,
            use_regional_fusion=model_config.get('use_regional_fusion', True),
        )

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        self.graph_laplacian_lambda = graph_laplacian_lambda
        self.edge_weight_penalty = edge_weight_penalty
        self.loss_type = loss_type

        # Loss function
        if loss_type == 'KLDivLoss':
            # KL divergence expects log probabilities as input
            self.criterion = nn.KLDivLoss(reduction='batchmean')
            self.class_weights = None
        elif class_weights is not None:
            self.register_buffer('class_weights', class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.class_weights = None
            self.criterion = nn.CrossEntropyLoss()

        # Metrics for each stage
        # Note: For vote prediction, accuracy measures if argmax(pred) == argmax(true_votes)
        metrics = MetricCollection({
            'acc': Accuracy(task='multiclass', num_classes=num_classes, average='micro'),
        })

        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

    def forward(self, eeg_graphs):
        """Forward pass through the EEG-only model."""
        return self.model(eeg_graphs)

    def _step(self, batch: Dict[str, Any], batch_idx: int, stage: str):
        """Shared step for train/val/test."""
        eeg_graphs = batch['eeg_graphs']
        targets = batch['targets']

        # Get batch size for proper metric logging
        batch_size = targets.size(0)

        # Forward pass (with intermediate outputs for regularization during training)
        if stage == 'train' and (self.graph_laplacian_lambda > 0 or self.edge_weight_penalty > 0):
            logits, intermediate = self.model(eeg_graphs, return_intermediate=True)
        else:
            logits = self.model(eeg_graphs)
            intermediate = None

        # Compute classification loss
        if self.loss_type == 'KLDivLoss':
            # For KL divergence:
            # - logits are raw model outputs
            # - targets should be probability distributions (sum to 1)
            # - criterion expects log probabilities as input and target probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Ensure targets are probabilities (normalize if they're vote counts)
            if targets.dim() == 1:
                # If targets are class indices, convert to one-hot (hard targets)
                target_probs = F.one_hot(targets, num_classes=self.num_classes).float()
            else:
                # If targets are already distributions (vote counts), normalize
                target_probs = targets / targets.sum(dim=-1, keepdim=True)

            ce_loss = self.criterion(log_probs, target_probs)
        else:
            # CrossEntropyLoss expects logits and class indices
            ce_loss = self.criterion(logits, targets)
            target_probs = None

        # Safety check for NaN in loss
        if torch.isnan(ce_loss):
            print(f"WARNING: NaN in loss at batch {batch_idx}")
            print(f"  Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
            print(f"  Targets: {targets}")
            print(f"  Has NaN in logits: {torch.isnan(logits).any()}")
            print(f"  Has Inf in logits: {torch.isinf(logits).any()}")
            # Skip this batch to prevent NaN propagation
            return None

        # Add graph regularization (only during training)
        if stage == 'train' and intermediate is not None:
            eeg_reg = compute_graph_regularization(
                intermediate['eeg_graphs'],
                lambda_laplacian=self.graph_laplacian_lambda,
                lambda_edge=self.edge_weight_penalty,
            )

            # Safety check for NaN in regularization
            if torch.isnan(eeg_reg):
                print(f"WARNING: NaN in regularization loss at batch {batch_idx}")
                eeg_reg = torch.tensor(0.0, device=ce_loss.device)

            total_loss = ce_loss + eeg_reg

            # Log individual components
            # Note: on_step=True creates {metric}_step, on_epoch=True creates {metric}_epoch
            self.log(f'{stage}/ce_loss', ce_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log(f'{stage}/reg_loss', eeg_reg, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log(f'{stage}/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
            loss = total_loss
        else:
            # No regularization for val/test
            self.log(f'{stage}/loss', ce_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
            loss = ce_loss

        # Get predictions
        preds = torch.argmax(logits, dim=1)

        # For KL divergence with vote distributions, get ground truth class for metrics
        if self.loss_type == 'KLDivLoss' and targets.dim() > 1:
            target_classes = torch.argmax(targets, dim=1)
        else:
            target_classes = targets

        # Update classification metrics
        if stage == 'train':
            metrics = self.train_metrics(preds, target_classes)
        elif stage == 'val':
            metrics = self.val_metrics(preds, target_classes)
        else:
            metrics = self.test_metrics(preds, target_classes)

        # Log classification metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        # Additional metrics for vote prediction (KL Divergence)
        if self.loss_type == 'KLDivLoss' and targets.dim() > 1:
            # Compute predicted probabilities
            pred_probs = F.softmax(logits, dim=-1)

            # MSE between predicted and true vote distributions
            mse = F.mse_loss(pred_probs, target_probs, reduction='mean')
            self.log(f'{stage}/mse', mse, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)

            # Total Variation Distance: 0.5 * sum(|p - q|)
            tv_distance = 0.5 * torch.abs(pred_probs - target_probs).sum(dim=-1).mean()
            self.log(f'{stage}/tv_distance', tv_distance, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)

            # Per-class MSE
            per_class_mse = ((pred_probs - target_probs) ** 2).mean(dim=0)
            for i, class_mse in enumerate(per_class_mse):
                self.log(f'{stage}/mse_class_{i}', class_mse, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """Training step."""
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step."""
        return self._step(batch, batch_idx, 'val')

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """Test step."""
        return self._step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Return optimizer only if no scheduler
        if not self.scheduler_config:
            return optimizer

        scheduler_type = self.scheduler_config.get('type', 'ReduceLROnPlateau')

        if scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.scheduler_config.get('mode', 'min'),
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 5),
                min_lr=self.scheduler_config.get('min_lr', 1e-6),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': self.scheduler_config.get('monitor', 'val/loss'),
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }

        if scheduler_type == 'CosineAnnealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_config.get('T_max', 50),
                eta_min=self.scheduler_config.get('min_lr', 1e-6),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }

        if scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config.get('step_size', 10),
                gamma=self.scheduler_config.get('gamma', 0.1),
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }

        # Unknown scheduler, return optimizer only
        print(f"Warning: Unknown scheduler type '{scheduler_type}', using no scheduler")
        return optimizer

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        return self.model.get_model_info()


__all__ = ['HMSEEGOnlyLightningModule']
