"""PyTorch Lightning module for HMS Multi-Modal GNN."""

from __future__ import annotations

from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall

from src.models import HMSMultiModalGNN
from src.models.regularization import compute_graph_regularization


class HMSLightningModule(LightningModule):
    """Lightning wrapper for HMS Multi-Modal GNN.
    
    Handles:
    - Training, validation, and test steps
    - Loss computation with optional class weights
    - Graph Laplacian regularization
    - Metrics: accuracy, F1, precision, recall (macro & per-class)
    - WandB logging
    - Learning rate scheduling
    - Optimizer configuration
    
    Parameters
    ----------
    model_config : dict
        Model configuration (eeg_encoder, spec_encoder, fusion, classifier)
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
        graph_laplacian_lambda: float = 0.001,
        edge_weight_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        
        # Save hyperparameters (will be logged to WandB)
        self.save_hyperparameters(ignore=['class_weights'])
        
        # Initialize model
        self.model = HMSMultiModalGNN(
            eeg_config=model_config.get('eeg_encoder'),
            spec_config=model_config.get('spec_encoder'),
            fusion_config=model_config.get('fusion'),
            classifier_config=model_config.get('classifier'),
            num_classes=num_classes,
        )
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        self.graph_laplacian_lambda = graph_laplacian_lambda
        self.edge_weight_penalty = edge_weight_penalty
        
        # Loss function
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.class_weights = None
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics for each stage
        metrics = MetricCollection({
            'acc': Accuracy(task='multiclass', num_classes=num_classes, average='micro'),
            'acc_macro': Accuracy(task='multiclass', num_classes=num_classes, average='macro'),
            'f1_macro': MulticlassF1Score(num_classes=num_classes, average='macro'),
            'precision_macro': MulticlassPrecision(num_classes=num_classes, average='macro'),
            'recall_macro': MulticlassRecall(num_classes=num_classes, average='macro'),
        })
        
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        
        # Per-class accuracy (for detailed analysis)
        self.train_acc_per_class = Accuracy(
            task='multiclass', num_classes=num_classes, average='none'
        )
        self.val_acc_per_class = Accuracy(
            task='multiclass', num_classes=num_classes, average='none'
        )
        self.test_acc_per_class = Accuracy(
            task='multiclass', num_classes=num_classes, average='none'
        )
    
    def forward(self, eeg_graphs, spec_graphs):
        """Forward pass through the model."""
        return self.model(eeg_graphs, spec_graphs)
    
    def _step(self, batch: Dict, batch_idx: int, stage: str):
        """Shared step for train/val/test.
        
        Parameters
        ----------
        batch : dict
            Batch from dataloader
        batch_idx : int
            Batch index
        stage : str
            One of 'train', 'val', 'test'
        """
        eeg_graphs = batch['eeg_graphs']
        spec_graphs = batch['spec_graphs']
        targets = batch['targets']
        
        # Forward pass (with intermediate outputs for regularization during training)
        if stage == 'train' and (self.graph_laplacian_lambda > 0 or self.edge_weight_penalty > 0):
            logits, intermediate = self.model(eeg_graphs, spec_graphs, return_intermediate=True)
        else:
            logits = self.model(eeg_graphs, spec_graphs)
            intermediate = None
        
        # Compute classification loss
        ce_loss = self.criterion(logits, targets)
        
        # Safety check for NaN in CE loss
        if torch.isnan(ce_loss):
            print(f"WARNING: NaN in CE loss at batch {batch_idx}")
            print(f"  Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
            print(f"  Targets: {targets}")
            print(f"  Has NaN in logits: {torch.isnan(logits).any()}")
            print(f"  Has Inf in logits: {torch.isinf(logits).any()}")
            # Skip this batch to prevent NaN propagation
            return None
        
        # Add graph regularization (only during training)
        if stage == 'train' and intermediate is not None:
            # EEG graph regularization
            eeg_reg = compute_graph_regularization(
                intermediate['eeg_graphs'],
                lambda_laplacian=self.graph_laplacian_lambda,
                lambda_edge=self.edge_weight_penalty,
            )
            
            # Spectrogram graph regularization
            spec_reg = compute_graph_regularization(
                intermediate['spec_graphs'],
                lambda_laplacian=self.graph_laplacian_lambda,
                lambda_edge=self.edge_weight_penalty,
            )
            
            # Total regularization
            reg_loss = eeg_reg + spec_reg
            
            # Safety check for NaN in regularization
            if torch.isnan(reg_loss):
                print(f"WARNING: NaN in regularization loss at batch {batch_idx}")
                print(f"  EEG reg: {eeg_reg}, Spec reg: {spec_reg}")
                # Use CE loss only if regularization is NaN
                reg_loss = torch.tensor(0.0, device=ce_loss.device)
            
            total_loss = ce_loss + reg_loss
            
            # Log individual components (step-level for progress bar, epoch-level for WandB)
            self.log(f'{stage}/ce_loss', ce_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log(f'{stage}/reg_loss', reg_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log(f'{stage}/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
            loss = total_loss
        else:
            # No regularization for val/test
            self.log(f'{stage}/loss', ce_loss, on_step=True, on_epoch=True, prog_bar=True)
            loss = ce_loss
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        if stage == 'train':
            metrics = self.train_metrics(preds, targets)
            self.train_acc_per_class(preds, targets)
        elif stage == 'val':
            metrics = self.val_metrics(preds, targets)
            self.val_acc_per_class(preds, targets)
        else:  # test
            metrics = self.test_metrics(preds, targets)
            self.test_acc_per_class(preds, targets)
        
        # Log metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def training_step(self, batch: Dict, batch_idx: int):
        """Training step."""
        return self._step(batch, batch_idx, 'train')
    
    def validation_step(self, batch: Dict, batch_idx: int):
        """Validation step."""
        return self._step(batch, batch_idx, 'val')
    
    def test_step(self, batch: Dict, batch_idx: int):
        """Test step."""
        return self._step(batch, batch_idx, 'test')
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Compute and log per-class accuracy
        per_class_acc = self.train_acc_per_class.compute()
        for i, acc in enumerate(per_class_acc):
            self.log(f'train/acc_class_{i}', acc, prog_bar=False)
        self.train_acc_per_class.reset()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Compute and log per-class accuracy
        per_class_acc = self.val_acc_per_class.compute()
        for i, acc in enumerate(per_class_acc):
            self.log(f'val/acc_class_{i}', acc, prog_bar=False)
        self.val_acc_per_class.reset()
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        # Compute and log per-class accuracy
        per_class_acc = self.test_acc_per_class.compute()
        for i, acc in enumerate(per_class_acc):
            self.log(f'test/acc_class_{i}', acc, prog_bar=False)
        self.test_acc_per_class.reset()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Return optimizer only if no scheduler
        if not self.scheduler_config:
            return optimizer
        
        # Configure scheduler
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
        
        elif scheduler_type == 'CosineAnnealing':
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
        
        elif scheduler_type == 'StepLR':
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
        
        else:
            # Unknown scheduler, return optimizer only
            print(f"Warning: Unknown scheduler type '{scheduler_type}', using no scheduler")
            return optimizer
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        return self.model.get_model_info()


__all__ = ["HMSLightningModule"]
