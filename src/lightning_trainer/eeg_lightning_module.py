"""PyTorch Lightning module for EEG-only HMS baseline."""

from __future__ import annotations

from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall

from src.models import HMSEEGOnlyGNN
from src.models.regularization import compute_graph_regularization


class HMSEEGOnlyLightningModule(LightningModule):
    """Lightning wrapper for EEG-only HMS GNN."""

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
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=['class_weights'])

        self.model = HMSEEGOnlyGNN(
            eeg_config=model_config.get('eeg_encoder'),
            classifier_config=model_config.get('classifier'),
            num_classes=num_classes,
        )

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_config = scheduler_config or {}
        self.graph_laplacian_lambda = graph_laplacian_lambda
        self.edge_weight_penalty = edge_weight_penalty

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.class_weights = None
            self.criterion = nn.CrossEntropyLoss()

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

        self.train_acc_per_class = Accuracy(
            task='multiclass', num_classes=num_classes, average='none'
        )
        self.val_acc_per_class = Accuracy(
            task='multiclass', num_classes=num_classes, average='none'
        )
        self.test_acc_per_class = Accuracy(
            task='multiclass', num_classes=num_classes, average='none'
        )

    def forward(self, eeg_graphs):
        return self.model(eeg_graphs)

    def _step(self, batch: Dict, batch_idx: int, stage: str):
        eeg_graphs = batch['eeg_graphs']
        targets = batch['targets']

        if stage == 'train' and (self.graph_laplacian_lambda > 0 or self.edge_weight_penalty > 0):
            logits, intermediate = self.model(eeg_graphs, return_intermediate=True)
        else:
            logits = self.model(eeg_graphs)
            intermediate = None

        ce_loss = self.criterion(logits, targets)

        if torch.isnan(ce_loss):
            print(f"WARNING: NaN in CE loss at batch {batch_idx}")
            print(f"  Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
            print(f"  Targets: {targets}")
            print(f"  Has NaN in logits: {torch.isnan(logits).any()}")
            print(f"  Has Inf in logits: {torch.isinf(logits).any()}")
            return None

        if stage == 'train' and intermediate is not None:
            eeg_reg = compute_graph_regularization(
                intermediate['eeg_graphs'],
                lambda_laplacian=self.graph_laplacian_lambda,
                lambda_edge=self.edge_weight_penalty,
            )

            if torch.isnan(eeg_reg):
                print(f"WARNING: NaN in regularization loss at batch {batch_idx}")
                eeg_reg = torch.tensor(0.0, device=ce_loss.device)

            total_loss = ce_loss + eeg_reg
            self.log(f'{stage}/ce_loss', ce_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log(f'{stage}/reg_loss', eeg_reg, on_step=True, on_epoch=True, prog_bar=False)
            self.log(f'{stage}/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
            loss = total_loss
        else:
            self.log(f'{stage}/loss', ce_loss, on_step=True, on_epoch=True, prog_bar=True)
            loss = ce_loss

        preds = torch.argmax(logits, dim=1)

        if stage == 'train':
            metrics = self.train_metrics(preds, targets)
            self.train_acc_per_class(preds, targets)
        elif stage == 'val':
            metrics = self.val_metrics(preds, targets)
            self.val_acc_per_class(preds, targets)
        else:
            metrics = self.test_metrics(preds, targets)
            self.test_acc_per_class(preds, targets)

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: Dict, batch_idx: int):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch: Dict, batch_idx: int):
        return self._step(batch, batch_idx, 'val')

    def test_step(self, batch: Dict, batch_idx: int):
        return self._step(batch, batch_idx, 'test')

    def on_train_epoch_end(self):
        per_class_acc = self.train_acc_per_class.compute()
        for i, acc in enumerate(per_class_acc):
            self.log(f'train/acc_class_{i}', acc, prog_bar=False)
        self.train_acc_per_class.reset()

    def on_validation_epoch_end(self):
        per_class_acc = self.val_acc_per_class.compute()
        for i, acc in enumerate(per_class_acc):
            self.log(f'val/acc_class_{i}', acc, prog_bar=False)
        self.val_acc_per_class.reset()

    def on_test_epoch_end(self):
        per_class_acc = self.test_acc_per_class.compute()
        for i, acc in enumerate(per_class_acc):
            self.log(f'test/acc_class_{i}', acc, prog_bar=False)
        self.test_acc_per_class.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

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

        print(f"Warning: Unknown scheduler type '{scheduler_type}', using no scheduler")
        return optimizer

    def get_model_info(self) -> Dict[str, Any]:
        return self.model.get_model_info()


__all__ = ["HMSEEGOnlyLightningModule"]
