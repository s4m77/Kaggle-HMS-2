"""PyTorch Lightning module for the raw EEG MLP baseline."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.classification import MulticlassF1Score

from src.models.eeg_mlp import EEGMLPBaseline


class EEGMLPLightningModule(LightningModule):
    """Lightning wrapper for the EEG-only MLP baseline that trains on vote distributions."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        if config is None:
            raise ValueError("A DictConfig must be provided to initialise EEGMLPLightningModule.")

        self.save_hyperparameters(ignore=[])
        self.cfg = config

        vote_keys = getattr(self.cfg.data, "vote_keys", None)
        if not vote_keys:
            raise ValueError("cfg.data.vote_keys must list the vote columns for training.")
        self.vote_keys = [str(v) for v in vote_keys]
        self.num_classes = len(self.vote_keys)

        self.model = EEGMLPBaseline(self.cfg)

        loss_cfg = getattr(self.cfg, "loss", None)
        self.loss_type = str(getattr(loss_cfg, "type", "kl")).lower()

        trainer_cfg = getattr(self.cfg, "trainer", None)
        self.max_epochs = int(getattr(trainer_cfg, "max_epochs", 1)) if trainer_cfg else 1

        metrics = MetricCollection(
            {
                "acc_macro": Accuracy(task="multiclass", num_classes=self.num_classes, average="macro"),
                "f1_macro": MulticlassF1Score(num_classes=self.num_classes, average="macro"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, eeg_signal: torch.Tensor) -> torch.Tensor:
        """Run the baseline MLP and return logits."""
        outputs = self.model(eeg_signal, return_aux=False)
        return outputs["logits"]

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        logits = self(batch["eeg_signal"])
        probs = torch.softmax(logits, dim=-1)
        return {"probs": probs}

    def configure_optimizers(self) -> Any:
        opt_cfg = getattr(self.cfg, "optimizer", None)
        if opt_cfg is None:
            raise ValueError("cfg.optimizer section is required for EEG MLP training.")

        name = str(getattr(opt_cfg, "name", "adamw")).lower()
        lr = float(getattr(opt_cfg, "lr", 1e-3))
        weight_decay = float(getattr(opt_cfg, "weight_decay", 0.0))

        if name == "adamw":
            betas = tuple(getattr(opt_cfg, "betas", (0.9, 0.999)))  # type: ignore[assignment]
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        elif name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer '{name}' for EEG MLP baseline.")

        sched_cfg = getattr(self.cfg, "scheduler", None)
        if sched_cfg is None or getattr(sched_cfg, "name", None) is None:
            return optimizer

        sched_name = str(getattr(sched_cfg, "name")).lower()
        if sched_name == "cosine_with_warmup":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self._cosine_with_warmup_lambda(opt_cfg, sched_cfg))
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        raise ValueError(f"Unsupported scheduler '{sched_name}' for EEG MLP baseline.")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        eeg_signal = batch["eeg_signal"]
        targets = batch.get("target")
        confidences = batch.get("confidence")
        if targets is None:
            raise ValueError("Batch is missing 'target' tensors required for vote-based training.")

        logits = self(eeg_signal)
        loss = self._compute_loss(logits, targets, confidences)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage != "train"),
            batch_size=eeg_signal.size(0),
        )

        metrics = getattr(self, f"{stage}_metrics", None)
        if metrics is not None:
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                target_indices = torch.argmax(targets, dim=-1)
            metric_values = metrics(preds, target_indices)
            self.log_dict(metric_values, on_step=False, on_epoch=True, prog_bar=False, batch_size=eeg_signal.size(0))

        return loss

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, confidences: Optional[torch.Tensor]) -> torch.Tensor:
        targets = self._normalise_targets(targets)
        if self.loss_type == "kl":
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = F.kl_div(log_probs, targets, reduction="none")
            loss = loss.sum(dim=-1)
        elif self.loss_type == "bce":
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = loss.mean(dim=-1)
        elif self.loss_type == "mse":
            probs = torch.softmax(logits, dim=-1)
            loss = F.mse_loss(probs, targets, reduction="none")
            loss = loss.mean(dim=-1)
        else:
            raise ValueError(f"Unsupported loss type '{self.loss_type}'.")

        if confidences is not None:
            confidence = confidences.view(-1)
            loss = loss * confidence

        return loss.mean()

    def _normalise_targets(self, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        sums = targets.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(targets, 1.0 / self.num_classes)
        return torch.where(sums > 0, targets / sums.clamp_min(1e-6), uniform)

    def _cosine_with_warmup_lambda(self, opt_cfg: DictConfig, sched_cfg: DictConfig):
        base_lr = float(getattr(opt_cfg, "lr", 1e-3))
        min_lr = float(getattr(sched_cfg, "min_lr", base_lr * 0.01))
        warmup_epochs = int(getattr(sched_cfg, "warmup_epochs", 0))
        min_factor = min_lr / base_lr if base_lr > 0 else 0.0
        total_epochs = max(1, self.max_epochs)

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return max(1e-8, (epoch + 1) / max(1, warmup_epochs))

            progress = 0.0
            if total_epochs - warmup_epochs > 0:
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                progress = min(max(progress, 0.0), 1.0)

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_factor + (1.0 - min_factor) * cosine)

        return lr_lambda


__all__ = ["EEGMLPLightningModule"]
