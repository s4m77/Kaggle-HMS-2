"""Train the EEG-only MLP baseline with patient-level K-fold cross-validation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from src.data.raw_datamodule import RawEEGDataModule
from src.models.lightning_model import HMSLightningModule

FoldIterator = Iterator[Tuple[int, DataLoader, DataLoader, Dict[str, Sequence[str]]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG MLP baseline with cross-validation.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_mlp.yaml",
        help="Path to the MLP training configuration.",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Enable Lightning's fast_dev_run for debugging.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of folds for patient-level cross-validation.",
    )
    return parser.parse_args()


def load_config(path: str) -> DictConfig:
    cfg = OmegaConf.load(path)
    if cfg is None:
        raise ValueError(f"Unable to load configuration from {path}.")
    return cfg


def init_seed(cfg: DictConfig) -> None:
    seed = int(cfg.seed)
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_wandb_logger(cfg: DictConfig, fold: int) -> WandbLogger:
    wandb_cfg = cfg.logging.wandb
    tags = list(getattr(wandb_cfg, "tags", [])) + [f"fold-{fold}"]
    run_name = (
        f"{wandb_cfg.run_name}-fold-{fold}"
        if getattr(wandb_cfg, "run_name", None)
        else f"mlp-fold-{fold}-{random.randint(0, 999999):06d}"
    )

    logger = WandbLogger(
        project=wandb_cfg.project,
        name=run_name,
        entity=getattr(wandb_cfg, "entity", None),
        tags=tags,
        log_model=True,
    )
    logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    return logger


def build_callbacks(cfg: DictConfig, fold: int) -> list[Any]:
    callbacks: list[Any] = []

    if hasattr(cfg, "checkpointing"):
        ckpt_cfg = cfg.checkpointing
        fold_dir = Path(ckpt_cfg.dirpath) / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(fold_dir),
                monitor=ckpt_cfg.monitor,
                mode=ckpt_cfg.mode,
                save_top_k=int(ckpt_cfg.save_top_k),
                save_last=bool(ckpt_cfg.save_last),
            )
        )

    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    if hasattr(cfg, "early_stopping"):
        es_cfg = cfg.early_stopping
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.monitor,
                patience=int(es_cfg.patience),
                mode=es_cfg.mode,
                min_delta=float(es_cfg.min_delta),
            )
        )

    return callbacks


def build_trainer(
    cfg: DictConfig,
    logger: WandbLogger,
    callbacks: list[Any],
    fast_dev_run: bool,
) -> pl.Trainer:
    trainer_cfg = cfg.trainer
    trainer_kwargs: Dict[str, Any] = {
        "accelerator": trainer_cfg.accelerator,
        "devices": trainer_cfg.devices,
        "max_epochs": int(trainer_cfg.max_epochs),
        "precision": trainer_cfg.precision,
        "gradient_clip_val": float(trainer_cfg.gradient_clip_val),
        "log_every_n_steps": int(trainer_cfg.log_every_n_steps),
        "fast_dev_run": fast_dev_run,
        "logger": logger,
        "callbacks": callbacks,
    }

    if hasattr(trainer_cfg, "deterministic"):
        trainer_kwargs["deterministic"] = bool(trainer_cfg.deterministic)
    if hasattr(trainer_cfg, "benchmark"):
        trainer_kwargs["benchmark"] = bool(trainer_cfg.benchmark)

    return pl.Trainer(**trainer_kwargs)


def build_raw_eeg_fold_iterator(cfg: DictConfig, n_splits: int) -> FoldIterator:
    data_module = RawEEGDataModule(cfg)
    data_module.prepare_data()
    patient_ids = np.array(data_module.patient_ids())
    if patient_ids.size < n_splits:
        raise ValueError(
            f"Requested {n_splits} folds but only {patient_ids.size} patients were found in metadata."
        )

    groups = patient_ids
    gkf = GroupKFold(n_splits=n_splits)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(patient_ids, groups=groups)):
        train_ids = [str(pid) for pid in patient_ids[train_idx].tolist()]
        val_ids = [str(pid) for pid in patient_ids[val_idx].tolist()]

        train_dataset = data_module.dataset_for_patients(train_ids)
        val_dataset = data_module.dataset_for_patients(val_ids)

        train_loader = data_module.dataloader(train_dataset, shuffle=True)
        val_loader = data_module.dataloader(val_dataset, shuffle=False)

        fold_info = {
            "train_patient_ids": train_ids,
            "val_patient_ids": val_ids,
        }
        yield fold_idx, train_loader, val_loader, fold_info


def train_single_fold(
    fold_idx: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: DictConfig,
    fast_dev_run: bool,
) -> Optional[float]:
    model = HMSLightningModule(cfg)

    should_compile = bool(getattr(cfg.trainer, "compile", False))
    if should_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile for this fold.")
        except Exception as exc:
            print(f"torch.compile failed (continuing without compilation): {exc}")

    logger = init_wandb_logger(cfg, fold_idx)
    callbacks = build_callbacks(cfg, fold_idx)
    trainer = build_trainer(cfg, logger, callbacks, fast_dev_run)

    best_val_loss: Optional[float] = None
    try:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        checkpoint_cb = getattr(trainer, "checkpoint_callback", None)
        best_score = getattr(checkpoint_cb, "best_model_score", None) if checkpoint_cb else None
        if best_score is not None:
            best_val_loss = float(best_score.item())
            print(f"Fold {fold_idx + 1} best validation loss: {best_val_loss:.4f}")
    finally:
        experiment = getattr(logger, "experiment", None)
        if experiment is not None:
            experiment.finish()
            print(f"Closed WandB run for fold {fold_idx + 1}.")

    return best_val_loss


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    init_seed(cfg)
    torch.set_float32_matmul_precision("medium")

    fold_iterator = build_raw_eeg_fold_iterator(cfg, args.n_splits)
    all_val_losses = []

    for fold_idx, train_loader, val_loader, info in fold_iterator:
        print("-" * 80)
        print(f"Starting fold {fold_idx + 1}/{args.n_splits} for EEG MLP")
        print("-" * 80)

        train_patients = info.get("train_patient_ids", [])
        val_patients = info.get("val_patient_ids", [])
        print(f"Train patients: {len(train_patients)} | Val patients: {len(val_patients)}")

        try:
            train_samples = len(train_loader.dataset)  # type: ignore[attr-defined]
            val_samples = len(val_loader.dataset)  # type: ignore[attr-defined]
        except TypeError:
            train_samples = val_samples = None

        if train_samples is not None and val_samples is not None:
            print(f"Train samples: {train_samples} | Val samples: {val_samples}")

        best_val_loss = train_single_fold(
            fold_idx=fold_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            fast_dev_run=args.fast_dev_run,
        )
        if best_val_loss is not None:
            all_val_losses.append(best_val_loss)

    print("-" * 80)
    if all_val_losses:
        mean_val = float(np.mean(all_val_losses))
        std_val = float(np.std(all_val_losses))
        print(f"Average validation loss across {len(all_val_losses)} folds: {mean_val:.4f} +/- {std_val:.4f}")
    else:
        print("Could not retrieve validation losses.")


if __name__ == "__main__":
    main()
