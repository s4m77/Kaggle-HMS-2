"""
Training script with on-the-fly graph generation.

This script uses HMSOnlineDataModule to generate graphs dynamically during training
instead of loading pre-computed graphs from disk. Optimized for modern GPUs (H100/H200)
with BF16/TF32, optional torch.compile, and tuned DataLoader settings.

Usage:
    python src/train_online.py --fold 0
    python src/train_online.py --fold 0 --config configs/train.yaml
"""

import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import HMSOnlineDataModule, HMSDataModule
from src.data.build_processed import build_missing_processed
from src.lightning_trainer import HMSLightningModule
from src.models import HMSMultiModalGNN


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train HMS model with online graph generation")
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to training config')
    parser.add_argument('--graph_config', type=str, default='configs/graphs.yaml', help='Path to graph config')
    parser.add_argument('--model_config', type=str, default='configs/model.yaml', help='Path to model config')
    parser.add_argument('--fold', type=int, default=None, help='Override fold number')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Prefer TF32/better matmul perf on Tensor Core GPUs
    try:
        import torch.backends.cuda as cuda_backends
        torch.set_float32_matmul_precision('high')
        cuda_backends.matmul.allow_tf32 = True
        cuda_backends.cudnn.allow_tf32 = True
        try:
            import torch._dynamo as dynamo
            dynamo.config.capture_scalar_outputs = True
        except Exception:
            pass
    except Exception:
        pass

    # Load configurations
    train_config = OmegaConf.load(args.config)
    graph_config = OmegaConf.load(args.graph_config)
    # Resolve model config path robustly (allow relative to config dir or project root)
    model_cfg_input = train_config.model_config
    cfg_base = Path(args.config).parent
    model_cfg_raw = Path(model_cfg_input)
    if model_cfg_raw.is_absolute():
        model_cfg_path = model_cfg_raw
    else:
        cand1 = (cfg_base / model_cfg_raw).resolve()
        cand2 = (project_root / model_cfg_raw).resolve()
        if cand1.exists():
            model_cfg_path = cand1
        elif cand2.exists():
            model_cfg_path = cand2
        else:
            model_cfg_path = model_cfg_raw
    model_config = OmegaConf.load(str(model_cfg_path))
    train_config.model_config = str(model_cfg_path)
    
    # Override fold if specified
    if args.fold is not None:
        train_config.data.current_fold = args.fold
    
    # Parse resume checkpoint
    resume_from_checkpoint = args.resume
    
    # WandB project name
    wandb_project = train_config.wandb_project
    wandb_name = train_config.wandb_name or f"online-fold{train_config.data.current_fold}"
    
    print(f"\n{'='*70}")
    print(f"Training Configuration (ONLINE Graph Generation)")
    print(f"{'='*70}")
    print(f"Model config:  {args.model_config}")
    print(f"Graph config:  {args.graph_config}")
    print(f"Train config:  {args.config}")
    print(f"Fold:          {train_config.data.current_fold}/{train_config.data.n_folds - 1}")
    print(f"Batch size:    {train_config.batch_size}")
    print(f"Learning rate: {train_config.learning_rate}")
    print(f"Loss:          {train_config.loss}")
    print(f"Device:        {train_config.device}")
    print(f"Precision:     {'FP16' if train_config.mixed_precision else 'FP32'}")
    print(f"WandB project: {wandb_project}")
    print(f"WandB run:     {wandb_name}")
    if resume_from_checkpoint:
        print(f"Resume from:   {resume_from_checkpoint}")
    print(f"{'='*70}\n")
    
    # DataModule with online graph generation
    print("\n[1/4] Setting up DataModule (online graph generation)...")
    datamodule = HMSOnlineDataModule(
        raw_data_dir=Path("data/raw"),
        train_csv=Path(train_config.data.train_csv),
        graph_config=graph_config,
        batch_size=train_config.batch_size,
        n_folds=train_config.data.n_folds,
        current_fold=train_config.data.current_fold,
        stratify_by_class=train_config.data.stratify_by_class,
        stratify_by_evaluators=train_config.data.stratify_by_evaluators,
        evaluator_bins=train_config.data.evaluator_bins,
        min_evaluators=train_config.data.min_evaluators,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        # Per latest direction: no caching across or within run
        cache_raw_data=False,
        prefetch_factor=getattr(train_config.data, 'prefetch_factor', 2),
        shuffle_seed=train_config.data.shuffle_seed,
    )
    
    # Setup datasets (creates train/val split)
    datamodule.setup()

    # Optional: Build processed graphs once, then switch to processed DataModule
    auto_build = bool(getattr(train_config.data, 'auto_build_processed', True))
    processed_dir = Path(getattr(train_config.data, 'data_dir', 'data/processed'))
    if auto_build:
        print("\n[1b/4] Building missing processed patient files (one-time)...")
        # Build for patients present in this fold (train + val)
        train_meta = datamodule.train_dataset.metadata_df
        val_meta = datamodule.val_dataset.metadata_df
        build_scope = str(getattr(train_config.data, 'build_scope', 'train+val'))
        if build_scope == 'train-only':
            fold_meta = train_meta
            print("Building scope: train-only")
        elif build_scope == 'val-only':
            fold_meta = val_meta
            print("Building scope: val-only")
        else:
            fold_meta = pd.concat([train_meta, val_meta], ignore_index=True)
            print("Building scope: train+val")
        build_workers = int(getattr(train_config.data, 'build_workers', 0))
        build_missing_processed(
            metadata=fold_meta,
            raw_data_dir=Path("data/raw"),
            processed_dir=processed_dir,
            graph_config=OmegaConf.to_container(graph_config, resolve=True),
            patient_ids=fold_meta['patient_id'].unique(),
            overwrite=False,
            num_workers=build_workers,
        )
        # Switch to processed DataModule with in-RAM preload for fast training
        # When preloading, prefer num_workers=0 to avoid shared-memory pressure
        preload_flag = True
        dm_workers = 0
        prefetch = None
        pin_mem = bool(getattr(train_config.data, 'pin_memory', False))
        print("[1c/4] Switching to processed DataModule with preload (RAM)...")
        datamodule = HMSDataModule(
            data_dir=str(processed_dir),
            train_csv=train_config.data.train_csv,
            batch_size=train_config.batch_size,
            n_folds=train_config.data.n_folds,
            current_fold=train_config.data.current_fold,
            stratify_by_class=train_config.data.stratify_by_class,
            stratify_by_evaluators=train_config.data.stratify_by_evaluators,
            evaluator_bins=train_config.data.evaluator_bins,
            min_evaluators=train_config.data.min_evaluators,
            num_workers=dm_workers,
            pin_memory=pin_mem,
            prefetch_factor=prefetch,
            shuffle_seed=train_config.data.shuffle_seed,
            preload_patients=preload_flag,
        )
        datamodule.setup(stage="fit")
    
    # Model
    print("\n[2/4] Initializing model...")
    model = HMSLightningModule(
        model_config=model_config.model if 'model' in model_config else model_config,
        num_classes=6,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        scheduler_config=dict(train_config.scheduler) if train_config.scheduler else None,
        class_weights=None,  # No class weights for KL divergence
        graph_laplacian_lambda=train_config.regularization.graph_laplacian_lambda,
        edge_weight_penalty=train_config.regularization.edge_weight_penalty,
        loss_type=train_config.loss,
    )
    
    # Print model info
    try:
        model_info = model.get_model_info()
    except Exception:
        # Fallback for older interface
        model_info = model.model.get_model_info()
    print(f"\nModel Architecture:")
    print(f"  EEG output dim:    {model_info['eeg_output_dim']}")
    print(f"  Spec output dim:   {model_info['spec_output_dim']}")
    print(f"  Fusion output dim: {model_info['fusion_output_dim']}")
    print(f"  Num classes:       {model_info['num_classes']}")
    print(f"  Total parameters:  {model_info['total_params']:,}")
    print(f"  Trainable params:  {model_info['trainable_params']:,}\n")
    
    # WandB Logger
    print("\n[3/4] Setting up logging and callbacks...")
    logger_kwargs = {
        "project": wandb_project,
        "name": wandb_name,
        "save_dir": "logs",
        "log_model": True,
        "tags": list(getattr(train_config, "wandb_tags", []) or []),
        "group": getattr(train_config, "wandb_group", None),
        "job_type": getattr(train_config, "wandb_job_type", None),
        "anonymous": "allow",
    }
    if hasattr(train_config, "wandb_entity") and train_config.wandb_entity:
        logger_kwargs["entity"] = train_config.wandb_entity
    wandb_logger = WandbLogger(**logger_kwargs)
    
    # Configure WandB to log step-level metrics
    wandb_logger.experiment.define_metric("train/loss_step", step_metric="trainer/global_step")
    wandb_logger.experiment.define_metric("train/loss_epoch", step_metric="epoch")
    
    # Log configuration to WandB
    wandb_logger.experiment.config.update({
        "model": OmegaConf.to_container(model_config, resolve=True),
        "training": OmegaConf.to_container(train_config, resolve=True),
        "graph": OmegaConf.to_container(graph_config, resolve=True),
        "online_generation": True,  # Flag to indicate online mode
    })
    
    # Callbacks
    callbacks = []
    
    # Model Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"online-fold{train_config.data.current_fold}-" + "epoch={epoch:02d}-val_loss={val/loss_epoch:.4f}",
        monitor="val/loss_epoch",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early Stopping
    if train_config.early_stopping.get('patience'):
        early_stop_callback = EarlyStopping(
            monitor=train_config.early_stopping.monitor,
            patience=train_config.early_stopping.patience,
            mode=train_config.early_stopping.mode,
            verbose=True,
        )
        callbacks.append(early_stop_callback)
    
    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Trainer
    print("\n[4/4] Creating trainer...")

    # Determine accelerator and precision
    if hasattr(train_config, 'hardware') and train_config.hardware is not None:
        mixed_precision = bool(train_config.hardware.mixed_precision)
        device = getattr(train_config.hardware, 'device', 'auto')
    else:
        mixed_precision = bool(getattr(train_config, 'mixed_precision', False))
        device = getattr(train_config, 'device', 'auto')

    if device == 'cuda' or (device == 'auto' and torch.cuda.is_available()):
        accelerator = 'gpu'
    elif device == 'mps':
        accelerator = 'mps'
    elif device == 'cpu':
        accelerator = 'cpu'
    else:
        accelerator = 'auto'

    if accelerator == 'gpu' and torch.cuda.is_available():
        try:
            cc = torch.cuda.get_device_capability(0)
        except Exception:
            cc = (0, 0)
        bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        if mixed_precision and (bf16_supported or cc[0] >= 8):
            precision = "bf16-mixed"
        elif mixed_precision:
            precision = "16-mixed"
        else:
            precision = 32
        # Enable cuDNN benchmark for static shapes
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
        except Exception:
            pass
    else:
        precision = 32

    # Optional torch.compile for PyTorch 2.x
    if getattr(train_config, "compile_model", False) and hasattr(torch, "compile") and accelerator == 'gpu':
        try:
            compile_mode = str(getattr(train_config, "compile_mode", "reduce-overhead"))
            model = torch.compile(model, mode=compile_mode)
            print(f"Enabled torch.compile with mode={compile_mode}")
        except Exception as e:
            print(f"torch.compile failed, continuing without compile: {e}")

    trainer = Trainer(
        max_epochs=train_config.num_epochs,
        accelerator=accelerator,
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=precision,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.25,
        deterministic=False,
        num_sanity_val_steps=0,
    )
    
    # Train
    print("\n" + "="*70)
    print("Starting training with ONLINE graph generation...")
    print("="*70 + "\n")
    
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=resume_from_checkpoint,
    )
    
    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val/loss: {checkpoint_callback.best_model_score:.4f}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
