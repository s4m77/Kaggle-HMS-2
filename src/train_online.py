"""
Training script with on-the-fly graph generation.

This script uses HMSOnlineDataModule to generate graphs dynamically during training
instead of loading pre-computed graphs from disk.

Usage:
    python src/train_online.py --fold 0
    python src/train_online.py --fold 0 --config configs/train.yaml
"""

import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import HMSOnlineDataModule
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
    
    # Load configurations
    train_config = OmegaConf.load(args.config)
    graph_config = OmegaConf.load(args.graph_config)
    model_config = OmegaConf.load(train_config.model_config)
    
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
        cache_raw_data=True,  # Cache raw data in memory for speed
        shuffle_seed=train_config.data.shuffle_seed,
    )
    
    # Setup datasets (creates train/val split)
    datamodule.setup()
    
    # Model
    print("\n[2/4] Initializing model...")
    model = HMSLightningModule(
        model_config=model_config,
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
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=wandb_name,
        save_dir="logs",
        log_model=True,
    )
    
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
    
    # Configure precision based on device
    mixed_precision = getattr(train_config, 'mixed_precision', False)
    device = getattr(train_config, 'device', 'auto')
    
    if mixed_precision and device == 'cuda':
        precision = 16
    else:
        # Use 32-bit precision for MPS and CPU
        precision = 32
    
    trainer = Trainer(
        max_epochs=train_config.num_epochs,
        accelerator='auto',
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=precision,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.25,
        deterministic=False,
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
