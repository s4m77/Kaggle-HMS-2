"""Main training script for HMS Multi-Modal GNN."""

import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import HMSDataModule
from src.lightning_trainer import HMSLightningModule


def train(
    model_config_path: str = "configs/model.yaml",
    train_config_path: str = "configs/train.yaml",
    wandb_project: str = "hms-brain-activity",
    wandb_name: str | None = None,
    resume_from_checkpoint: str | None = None,
):
    """Train HMS Multi-Modal GNN model.
    
    Parameters
    ----------
    model_config_path : str
        Path to model configuration file
    train_config_path : str
        Path to training configuration file
    wandb_project : str
        WandB project name
    wandb_name : str, optional
        WandB run name (auto-generated if None)
    resume_from_checkpoint : str, optional
        Path to checkpoint to resume from
    """
    # Load configurations
    model_config = OmegaConf.load(model_config_path)
    train_config = OmegaConf.load(train_config_path)
    
    print("\n" + "="*60)
    print("HMS Multi-Modal GNN Training")
    print("="*60)
    print(f"Model Config: {model_config_path}")
    print(f"Train Config: {train_config_path}")
    print(f"WandB Project: {wandb_project}")
    print(f"WandB Run: {wandb_name or 'auto-generated'}")
    print(f"Fold: {train_config.data.current_fold}/{train_config.data.n_folds-1}")
    print("="*60 + "\n")
    
    # Initialize DataModule with K-Fold CV
    print("Initializing DataModule...")
    datamodule = HMSDataModule(
        data_dir=train_config.data.data_dir,
        train_csv=train_config.data.train_csv,
        batch_size=train_config.batch_size,
        n_folds=train_config.data.n_folds,
        current_fold=train_config.data.current_fold,
        stratify_by_evaluators=train_config.data.stratify_by_evaluators,
        evaluator_bins=train_config.data.evaluator_bins,
        min_evaluators=train_config.data.get('min_evaluators', 0),
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        shuffle_seed=train_config.data.shuffle_seed,
    )
    
    # Setup to get class weights
    datamodule.setup(stage="fit")
    class_weights = datamodule.get_class_weights() if train_config.use_class_weights else None
    
    # Initialize Lightning Module
    print("Initializing Model...")
    model = HMSLightningModule(
        model_config=model_config.model,
        num_classes=model_config.model.num_classes,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.regularization.weight_decay,
        class_weights=class_weights,
        scheduler_config=train_config.scheduler,
        graph_laplacian_lambda=train_config.regularization.graph_laplacian_lambda,
        edge_weight_penalty=train_config.regularization.edge_weight_penalty,
    )
    
    # Print model info
    model_info = model.get_model_info()
    print(f"\nModel Architecture:")
    print(f"  EEG output dim:    {model_info['eeg_output_dim']}")
    print(f"  Spec output dim:   {model_info['spec_output_dim']}")
    print(f"  Fusion output dim: {model_info['fusion_output_dim']}")
    print(f"  Num classes:       {model_info['num_classes']}")
    print(f"  Total parameters:  {model_info['total_params']:,}")
    print(f"  Trainable params:  {model_info['trainable_params']:,}\n")
    
    # WandB Logger
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=wandb_name,
        save_dir="logs",
        log_model=True,  # Log model checkpoints to WandB
    )
    
    # Log configuration to WandB
    wandb_logger.experiment.config.update({
        "model": OmegaConf.to_container(model_config, resolve=True),
        "training": OmegaConf.to_container(train_config, resolve=True),
    })
    
    # Callbacks
    callbacks = []
    
    # Model Checkpoint - save best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="hms-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
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
    # Handle both nested (hardware.mixed_precision) and flat (mixed_precision) config structures
    if hasattr(train_config, 'hardware') and train_config.hardware is not None:
        mixed_precision = train_config.hardware.mixed_precision
    else:
        mixed_precision = getattr(train_config, 'mixed_precision', False)
    
    trainer = Trainer(
        max_epochs=train_config.num_epochs,
        accelerator="auto",  # Auto-detect GPU/CPU/MPS
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=16 if mixed_precision else 32,
        gradient_clip_val=1.0,  # Clip gradients to prevent exploding gradients
        log_every_n_steps=10,
        deterministic=False,  # Set to True for reproducibility (slower)
    )
    
    # Train
    print("Starting training...\n")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=resume_from_checkpoint,
    )
    
    # Test on best model
    print("\nTesting best model...\n")
    trainer.test(
        model,
        datamodule=datamodule,
        ckpt_path="best",
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"WandB run: {wandb_logger.experiment.url}")
    print("="*60 + "\n")
    
    return trainer, model, datamodule


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train HMS Multi-Modal GNN")
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/train.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="hms-brain-activity",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="WandB run name",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Override current_fold from config (0-4 for 5-fold CV)",
    )
    
    args = parser.parse_args()
    
    # Override fold if specified
    if args.fold is not None:
        train_cfg = OmegaConf.load(args.train_config)
        train_cfg.data.current_fold = args.fold
        OmegaConf.save(train_cfg, args.train_config)
        print(f"Updated current_fold to {args.fold} in {args.train_config}")
    
    train(
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        resume_from_checkpoint=args.resume,
    )
