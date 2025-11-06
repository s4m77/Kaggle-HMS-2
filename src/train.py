"""Main training script for HMS GNN models."""

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
from src.lightning_trainer import HMSLightningModule, HMSEEGOnlyLightningModule
import subprocess
import time

MODEL_REGISTRY = {
    "multi_modal": {
        "module": HMSLightningModule,
        "title": "HMS Multi-Modal GNN Training",
        "checkpoint_prefix": "hms",
    },
    "eeg_only": {
        "module": HMSEEGOnlyLightningModule,
        "title": "HMS EEG-Only GNN Training",
        "checkpoint_prefix": "hms-eeg",
    },
}

DEFAULT_MODEL_TYPE = "multi_modal"


def train(
    train_config_path: str = "configs/train.yaml",
    model_config_path: str | None = None,
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
    # Prefer TF32/better matmul perf on Tensor Core GPUs
    try:
        import torch.backends.cuda as cuda_backends
        torch.set_float32_matmul_precision('high')
        cuda_backends.matmul.allow_tf32 = True
        cuda_backends.cudnn.allow_tf32 = True
        # Reduce torch.compile graph breaks on scalar ops
        try:
            import torch._dynamo as dynamo
            dynamo.config.capture_scalar_outputs = True
        except Exception:
            pass
    except Exception:
        pass
    # Load configurations
    train_config = OmegaConf.load(train_config_path)
    # Determine model config path (CLI override wins), then resolve path robustly
    model_cfg_input = model_config_path or train_config.model_config
    cfg_base = Path(train_config_path).parent
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
    # Persist resolved path back into train_config for logging
    train_config.model_config = str(model_cfg_path)

    # Select model type/module
    model_type: str = model_config.get("model_type", DEFAULT_MODEL_TYPE)
    if model_type not in MODEL_REGISTRY:
        valid = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unsupported model_type '{model_type}'. Valid options: {valid}")

    registry_entry = MODEL_REGISTRY[model_type]
    lightning_module_cls = registry_entry["module"]
    training_title = registry_entry["title"]
    checkpoint_prefix = registry_entry["checkpoint_prefix"]
    
    print("\n" + "="*60)
    print(training_title)
    print("="*60)
    print(f"Model Type: {model_type}")
    print(f"Model Config: {model_cfg_path}")
    print(f"Train Config: {train_config_path}")
    print(f"WandB Project: {wandb_project}")
    print(f"WandB Run: {wandb_name or 'auto-generated'}")
    print(f"Fold: {train_config.data.current_fold}/{train_config.data.n_folds-1}")
    print("="*60 + "\n")
    
    # Initialize DataModule with K-Fold CV
    print("Initializing DataModule...")
    # Adjust loader aggressiveness during smoke tests to avoid system limits
    _is_smoke = bool(getattr(train_config, "smoke_test", False))
    preload_flag = bool(train_config.data.get('preload_patients', False))
    use_cache_server = bool(train_config.data.get('use_cache_server', False))
    dm_num_workers = train_config.data.num_workers if not _is_smoke else max(0, min(8, int(train_config.data.num_workers)))
    # When preloading all patients into the Dataset, avoid multiprocessing to prevent pickling/FD sharing of large caches
    if preload_flag and dm_num_workers > 0:
        print(f"Note: preload_patients=True → forcing num_workers=0 to avoid shared-memory mmap pressure.")
        dm_num_workers = 0
    dm_pin_memory = train_config.data.pin_memory if not _is_smoke else False
    dm_prefetch = train_config.data.get('prefetch_factor', 4)
    if _is_smoke:
        dm_prefetch = 2

    # Optionally ensure CacheServer is running (best-effort)
    if use_cache_server:
        # Try quick connect first via DataModule (will print warning if fails)
        pass

    datamodule = HMSDataModule(
        data_dir=train_config.data.data_dir,
        train_csv=train_config.data.train_csv,
        batch_size=train_config.batch_size,
        n_folds=train_config.data.n_folds,
        current_fold=train_config.data.current_fold,
        stratify_by_class=train_config.data.get('stratify_by_class', True),
        stratify_by_evaluators=train_config.data.get('stratify_by_evaluators', False),
        evaluator_bins=train_config.data.get('evaluator_bins', [0, 5, 10, 15, 20, 999]),
        min_evaluators=train_config.data.get('min_evaluators', 0),
        num_workers=dm_num_workers,
        pin_memory=dm_pin_memory,
        prefetch_factor=dm_prefetch if dm_num_workers > 0 else None,
        shuffle_seed=train_config.data.shuffle_seed,
        preload_patients=preload_flag,
        use_cache_server=use_cache_server,
        cache_host=str(train_config.data.get('cache_host', '127.0.0.1')),
        cache_port=int(train_config.data.get('cache_port', 50000)),
        cache_authkey=str(train_config.data.get('cache_authkey', 'hms-cache')),
    )
    
    # Setup to get class weights
    datamodule.setup(stage="fit")
    class_weights = datamodule.get_class_weights() if train_config.use_class_weights else None
    
    # Initialize Lightning Module
    print("Initializing Model...")
    model = lightning_module_cls(
        model_config=model_config.model,
        num_classes=model_config.model.num_classes,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.regularization.weight_decay,
        class_weights=class_weights,
        scheduler_config=train_config.scheduler,
        graph_laplacian_lambda=train_config.regularization.graph_laplacian_lambda,
        edge_weight_penalty=train_config.regularization.edge_weight_penalty,
        loss_type=train_config.loss,
    )
    
    # Print model info
    model_info = model.get_model_info()
    print(f"\nModel Architecture:")
    for key, value in model_info.items():
        label = key.replace('_', ' ').title()
        if isinstance(value, int):
            value_str = f"{value:,}"
        else:
            value_str = value
        print(f"  {label:<20} {value_str}")
    print()
    
    # WandB Logger
    logger_kwargs = {
        "project": wandb_project,
        "name": wandb_name,
        "save_dir": "logs",
        "log_model": True,
        "tags": list(getattr(train_config, "wandb_tags", []) or []),
        "group": getattr(train_config, "wandb_group", None),
        "job_type": getattr(train_config, "wandb_job_type", None),
        "anonymous": "allow",  # allow logging without explicit login
    }
    # Optional entity
    if hasattr(train_config, "wandb_entity") and train_config.wandb_entity:
        logger_kwargs["entity"] = train_config.wandb_entity

    # If smoke test, ensure tagging and run name reflects it
    if getattr(train_config, "smoke_test", False):
        logger_kwargs["tags"].append("smoke")
        if logger_kwargs["name"]:
            logger_kwargs["name"] = f"SMOKE-{logger_kwargs['name']}"
        else:
            logger_kwargs["name"] = "SMOKE"

    wandb_logger = WandbLogger(**logger_kwargs)
    
    # Configure WandB to log step-level metrics
    # Note: Metrics logged with on_step=True will appear as {metric}_step in WandB
    # Metrics logged with on_epoch=True will appear as {metric}_epoch in WandB
    wandb_logger.experiment.define_metric("train/loss_step", step_metric="trainer/global_step")
    wandb_logger.experiment.define_metric("train/loss_epoch", step_metric="epoch")
    
    # Log configuration to WandB
    wandb_logger.experiment.config.update({
        "model": OmegaConf.to_container(model_config, resolve=True),
        "training": OmegaConf.to_container(train_config, resolve=True),
    })
    
    # Callbacks
    callbacks = []
    
    # Model Checkpoint - save best model based on validation loss
    # Note: val/loss becomes val/loss_epoch when on_epoch=True
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{checkpoint_prefix}" + "-{epoch:02d}-{val/loss:.4f}",
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
        mixed_precision = bool(train_config.hardware.mixed_precision)
        device = getattr(train_config.hardware, 'device', 'auto')
    else:
        mixed_precision = bool(getattr(train_config, 'mixed_precision', False))
        device = getattr(train_config, 'device', 'auto')

    # Accelerator selection
    if device == 'cuda' or (device == 'auto' and torch.cuda.is_available()):
        accelerator = 'gpu'
    elif device == 'mps':
        accelerator = 'mps'
    elif device == 'cpu':
        accelerator = 'cpu'
    else:
        accelerator = 'auto'

    # Precision selection: prefer bf16 on Hopper/Ampere
    if accelerator == 'gpu' and torch.cuda.is_available():
        # Try BF16 mixed precision when available (H100/H200 and many Ampere GPUs)
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
    else:
        precision = 32

    # Optionally compile model for PyTorch 2.x
    if getattr(train_config, "compile_model", False) and hasattr(torch, "compile") and accelerator == 'gpu':
        try:
            compile_mode = str(getattr(train_config, "compile_mode", "reduce-overhead"))
            model = torch.compile(model, mode=compile_mode)
            print(f"Enabled torch.compile with mode={compile_mode}")
        except Exception as e:
            print(f"torch.compile failed, continuing without compile: {e}")

    # Enable cuDNN benchmark for faster LSTM/convolution if shapes are static
    try:
        if accelerator == 'gpu':
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
    except Exception:
        pass
    
    trainer_kwargs = dict(
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

    # Smoke test limits for super-fast runs
    if getattr(train_config, "smoke_test", False):
        trainer_kwargs.update({
            "max_epochs": 1,
            "limit_train_batches": 2,
            "limit_val_batches": 2,
            "limit_test_batches": 2,
            "num_sanity_val_steps": 0,
            "log_every_n_steps": 1,
        })

    trainer = Trainer(**trainer_kwargs)
    
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
    
    parser = argparse.ArgumentParser(description="Train HMS GNN models")
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/train.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Override path to model configuration file",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
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

    # Load config to allow defaults for WandB and to update fold
    train_cfg = OmegaConf.load(args.train_config)
    if args.fold is not None:
        train_cfg.data.current_fold = args.fold
        OmegaConf.save(train_cfg, args.train_config)
        print(f"Updated current_fold to {args.fold} in {args.train_config}")

    train(
        train_config_path=args.train_config,
        model_config_path=args.model_config,
        wandb_project=(args.wandb_project or train_cfg.get('wandb_project', 'hms-brain-activity')),
        wandb_name=(args.wandb_name or train_cfg.get('wandb_name', None)),
        resume_from_checkpoint=args.resume,
    )
