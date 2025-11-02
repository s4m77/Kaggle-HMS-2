"""Quick check to verify training setup is ready."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from src.data import HMSDataModule
from src.lightning_trainer import HMSLightningModule


def check_configs():
    """Check if config files are properly structured."""
    print("=" * 60)
    print("Checking Configuration Files")
    print("=" * 60)
    
    # Load configs
    try:
        model_config = OmegaConf.load("configs/model.yaml")
        print("✓ Model config loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model config: {e}")
        return False
    
    try:
        train_config = OmegaConf.load("configs/train.yaml")
        print("✓ Train config loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load train config: {e}")
        return False
    
    # Check required fields
    required_model_fields = ['model.eeg_encoder', 'model.spec_encoder', 'model.fusion', 'model.classifier', 'model.num_classes']
    for field in required_model_fields:
        if OmegaConf.select(model_config, field) is None:
            print(f"✗ Missing field in model config: {field}")
            return False
    print(f"✓ All required model fields present")
    
    required_train_fields = ['batch_size', 'num_epochs', 'learning_rate', 'regularization.weight_decay', 
                             'regularization.graph_laplacian_lambda', 'data.n_folds', 'data.current_fold']
    for field in required_train_fields:
        if OmegaConf.select(train_config, field) is None:
            print(f"✗ Missing field in train config: {field}")
            return False
    print(f"✓ All required training fields present")
    
    print()
    return True


def check_datamodule():
    """Check if DataModule initializes correctly."""
    print("=" * 60)
    print("Checking DataModule")
    print("=" * 60)
    
    try:
        train_config = OmegaConf.load("configs/train.yaml")
        
        datamodule = HMSDataModule(
            data_dir=train_config.data.data_dir,
            train_csv=train_config.data.train_csv,
            batch_size=train_config.batch_size,
            n_folds=train_config.data.n_folds,
            current_fold=train_config.data.current_fold,
            stratify_by_evaluators=train_config.data.stratify_by_evaluators,
            evaluator_bins=train_config.data.evaluator_bins,
            min_evaluators=train_config.data.get('min_evaluators', 0),
            num_workers=0,  # Use 0 for testing
            pin_memory=False,
            shuffle_seed=train_config.data.shuffle_seed,
        )
        print("✓ DataModule initialized successfully")
        
        # Setup
        datamodule.setup(stage="fit")
        print(f"✓ DataModule setup completed")
        print(f"  - Train samples: {len(datamodule.train_dataset)}")
        print(f"  - Val samples: {len(datamodule.val_dataset)}")
        print(f"  - Class weights: {datamodule.class_weights.tolist() if datamodule.class_weights is not None else 'None'}")
        
        # Try creating dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        print(f"✓ DataLoaders created successfully")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ DataModule check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model():
    """Check if Lightning Module initializes correctly."""
    print("=" * 60)
    print("Checking Lightning Module")
    print("=" * 60)
    
    try:
        model_config = OmegaConf.load("configs/model.yaml")
        train_config = OmegaConf.load("configs/train.yaml")
        
        model = HMSLightningModule(
            model_config=model_config.model,
            num_classes=model_config.model.num_classes,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.regularization.weight_decay,
            class_weights=None,  # Not testing with weights
            scheduler_config=train_config.scheduler,
            graph_laplacian_lambda=train_config.regularization.graph_laplacian_lambda,
            edge_weight_penalty=train_config.regularization.edge_weight_penalty,
        )
        print("✓ Lightning Module initialized successfully")
        
        # Get model info
        model_info = model.get_model_info()
        print(f"✓ Model architecture:")
        print(f"  - EEG output dim: {model_info['eeg_output_dim']}")
        print(f"  - Spec output dim: {model_info['spec_output_dim']}")
        print(f"  - Fusion output dim: {model_info['fusion_output_dim']}")
        print(f"  - Num classes: {model_info['num_classes']}")
        print(f"  - Total params: {model_info['total_params']:,}")
        print(f"  - Trainable params: {model_info['trainable_params']:,}")
        
        # Check regularization parameters
        print(f"✓ Regularization:")
        print(f"  - Weight decay: {model.weight_decay}")
        print(f"  - Graph Laplacian λ: {model.graph_laplacian_lambda}")
        print(f"  - Edge penalty: {model.edge_weight_penalty}")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Lightning Module check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("HMS Training Setup Verification")
    print("=" * 60 + "\n")
    
    results = {
        'Configs': check_configs(),
        'DataModule': check_datamodule(),
        'Lightning Module': check_model(),
    }
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to start training!")
        print("\nTo train, run:")
        print("  python src/train.py")
        print("\nOr for a specific fold:")
        print("  python src/train.py --fold 0")
        print("\nOr to run all 5 folds:")
        print("  for fold in {0..4}; do python src/train.py --fold $fold; done")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above before training.")
    
    print()
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
