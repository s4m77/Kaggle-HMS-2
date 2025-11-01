"""Example script to test the HMSDataModule with StratifiedGroupKFold."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from src.data.graph_datamodule import HMSDataModule


def main():
    """Test the DataModule setup."""
    
    # Load config
    config = OmegaConf.load("configs/train.yaml")
    
    print("="*80)
    print("Testing HMSDataModule with StratifiedGroupKFold")
    print("="*80)
    
    # Create DataModule
    datamodule = HMSDataModule(
        data_dir=config.data.data_dir,
        train_csv=config.data.train_csv,
        batch_size=config.batch_size,
        n_folds=config.data.n_folds,
        current_fold=config.data.current_fold,
        stratify_by_evaluators=config.data.stratify_by_evaluators,
        evaluator_bins=config.data.evaluator_bins,
        min_evaluators=config.data.min_evaluators,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        shuffle_seed=config.data.shuffle_seed,
    )
    
    # Setup
    datamodule.setup(stage="fit")
    
    # Get dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    print(f"\nDataLoader Info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    # Test loading one batch
    print(f"\nTesting batch loading...")
    batch = next(iter(train_loader))
    
    print(f"\nBatch structure:")
    print(f"  EEG graphs: {len(batch['eeg_graphs'])} timesteps")
    print(f"    - Each timestep is a PyG Batch with {batch['eeg_graphs'][0].num_graphs} graphs")
    print(f"    - Node features shape: {batch['eeg_graphs'][0].x.shape}")
    print(f"    - Edge index shape: {batch['eeg_graphs'][0].edge_index.shape}")
    print(f"\n  Spec graphs: {len(batch['spec_graphs'])} timesteps")
    print(f"    - Each timestep is a PyG Batch with {batch['spec_graphs'][0].num_graphs} graphs")
    print(f"    - Node features shape: {batch['spec_graphs'][0].x.shape}")
    print(f"    - Edge index shape: {batch['spec_graphs'][0].edge_index.shape}")
    print(f"\n  Targets shape: {batch['targets'].shape}")
    print(f"  Target values: {batch['targets'].tolist()}")
    print(f"\n  Patient IDs: {batch['patient_ids']}")
    print(f"  Label IDs: {batch['label_ids']}")
    
    print(f"\n{'='*80}")
    print("✓ DataModule test successful!")
    print("="*80)
    
    # Print class weights
    print(f"\nClass weights for loss balancing:")
    class_weights = datamodule.get_class_weights()
    if class_weights is not None:
        label_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
        for i, (name, weight) in enumerate(zip(label_names, class_weights)):
            print(f"  {name:10s} (idx={i}): {weight:.4f}")


if __name__ == "__main__":
    main()
