"""Test script to verify HMSDataset and HMSDataModule functionality."""

import sys
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.graph_datamodule import HMSDataModule


@pytest.fixture
def datamodule():
    """Create a DataModule instance for testing."""
    return HMSDataModule(
        data_dir="data/processed",
        train_csv="data/raw/train_unique.csv",
        batch_size=4,
        n_folds=5,
        current_fold=0,
        stratify_by_evaluators=True,
        evaluator_bins=[0, 5, 10, 15, 20, 999],
        min_evaluators=0,
        num_workers=0,  # 0 for testing
        pin_memory=False,
        shuffle_seed=42,
    )


def test_datamodule_initialization(datamodule):
    """Test that DataModule initializes correctly."""
    assert datamodule.n_folds == 5
    assert datamodule.current_fold == 0
    assert datamodule.batch_size == 4
    assert datamodule.stratify_by_evaluators is True
    print("✓ DataModule initialization test passed")


def test_datamodule_setup(datamodule):
    """Test that DataModule setup works correctly."""
    datamodule.setup(stage="fit")
    
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.metadata_df is not None
    assert datamodule.class_weights is not None
    
    # Check that folds are assigned
    assert 'fold' in datamodule.metadata_df.columns
    assert datamodule.metadata_df['fold'].nunique() == 5
    
    print(f"✓ Setup test passed - Train: {len(datamodule.train_dataset)}, Val: {len(datamodule.val_dataset)}")


def test_datamodule_stratification(datamodule):
    """Test that stratification works correctly."""
    datamodule.setup(stage="fit")
    
    # Check that evaluator bins are created
    assert 'evaluator_bin' in datamodule.metadata_df.columns
    assert 'total_evaluators' in datamodule.metadata_df.columns
    
    # Check that bins are distributed across folds
    for fold in range(5):
        fold_data = datamodule.metadata_df[datamodule.metadata_df['fold'] == fold]
        assert len(fold_data) > 0, f"Fold {fold} is empty"
    
    print("✓ Stratification test passed")


def test_datamodule_patient_separation(datamodule):
    """Test that patients are not shared between train and val."""
    datamodule.setup(stage="fit")
    
    train_df = datamodule.metadata_df[datamodule.metadata_df['fold'] != 0]
    val_df = datamodule.metadata_df[datamodule.metadata_df['fold'] == 0]
    
    train_patients = set(train_df['patient_id'].unique())
    val_patients = set(val_df['patient_id'].unique())
    
    overlap = train_patients & val_patients
    assert len(overlap) == 0, f"Found {len(overlap)} patients in both train and val!"
    
    print(f"✓ Patient separation test passed - No overlap between {len(train_patients)} train and {len(val_patients)} val patients")


def test_dataloader_creation(datamodule):
    """Test that DataLoaders can be created."""
    datamodule.setup(stage="fit")
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    
    print(f"✓ DataLoader creation test passed - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")


def test_batch_structure(datamodule):
    """Test that batches have correct structure."""
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    
    batch = next(iter(train_loader))
    
    # Check keys
    assert 'eeg_graphs' in batch
    assert 'spec_graphs' in batch
    assert 'targets' in batch
    assert 'patient_ids' in batch
    assert 'label_ids' in batch
    
    # Check EEG graphs (should have 9 timesteps)
    assert isinstance(batch['eeg_graphs'], list)
    assert len(batch['eeg_graphs']) == 9
    
    # Check Spec graphs (subsampled to 40 timesteps)
    assert isinstance(batch['spec_graphs'], list)
    assert len(batch['spec_graphs']) == 40
    
    # Check targets shape
    assert batch['targets'].shape[0] == 4  # batch size
    
    print(f"✓ Batch structure test passed")
    print(f"  - EEG: {len(batch['eeg_graphs'])} timesteps")
    print(f"  - Spec: {len(batch['spec_graphs'])} timesteps")
    print(f"  - Targets: {batch['targets'].shape}")


def test_class_weights(datamodule):
    """Test that class weights are computed."""
    datamodule.setup(stage="fit")
    
    class_weights = datamodule.get_class_weights()
    
    assert class_weights is not None
    assert len(class_weights) == 6  # 6 classes
    assert all(w > 0 for w in class_weights)
    
    print(f"✓ Class weights test passed: {class_weights.tolist()}")


def test_datamodule():
    """Legacy test function for backward compatibility."""
    print("\n" + "="*60)
    print("Testing HMSDataModule with StratifiedGroupKFold")
    print("="*60)
    
    # Initialize datamodule with new parameters
    dm = HMSDataModule(
        data_dir="data/processed",
        train_csv="data/raw/train_unique.csv",
        batch_size=4,
        n_folds=5,
        current_fold=0,
        stratify_by_evaluators=True,
        evaluator_bins=[0, 5, 10, 15, 20, 999],
        min_evaluators=0,
        num_workers=0,  # 0 for testing
        pin_memory=False,
        shuffle_seed=42,
    )
    
    # Setup
    print("\nSetting up datasets...")
    dm.setup(stage="fit")
    
    # Get dataloaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    print(f"\nDataLoader info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    # Test one batch
    print(f"\nTesting one training batch...")
    batch = next(iter(train_loader))
    
    print(f"\nBatch structure:")
    print(f"  EEG graphs:  {len(batch['eeg_graphs'])} timesteps")
    print(f"    - First timestep batch: {batch['eeg_graphs'][0]}")
    print(f"    - Nodes per graph: {batch['eeg_graphs'][0].num_nodes // 4}")  # Divide by batch size
    print(f"  Spec graphs: {len(batch['spec_graphs'])} timesteps")
    print(f"    - First timestep batch: {batch['spec_graphs'][0]}")
    print(f"    - Nodes per graph: {batch['spec_graphs'][0].num_nodes // 4}")  # Divide by batch size
    print(f"  Targets shape: {batch['targets'].shape}")
    print(f"  Targets: {batch['targets']}")
    print(f"  Patient IDs: {batch['patient_ids']}")
    print(f"  Label IDs: {batch['label_ids']}")
    
    # Test class weights
    class_weights = dm.get_class_weights()
    print(f"\nClass weights: {class_weights}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_datamodule()

