"""Debug NaN losses in training."""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from src.data import HMSDataModule
from src.lightning_trainer import HMSLightningModule


def check_data():
    """Check if data has NaN or invalid values."""
    print("\n" + "="*60)
    print("Checking Data for NaN/Inf")
    print("="*60)
    
    train_config = OmegaConf.load("configs/train.yaml")
    
    datamodule = HMSDataModule(
        data_dir=train_config.data.data_dir,
        train_csv=train_config.data.train_csv,
        batch_size=4,  # Small batch for testing
        n_folds=train_config.data.n_folds,
        current_fold=0,
        stratify_by_evaluators=train_config.data.stratify_by_evaluators,
        evaluator_bins=train_config.data.evaluator_bins,
        min_evaluators=train_config.data.get('min_evaluators', 0),
        num_workers=0,
        pin_memory=False,
        shuffle_seed=42,
    )
    
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    
    # Get first batch
    batch = next(iter(train_loader))
    
    print(f"\n✓ Loaded batch:")
    print(f"  - Targets shape: {batch['targets'].shape}")
    print(f"  - Targets: {batch['targets']}")
    print(f"  - Target min/max: {batch['targets'].min()}/{batch['targets'].max()}")
    print(f"  - Num EEG graphs: {len(batch['eeg_graphs'])}")
    print(f"  - Num Spec graphs: {len(batch['spec_graphs'])}")
    
    # Check targets
    if torch.isnan(batch['targets']).any():
        print("  ❌ TARGETS CONTAIN NaN!")
        return False
    
    if torch.isinf(batch['targets']).any():
        print("  ❌ TARGETS CONTAIN Inf!")
        return False
    
    if (batch['targets'] < 0).any() or (batch['targets'] >= 6).any():
        print(f"  ❌ TARGETS OUT OF RANGE [0, 5]: {batch['targets']}")
        return False
    
    print("  ✓ Targets are valid")
    
    # Check EEG graphs
    for i, graph in enumerate(batch['eeg_graphs'][:3]):  # Check first 3
        if torch.isnan(graph.x).any():
            print(f"  ❌ EEG graph {i} has NaN in node features!")
            return False
        if torch.isinf(graph.x).any():
            print(f"  ❌ EEG graph {i} has Inf in node features!")
            return False
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            if torch.isnan(graph.edge_attr).any():
                print(f"  ❌ EEG graph {i} has NaN in edge attributes!")
                return False
            if torch.isinf(graph.edge_attr).any():
                print(f"  ❌ EEG graph {i} has Inf in edge attributes!")
                return False
    
    print("  ✓ EEG graphs are valid")
    
    # Check Spec graphs
    for i, graph in enumerate(batch['spec_graphs'][:3]):  # Check first 3
        if torch.isnan(graph.x).any():
            print(f"  ❌ Spec graph {i} has NaN in node features!")
            return False
        if torch.isinf(graph.x).any():
            print(f"  ❌ Spec graph {i} has Inf in node features!")
            return False
    
    print("  ✓ Spec graphs are valid")
    
    return True, batch


def check_forward_pass(batch):
    """Check if forward pass produces NaN."""
    print("\n" + "="*60)
    print("Checking Forward Pass")
    print("="*60)
    
    model_config = OmegaConf.load("configs/model.yaml")
    train_config = OmegaConf.load("configs/train.yaml")
    
    model = HMSLightningModule(
        model_config=model_config.model,
        num_classes=6,
        learning_rate=0.001,
        weight_decay=0.0001,
        class_weights=None,
        scheduler_config=None,
        graph_laplacian_lambda=0.001,
        edge_weight_penalty=0.0,
    )
    
    model.eval()
    
    # Forward pass without regularization
    print("\n1. Forward pass (no regularization):")
    with torch.no_grad():
        logits = model(batch['eeg_graphs'], batch['spec_graphs'])
        
        print(f"  - Logits shape: {logits.shape}")
        print(f"  - Logits min/max: {logits.min():.4f}/{logits.max():.4f}")
        
        if torch.isnan(logits).any():
            print("  ❌ LOGITS CONTAIN NaN!")
            return False
        if torch.isinf(logits).any():
            print("  ❌ LOGITS CONTAIN Inf!")
            return False
        
        print("  ✓ Logits are valid")
    
    # Forward pass with intermediate
    print("\n2. Forward pass (with intermediate for regularization):")
    with torch.no_grad():
        logits, intermediate = model.model(
            batch['eeg_graphs'], 
            batch['spec_graphs'],
            return_intermediate=True
        )
        
        print(f"  - Logits shape: {logits.shape}")
        print(f"  - Intermediate keys: {intermediate.keys()}")
        
        if torch.isnan(logits).any():
            print("  ❌ LOGITS CONTAIN NaN!")
            return False
        
        print("  ✓ Forward pass with intermediate is valid")
    
    # Check loss computation
    print("\n3. Loss computation:")
    model.train()
    
    ce_loss = model.criterion(logits, batch['targets'])
    print(f"  - CE loss: {ce_loss.item():.4f}")
    
    if torch.isnan(ce_loss):
        print("  ❌ CE LOSS IS NaN!")
        print(f"     Logits stats: min={logits.min():.4f}, max={logits.max():.4f}")
        print(f"     Targets: {batch['targets']}")
        return False
    
    print("  ✓ CE loss is valid")
    
    # Check regularization
    print("\n4. Regularization computation:")
    from src.models.regularization import compute_graph_regularization
    
    eeg_reg = compute_graph_regularization(
        intermediate['eeg_graphs'],
        lambda_laplacian=0.001,
        lambda_edge=0.0,
    )
    
    spec_reg = compute_graph_regularization(
        intermediate['spec_graphs'],
        lambda_laplacian=0.001,
        lambda_edge=0.0,
    )
    
    print(f"  - EEG reg: {eeg_reg.item():.6f}")
    print(f"  - Spec reg: {spec_reg.item():.6f}")
    
    if torch.isnan(eeg_reg):
        print("  ❌ EEG REGULARIZATION IS NaN!")
        return False
    
    if torch.isnan(spec_reg):
        print("  ❌ SPEC REGULARIZATION IS NaN!")
        return False
    
    print("  ✓ Regularization is valid")
    
    # Total loss
    total_loss = ce_loss + eeg_reg + spec_reg
    print(f"\n5. Total loss: {total_loss.item():.4f}")
    
    if torch.isnan(total_loss):
        print("  ❌ TOTAL LOSS IS NaN!")
        return False
    
    print("  ✓ Total loss is valid")
    
    return True


def main():
    """Run all diagnostics."""
    print("\n" + "="*60)
    print("NaN Loss Diagnostic")
    print("="*60)
    
    # Check data
    result = check_data()
    if result is False:
        print("\n❌ Data check failed!")
        return False
    
    data_ok, batch = result
    if not data_ok:
        return False
    
    # Check forward pass
    if not check_forward_pass(batch):
        print("\n❌ Forward pass check failed!")
        return False
    
    print("\n" + "="*60)
    print("✓ ALL CHECKS PASSED!")
    print("="*60)
    print("\nIf training still shows NaN:")
    print("1. Check if it happens on first batch or later")
    print("2. Try reducing learning rate (e.g., 1e-4)")
    print("3. Try disabling mixed precision (set to false in config)")
    print("4. Check gradient clipping (currently set to 1.0)")
    print("5. Disable regularization temporarily (set lambda to 0)")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
