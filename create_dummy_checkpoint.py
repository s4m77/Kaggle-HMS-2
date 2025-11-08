"""
Create a dummy checkpoint for testing explain_model.py
This allows you to test the explanation pipeline without full training.
"""

import torch
from pathlib import Path
from src.lightning_trainer.graph_lightning_module import HMSLightningModule

# Create output directory
checkpoint_dir = Path("checkpoints/dummy_test")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

print("Creating dummy checkpoint for testing...")

# Create a dummy Lightning module with minimal config
dummy_module = HMSLightningModule(
    model_config={
        'eeg_encoder': None,  # Will use defaults
        'spec_encoder': None,
        'fusion': None,
        'classifier': None,
    },
    num_classes=6,
    learning_rate=1e-3,
    loss_type="KLDivLoss"
)

# Save the checkpoint
checkpoint_path = checkpoint_dir / "dummy_model.ckpt"
torch.save(dummy_module, checkpoint_path)

print(f"✅ Dummy checkpoint created at: {checkpoint_path}")
print(f"   File size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
print(f"\nYou can now test explain_model.py with:")
print(f"  python3 src/explain_model.py \\")
print(f"    --model_path {checkpoint_path} \\")
print(f"    --data_path data/processed/patient_10012.pt")
