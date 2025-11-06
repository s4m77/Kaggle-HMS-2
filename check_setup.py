"""Check and verify all dependencies are installed for training."""

import sys
from pathlib import Path


def check_imports():
    """Check if all required packages are installed."""
    print("\n" + "="*60)
    print("Checking Dependencies")
    print("="*60 + "\n")
    
    required_packages = {
        'torch': 'PyTorch',
        'pytorch_lightning': 'PyTorch Lightning',
        'torch_geometric': 'PyTorch Geometric',
        'wandb': 'Weights & Biases',
        'omegaconf': 'OmegaConf',
        'torchmetrics': 'TorchMetrics',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
    }
    
    missing = []
    installed = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            installed.append(f"✓ {name}")
        except ImportError:
            missing.append(f"✗ {name} (import: {package})")
    
    # Print results
    if installed:
        print("Installed packages:")
        for pkg in installed:
            print(f"  {pkg}")
    
    if missing:
        print("\nMissing packages:")
        for pkg in missing:
            print(f"  {pkg}")
        print("\nPlease install missing packages:")
        print("  conda env update -f environment.yaml")
        print("  or")
        print("  pip install pytorch-lightning wandb torchmetrics")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_data():
    """Check if preprocessed data exists."""
    print("\n" + "="*60)
    print("Checking Data")
    print("="*60 + "\n")
    
    data_dir = Path("data/processed")
    
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        print("\nPlease run preprocessing first:")
        print("  python src/data/make_dataset.py")
        return False
    
    patient_files = list(data_dir.glob("patient_*.pt"))
    
    if not patient_files:
        print(f"✗ No patient files found in {data_dir}")
        print("\nPlease run preprocessing first:")
        print("  python src/data/make_dataset.py")
        return False
    
    print(f"✓ Found {len(patient_files)} patient files in {data_dir}")
    return True


def check_wandb():
    """Check WandB login status."""
    print("\n" + "="*60)
    print("Checking WandB")
    print("="*60 + "\n")
    
    try:
        import wandb
        
        # Try to get API key
        api_key = wandb.api.api_key
        
        if api_key:
            print("✓ WandB is configured and logged in")
            return True
        else:
            print("✗ WandB API key not found")
            print("\nPlease login to WandB:")
            print("  wandb login")
            return False
            
    except Exception as e:
        print(f"✗ WandB check failed: {e}")
        print("\nPlease login to WandB:")
        print("  wandb login")
        return False


def check_cuda():
    """Check CUDA/GPU availability and details."""
    print("\n" + "="*60)
    print("Checking CUDA / GPU")
    print("="*60 + "\n")
    try:
        import torch, os
        from multiprocessing import cpu_count

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            num = torch.cuda.device_count()
            print(f"CUDA device count: {num}")
            for i in range(num):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                mem_total = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                print(f" - GPU {i}: {name}, CC {cap[0]}.{cap[1]}, VRAM ~{mem_total} GB")
            # BF16 support check
            bf16 = hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
            print(f"BF16 supported: {bf16}")
        else:
            print("No CUDA device detected.")

        # CPU/threads info
        try:
            cores = cpu_count()
        except Exception:
            cores = os.cpu_count() or 1
        print(f"CPU cores: {cores}")
        return True
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        return False


def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("HMS Training Setup Check")
    print("="*60)
    
    checks = [
        ("Dependencies", check_imports),
        ("CUDA/GPU", check_cuda),
        ("Data", check_data),
        ("WandB", check_wandb),
    ]
    
    all_passed = True
    for name, check_func in checks:
        if not check_func():
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All checks passed! Ready to train.")
        print("\nTo start training:")
        print("  python src/train.py")
        print("\nOr with custom config:")
        print("  python src/train.py --config configs/model.yaml --wandb-project my-project")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
