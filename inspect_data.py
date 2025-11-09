"""
Helper script to inspect preprocessed data.
Shows statistics and examples of processed graphs.
"""

import torch
from pathlib import Path


def inspect_processed_data():
    """Inspect preprocessed data and show statistics."""
    print("=" * 80)
    print("Inspecting Preprocessed Data")
    print("=" * 80)
    
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        print("\n❌ Error: Processed data directory not found!")
        print("   Please run preprocessing first:")
        print("   python src/data/make_dataset.py")
        return
    
    # Load metadata
    metadata_path = processed_dir / "metadata.pt"
    if not metadata_path.exists():
        print("\n⚠️  Warning: metadata.pt not found!")
        print("   Metadata will be created from existing patient files...")
        
        # Count patient files manually
        patient_files = list(processed_dir.glob("patient_*.pt"))
        if len(patient_files) == 0:
            print("\n❌ Error: No patient files found!")
            print("   Please run preprocessing first:")
            print("   python src/data/make_dataset.py")
            return
        
        patient_ids = [int(f.stem.split('_')[1]) for f in patient_files]
        
        # Load first patient as example
        example_patient_id = patient_ids[0]
        patient_path = processed_dir / f"patient_{example_patient_id}.pt"
        patient_data = torch.load(patient_path, weights_only=False)
        
        print(f"\n  Found {len(patient_files)} patient files")
        print(f"  Metadata is being updated as preprocessing continues...")
        
        metadata = {
            'n_patients': len(patient_files),
            'n_samples': 'Unknown (being processed)',
            'patient_ids': sorted(patient_ids),
        }
    else:
        print("\n[1] Loading metadata...")
        metadata = torch.load(metadata_path, weights_only=False)
    
    print(f"\n  Dataset Statistics:")
    print(f"  {'─' * 60}")
    print(f"  Total patients:        {metadata['n_patients']:,}")
    if isinstance(metadata['n_samples'], int):
        print(f"  Total samples:         {metadata['n_samples']:,}")
        print(f"  Avg samples/patient:   {metadata['n_samples'] / metadata['n_patients']:.1f}")
    else:
        print(f"  Total samples:         {metadata['n_samples']}")
    
    # Samples per patient distribution (if available)
    if 'samples_per_patient' in metadata:
        samples_per_patient = list(metadata['samples_per_patient'].values())
        print(f"\n  Samples per patient distribution:")
        print(f"    Min:     {min(samples_per_patient)}")
        print(f"    Max:     {max(samples_per_patient)}")
        print(f"    Median:  {sorted(samples_per_patient)[len(samples_per_patient)//2]}")
    
    # Load example patient
    print("\n[2] Loading example patient data...")
    example_patient_id = metadata['patient_ids'][0]
    patient_path = processed_dir / f"patient_{example_patient_id}.pt"
    patient_data = torch.load(patient_path, weights_only=False)
    
    print(f"\n  Patient ID: {example_patient_id}")
    print(f"  Number of samples: {len(patient_data)}")
    
    # Inspect first sample
    first_label_id = list(patient_data.keys())[0]
    sample = patient_data[first_label_id]
    
    print(f"\n[3] Inspecting sample (label_id={first_label_id})...")
    print(f"\n  EEG Graphs:")
    print(f"  {'─' * 60}")
    print(f"    Number of graphs:     {len(sample['eeg_graphs'])}")
    
    if len(sample['eeg_graphs']) > 0:
        g = sample['eeg_graphs'][0]
        print(f"    Graph 0 details:")
        print(f"      Nodes:             {g.x.shape[0]}")
        print(f"      Node features:     {g.x.shape}")
        print(f"      Edges:             {g.edge_index.shape[1]}")
        if hasattr(g, 'edge_attr') and g.edge_attr is not None:
            print(f"      Edge features:     {g.edge_attr.shape}")
        
        print(f"\n    Sample node features (first 3 channels):")
        print(f"      {g.x[:3]}")
    
    print(f"\n  Spectrogram Graphs:")
    print(f"  {'─' * 60}")
    print(f"    Number of graphs:     {len(sample['spec_graphs'])}")
    
    if len(sample['spec_graphs']) > 0:
        g = sample['spec_graphs'][0]
        print(f"    Graph 0 details:")
        print(f"      Nodes:             {g.x.shape[0]}")
        print(f"      Node features:     {g.x.shape}")
        print(f"      Edges:             {g.edge_index.shape[1]}")
        if hasattr(g, 'edge_attr') and g.edge_attr is not None:
            print(f"      Edge features:     {g.edge_attr.shape}")
        
        print(f"\n    Node features (all 4 regions):")
        print(f"      {g.x}")
    
    print(f"\n  Target:")
    print(f"  {'─' * 60}")
    print(f"    Class index:          {sample['target']}")
    
    # Class distribution
    print(f"\n[4] Class distribution across all patients...")
    
    label_names = {
        0: 'Seizure',
        1: 'LPD',
        2: 'GPD',
        3: 'LRDA',
        4: 'GRDA',
        5: 'Other'
    }
    
    class_counts = {i: 0 for i in range(6)}
    
    # Count all samples
    for patient_id in metadata['patient_ids'][:10]:  # Sample first 10 patients
        patient_path = processed_dir / f"patient_{patient_id}.pt"
        if patient_path.exists():
            patient_data = torch.load(patient_path, weights_only=False)
            for label_id, sample in patient_data.items():
                class_counts[sample['target']] += 1
    
    print(f"\n  Class counts (from first 10 patients):")
    print(f"  {'─' * 60}")
    total = sum(class_counts.values())
    for idx, count in class_counts.items():
        pct = 100 * count / total if total > 0 else 0
        print(f"    {idx}: {label_names[idx]:10s}  {count:6d} ({pct:5.1f}%)")
    
    # Disk usage
    print(f"\n[5] Disk usage...")
    total_size = sum(f.stat().st_size for f in processed_dir.glob("*.pt"))
    print(f"  Total size: {total_size / 1024**3:.2f} GB")
    print(f"  Avg per patient: {total_size / metadata['n_patients'] / 1024**2:.2f} MB")
    
    print("\n" + "=" * 80)
    print("✓ Inspection complete!")
    print("=" * 80)
    
    # Usage example
    print("\nTo load data in your training script:")
    print("```python")
    print("import torch")
    print("from pathlib import Path")
    print("")
    print("# Load metadata")
    print("metadata = torch.load('data/processed/metadata.pt')")
    print("patient_ids = metadata['patient_ids']")
    print("")
    print("# Load specific patient")
    print(f"patient_data = torch.load('data/processed/patient_{example_patient_id}.pt')")
    print("")
    print("# Access sample")
    print(f"sample = patient_data[{first_label_id}]")
    print("eeg_graphs = sample['eeg_graphs']    # List of 9 graphs")
    print("spec_graphs = sample['spec_graphs']  # List of 119 graphs")
    print("target = sample['target']            # Class index 0-5")
    print("```")
    print()


if __name__ == "__main__":
    inspect_processed_data()