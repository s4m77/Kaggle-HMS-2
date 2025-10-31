# Data Preprocessing Pipeline

This directory contains utilities to preprocess raw EEG and Spectrogram data into PyTorch Geometric graphs.

## Overview

The preprocessing pipeline:
1. Loads raw EEG (50s) and Spectrogram (600s) data for each labeled sample
2. Extracts temporal windows with 50% overlap
3. Computes features and builds graphs:
   - **EEG**: 9 temporal graphs per sample (19 nodes, ~100-200 edges based on coherence)
   - **Spectrogram**: 119 temporal graphs per sample (4 nodes, 8 fixed spatial edges)
4. Saves processed graphs grouped by `patient_id`

## File Structure

```
src/data/
├── utils/
│   ├── eeg_process.py          # EEG feature extraction & graph building
│   └── spectrogram_process.py  # Spectrogram feature extraction & graph building
└── make_dataset.py              # Main preprocessing script
```

## Configuration

All hyperparameters are defined in `configs/graphs.yaml`:

### EEG Parameters
- **Sampling rate**: 200 Hz
- **Window size**: 10 seconds (2,000 samples)
- **Stride**: 5 seconds (50% overlap)
- **Channels**: 19 EEG channels (excludes EKG)
- **Frequency bands**: Delta, Theta, Alpha, Beta, Gamma
- **Edge creation**: Coherence threshold = 0.5

### Spectrogram Parameters
- **Duration**: 600 seconds
- **Window size**: 10 seconds
- **Stride**: 5 seconds (50% overlap)
- **Regions**: LL, RL, LP, RP (4 spatial regions)
- **Frequency bands**: Delta, Theta, Alpha, Beta, Gamma
- **Aggregation**: Mean pooling over frequency bins
- **Edges**: Fixed spatial connectivity (8 edges)

## Usage

### 1. Test the Pipeline (Quick)

Test on a single sample to verify everything works:

```bash
python test_pipeline.py
```

This will:
- Load configuration
- Process the first sample from `train.csv`
- Display shapes and statistics
- Verify graph construction

### 2. Run Full Preprocessing

Process all training data:

```bash
python src/data/make_dataset.py
```

This will:
- Process all ~106k samples from `train.csv`
- Group by `patient_id`
- Save one `.pt` file per patient in `data/processed/`
- Save metadata with patient IDs and sample counts

**Expected output:**
```
data/processed/
├── patient_12345.pt
├── patient_12346.pt
├── ...
└── metadata.pt
```

**Processing time**: ~30-60 minutes (depending on hardware)

## Output Format

### Patient Files

Each `patient_{patient_id}.pt` contains a dictionary:

```python
{
    label_id_1: {
        'eeg_graphs': [graph_0, graph_1, ..., graph_8],    # 9 PyG Data objects
        'spec_graphs': [graph_0, graph_1, ..., graph_118],  # 119 PyG Data objects
        'target': 0  # Class index (0-5)
    },
    label_id_2: {
        ...
    },
    ...
}
```

### Metadata File

`metadata.pt` contains:

```python
{
    'n_patients': 5321,
    'n_samples': 106000,
    'patient_ids': [12345, 12346, ...],
    'samples_per_patient': {12345: 20, 12346: 15, ...},
    'config': {...}  # Full configuration used
}
```

## Graph Structure

### EEG Graph
- **Nodes**: 19 (one per EEG channel)
- **Node features**: (19, 5) - 5 band powers per channel
- **Edges**: ~100-200 (coherence-based, threshold > 0.5)
- **Edge features**: (n_edges, 1) - coherence values

### Spectrogram Graph
- **Nodes**: 4 (LL, RL, LP, RP regions)
- **Node features**: (4, 5) - 5 aggregated band powers per region
- **Edges**: 8 (fixed spatial connectivity)
- **Edge features**: None

## Loading Processed Data

```python
import torch

# Load single patient
patient_data = torch.load('data/processed/patient_12345.pt')

# Access specific sample
sample = patient_data[label_id]
eeg_graphs = sample['eeg_graphs']      # List[Data] of length 9
spec_graphs = sample['spec_graphs']    # List[Data] of length 119
target = sample['target']              # int (0-5)

# Load metadata
metadata = torch.load('data/processed/metadata.pt')
all_patient_ids = metadata['patient_ids']
```

## Label Mapping

```python
{
    'Seizure': 0,
    'LPD': 1,      # Lateralized Periodic Discharges
    'GPD': 2,      # Generalized Periodic Discharges
    'LRDA': 3,     # Lateralized Rhythmic Delta Activity
    'GRDA': 4,     # Generalized Rhythmic Delta Activity
    'Other': 5     # Other/Normal
}
```

## Memory Requirements

- **RAM**: ~16-32 GB recommended for full preprocessing
- **Disk**: ~5-10 GB for processed data (all patients)
- **GPU**: Not required for preprocessing

## Troubleshooting

### Missing Files
- Ensure `data/raw/train.csv` exists
- Ensure `data/raw/train_eegs/` and `data/raw/train_spectrograms/` contain parquet files

### Out of Memory
- Process patients in batches by modifying `make_dataset.py`
- Reduce batch size in the processing loop

### Shape Mismatches
- Check that EEG files have correct length (should be ≥10,000 samples)
- Verify spectrogram files have 'time' column

## Next Steps

After preprocessing:
1. Implement dataset/dataloader in PyTorch
2. Split patients into train/val/test (80/10/10)
3. Build model architecture
4. Train and evaluate

## References

- EEG coherence: Computes connectivity between channels
- Welch's method: Estimates power spectral density
- PyTorch Geometric: Graph neural network library
