# Cross-Modal Attention Explainability Guide

## Overview

This guide explains how to use the cross-modal attention explainability tools to interpret your HMS Multi-Modal GNN model. These tools help you understand **how the model reconciles EEG and Spectrogram modalities** when making brain activity predictions.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Installation & Setup](#installation--setup)
4. [Core Concepts](#core-concepts)
5. [Usage Examples](#usage-examples)
6. [Visualizations](#visualizations)
7. [Interpretation Guide](#interpretation-guide)
8. [API Reference](#api-reference)

---

## Quick Start

### 5-Minute Example

```python
import torch
from src.lightning_trainer import HMSLightningModule
from src.explainability import ExplainabilityCapture

# 1. Load trained model
model = HMSLightningModule.load_from_checkpoint("checkpoints/hms-best.ckpt")
model.eval()

# 2. Create explainability tool
explainer = ExplainabilityCapture(model, num_heads=8)

# 3. Explain a batch
with torch.no_grad():
    batch = next(iter(test_loader))  # Your test batch
    results = explainer.explain_batch(batch)

# 4. Generate report
from src.explainability import create_explainability_report
report = create_explainability_report(results)
print(report)

# 5. Visualize
from src.explainability.visualizations import plot_cross_modal_attention
fig = plot_cross_modal_attention(results['attention_dict'], sample_idx=0)
fig.show()
```

---

## Architecture Overview

Your model processes two modalities independently, then fuses them:

```
Input Data
    ↓
┌─────────────────────────────────────┐
│                                     │
│  EEG Stream (9 graphs)              │  Spec Stream (119 graphs)
│  - 19 channels                      │  - 4 spatial regions
│  - Temporal windows                 │  - Temporal windows
│                                     │
├─ GAT Encoder ─────────────────┬─────┤─ GAT Encoder ─────────────────┐
│  (Graph Attention Network)    │     │  (Graph Attention Network)    │
│  256-dim features             │     │  256-dim features             │
│                               │     │                               │
├─ Temporal Encoder (BiLSTM)    │     ├─ Temporal Encoder (BiLSTM)    │
│  Process temporal sequence    │     │  Process temporal sequence    │
│  256-dim output               │     │  256-dim output               │
└───────────────────────────────┼─────┘───────────────────────────────┘
                                │
                                ↓
                    ┌───────────────────────┐
                    │                       │
                    │ CROSS-MODAL FUSION    │  ← Explainability Focus!
                    │ (Attention-based)     │
                    │                       │
                    │ - EEG ↔ Spec          │
                    │   bidirectional       │
                    │   attention           │
                    │                       │
                    └───────────────────────┘
                                │
                                ↓
                    ┌───────────────────────┐
                    │  Classifier (MLP)     │
                    │                       │
                    │  512-dim → 6 classes  │
                    └───────────────────────┘
                                │
                                ↓
                          Prediction
```

**Key Point**: The cross-modal fusion is where the model decides "how much to trust EEG vs Spectrogram for this sample."

---

## Installation & Setup

### Requirements

```bash
pip install torch pytorch-lightning torch-geometric
pip install matplotlib seaborn scipy numpy
```

### Verify Installation

```python
from src.explainability import ExplainabilityCapture
from src.explainability.visualizations import plot_cross_modal_attention
print("✓ Explainability tools loaded successfully")
```

---

## Core Concepts

### 1. Cross-Modal Attention

Your fusion module uses **8-head attention** to compute:

- **EEG→Spectrogram Attention**: How much each EEG feature should pay attention to each Spectrogram feature
- **Spectrogram→EEG Attention**: How much each Spectrogram feature should pay attention to each EEG feature

Think of it like:
- A radiologist (EEG) asking "which spectrogram patterns should I focus on?"
- A cardiologist (Spectrogram) asking "which EEG patterns should I focus on?"

### 2. Attention Weights

Attention weights are shape: `(batch_size, num_heads, query_len, key_len)`

- **batch_size**: Number of samples in batch
- **num_heads**: 8 independent attention heads (each learns different patterns)
- **query_len**: Number of query positions (usually 1 after encoding)
- **key_len**: Number of key positions (usually 1 after encoding)

Values are normalized in range [0, 1], summing to 1 across keys for each query-head pair.

### 3. Attention Statistics

We compute several diagnostic statistics:

| Metric | Meaning | Interpretation |
|--------|---------|-----------------|
| **mean** | Average attention weight | Higher = more uniform attention distribution |
| **std** | Standard deviation | Higher = more peaked/focused attention |
| **max** | Maximum attention weight | Closer to 1.0 = one dominant position |
| **entropy** | Shannon entropy (0-1) | 1.0 = diffuse, 0.0 = sharp/focused |
| **sparsity** | % weights < 0.1 | Higher = sparser attention |

### 4. Modality Alignment

**Alignment score** = correlation between EEG→Spec and Spec→EEG attention patterns

- Score near **+1.0**: Modalities agree strongly ("yes, these patterns go together")
- Score near **0.0**: Modalities are independent ("patterns are unrelated")
- Score near **-1.0**: Modalities disagree ("inverse relationship")

---

## Usage Examples

### Example 1: Basic Batch Analysis

```python
from src.explainability import ExplainabilityCapture, create_explainability_report

# Load model and explainer
model = HMSLightningModule.load_from_checkpoint("checkpoints/best.ckpt")
explainer = ExplainabilityCapture(model)

# Analyze batch
with torch.no_grad():
    batch = next(iter(test_loader))
    results = explainer.explain_batch(batch)

# Print report
report = create_explainability_report(results)
print(report)
```

**Output Structure**:
```
BATCH-LEVEL STATISTICS
- EEG → Spectrogram Attention: mean=0.5234, std=0.2345, ...
- Spectrogram → EEG Attention: mean=0.4892, std=0.2156, ...
- Modality Alignment: 0.6543

PER-SAMPLE ANALYSIS
Sample 0:
  Predicted: 2 (LPD, confidence: 0.8234)
  Ground Truth: 2 ✓ CORRECT
  Modality Agreement: 0.7234
  Top EEG→Spec attention: (0, 0, 0.9123)
  Top Spec→EEG attention: (0, 0, 0.8756)
```

### Example 2: Analyze Single Sample with Visualizations

```python
from src.explainability.visualizations import (
    plot_cross_modal_attention,
    plot_head_contributions,
    plot_modality_alignment,
)

# Get results for one batch
results = explainer.explain_batch(batch)
attention_dict = results['attention_dict']

# Visualize cross-modal attention for sample 0
fig = plot_cross_modal_attention(attention_dict, sample_idx=0)
fig.savefig("attention_sample_0.png")

# Show per-head contributions
fig = plot_head_contributions(attention_dict)
fig.savefig("head_contributions.png")

# Show modality alignment
fig = plot_modality_alignment(attention_dict, sample_idx=0)
fig.savefig("modality_alignment.png")
```

### Example 3: Compare Correct vs Incorrect Predictions

```python
# Analyze entire test set
all_results = []
for batch in test_loader:
    with torch.no_grad():
        results = explainer.explain_batch(batch)
    all_results.append(results)

# Separate correct and incorrect predictions
correct_samples = []
incorrect_samples = []

for results in all_results:
    predictions = results['predictions']
    targets = results['targets']
    per_sample = results['per_sample_explanations']
    
    for i, expl in enumerate(per_sample):
        if expl['correct']:
            correct_samples.append(expl)
        else:
            incorrect_samples.append(expl)

# Compare modality agreement
correct_agreement = [s['modality_agreement'] for s in correct_samples]
incorrect_agreement = [s['modality_agreement'] for s in incorrect_samples]

import numpy as np
print(f"Correct samples: avg modality agreement = {np.mean(correct_agreement):.4f}")
print(f"Incorrect samples: avg modality agreement = {np.mean(incorrect_agreement):.4f}")
```

### Example 4: Head Specialization Analysis

```python
from src.explainability.attention_analysis import head_wise_attention_analysis

# Get per-head statistics
head_analysis = head_wise_attention_analysis(
    results['attention_dict'],
    num_heads=8
)

# Show which heads are most specialized
print("\nEEG → Spectrogram Head Specialization:")
for head_idx, stats in head_analysis['eeg_to_spec'].items():
    entropy = stats['specialization_score']
    print(f"  Head {head_idx}: entropy={entropy:.4f} (lower=more specialized)")
    print(f"    Top pairs: {stats['top_pairs']}")
```

### Example 5: Find Attention Patterns for Specific Class

```python
# Analyze all seizure predictions
seizure_samples = [
    expl for expl in all_explanations 
    if expl['predicted_class'] == 0  # 0 = Seizure
]

# Extract attention patterns
seizure_alignments = [s['modality_agreement'] for s in seizure_samples]
seizure_entropy = [
    np.mean(list(s['head_entropies'].values()))
    for s in seizure_samples
]

print(f"Seizure class (n={len(seizure_samples)}):")
print(f"  Avg modality agreement: {np.mean(seizure_alignments):.4f}")
print(f"  Avg attention entropy: {np.mean(seizure_entropy):.4f}")
```

---

## Visualizations

### 1. Cross-Modal Attention Heatmap

```python
fig = plot_cross_modal_attention(attention_dict, sample_idx=0)
```

Shows two heatmaps:
- **Left**: EEG queries attending to Spectrogram keys
- **Right**: Spectrogram queries attending to EEG keys

**Interpretation**:
- Bright spots = strong attention connections
- Uniform color = diffuse attention (less interpretable)
- Sparse color = focused attention (clear decision)

### 2. Per-Head Contributions

```python
fig = plot_head_contributions(attention_dict)
```

Shows bar chart of mean attention strength per head.

**Interpretation**:
- Different heights = heads have different "importance"
- Lower heights might mean certain heads are less active
- Compare EEG→Spec vs Spec→EEG to see modality asymmetry

### 3. Modality Alignment

```python
fig = plot_modality_alignment(attention_dict, sample_idx=0)
```

Three-panel visualization:
1. **Top**: Per-head forward-backward correlation
   - Green bars: high agreement (> 0.5)
   - Orange bars: medium agreement
   - Red bars: disagreement (< 0)
2. **Bottom-left**: Averaged EEG→Spec attention
3. **Bottom-right**: Averaged Spec→EEG attention

**Interpretation**:
- If many heads are red/orange: modalities might be in conflict for this sample
- If most heads are green: good consensus between modalities
- Compare left/right heatmaps to see if one modality dominates

---

## Interpretation Guide

### What Do Different Attention Patterns Mean?

#### Pattern 1: Focused Attention (Sparse, High Max)
```
Statistics: max=0.95, entropy=0.2, sparsity=85%
Interpretation: 
  ✓ Model is confident and focused
  ✓ Clear decision rule
  ✓ Interpretable (one pathway dominates)
```

#### Pattern 2: Diffuse Attention (Low Max, High Entropy)
```
Statistics: max=0.35, entropy=0.9, sparsity=5%
Interpretation:
  ⚠ Model is uncertain/averaging
  ⚠ Multiple competing signals
  ⚠ Harder to interpret (many pathways active)
```

#### Pattern 3: High Modality Agreement (r > 0.7)
```
Interpretation:
  ✓ EEG and Spectrogram patterns align
  ✓ Both modalities say the same thing
  ✓ Strong evidence for prediction
```

#### Pattern 4: Low Modality Agreement (r < 0.2)
```
Interpretation:
  ⚠ EEG and Spectrogram patterns conflict
  ⚠ Model is reconciling opposing signals
  ⚠ Risky predictions (might be misclassified)
```

### Diagnostic Checklist

**For a Correct Prediction:**
- [ ] Modality agreement > 0.5
- [ ] Both attention directions have low entropy
- [ ] Sparse attention patterns
- [ ] Max attention > 0.7

**For an Incorrect Prediction:**
- [ ] Low modality agreement (< 0.2)
- [ ] High attention entropy
- [ ] Diffuse attention patterns
- [ ] Max attention < 0.5

---

## Advanced Usage

### Custom Attention Analysis

```python
from src.explainability.attention_analysis import AttentionAnalyzer

analyzer = AttentionAnalyzer(num_heads=8)

# Get top attention pairs
eeg_to_spec_weights = attention_dict['eeg_to_spec']
top_pairs = analyzer.get_top_attention_pairs(eeg_to_spec_weights, top_k=10)

# Compute per-head entropy
head_entropies = analyzer.compute_attention_entropy(
    eeg_to_spec_weights,
    per_head=True
)

# Get statistics
stats = analyzer.analyze_attention_weights(eeg_to_spec_weights)
print(f"Mean attention: {stats.mean:.4f}")
print(f"Entropy: {stats.entropy:.4f}")
print(f"Sparsity: {stats.sparsity:.2f}%")
```

### Extract Raw Attention Weights

```python
# Get raw attention tensors
eeg_to_spec = results['attention_dict']['eeg_to_spec']  # (batch, heads, query, key)
spec_to_eeg = results['attention_dict']['spec_to_eeg']

# Convert to numpy for custom analysis
eeg_to_spec_np = eeg_to_spec.cpu().numpy()

# Analyze individual heads
for head_idx in range(8):
    head_attn = eeg_to_spec_np[:, head_idx, :, :]
    # Your custom analysis here...
```

### Batch Explanation Report

```python
# Process all test samples
test_explanations = []

for batch in test_loader:
    with torch.no_grad():
        results = explainer.explain_batch(batch)
    
    # Extract per-sample explanations
    test_explanations.extend(results['per_sample_explanations'])

# Aggregate statistics
import numpy as np

agreements = [e['modality_agreement'] for e in test_explanations]
predictions = [e['predicted_class'] for e in test_explanations]
correct = [e['correct'] for e in test_explanations]

# Class-wise analysis
for class_idx in range(6):
    class_mask = [p == class_idx for p in predictions]
    class_agreements = [a for a, m in zip(agreements, class_mask) if m]
    
    if class_agreements:
        print(f"Class {class_idx}: avg agreement = {np.mean(class_agreements):.4f}")
```

---

## API Reference

### ExplainabilityCapture

```python
from src.explainability import ExplainabilityCapture

explainer = ExplainabilityCapture(model, num_heads=8)
```

**Methods:**

- `explain_batch(batch)` → `Dict[str, Any]`
  - Explains entire batch
  - Returns attention weights and statistics

- `explain_single_sample(batch, sample_idx)` → `Dict[str, Any]`
  - Explains one sample in batch
  - Returns focused explanation for that sample

### AttentionAnalyzer

```python
from src.explainability.attention_analysis import AttentionAnalyzer

analyzer = AttentionAnalyzer(num_heads=8)
```

**Methods:**

- `analyze_attention_weights(weights)` → `AttentionStatistics`
- `get_top_attention_pairs(weights, top_k)` → `List[Tuple]`
- `compute_attention_entropy(weights, per_head)` → `float | Dict`
- `compare_modality_agreement(eeg_to_spec, spec_to_eeg)` → `float`

### Visualization Functions

```python
from src.explainability.visualizations import *

plot_cross_modal_attention(attention_dict, sample_idx=0)
plot_attention_heatmap(attention_weights, title="Title")
plot_head_contributions(attention_dict, num_heads=8)
plot_modality_alignment(attention_dict, sample_idx=0)
```

All return `matplotlib.Figure` objects with optional `save_path` parameter.

---

## Troubleshooting

### Issue: "AttentionStatistics object is not subscriptable"

**Solution**: Access statistics as attributes, not dict keys:
```python
# ✗ Wrong
mean = stats['mean']

# ✓ Correct
mean = stats.mean
```

### Issue: Attention weights are all NaN

**Solution**: Ensure model is in eval mode:
```python
model.eval()  # Don't forget this!

with torch.no_grad():
    results = explainer.explain_batch(batch)
```

### Issue: Out of memory when processing large batches

**Solution**: Reduce batch size or process samples individually:
```python
# Process one sample at a time
single_sample_batch = {
    'eeg_graphs': [g[[0]] for g in batch['eeg_graphs']],
    'spec_graphs': [g[[0]] for g in batch['spec_graphs']],
    'targets': batch['targets'][[0]],
}
```

### Issue: Attention dict keys are incorrect

**Solution**: Ensure fusion module has `return_attention=True`:
```python
# In fusion forward pass
fused, attention_dict = fusion(eeg_features, spec_features, return_attention=True)

# Check keys
assert 'eeg_to_spec' in attention_dict
assert 'spec_to_eeg' in attention_dict
```

---

## Best Practices

1. **Always use eval mode**: `model.eval()` before analysis
2. **Use with torch.no_grad()**: Prevents unnecessary gradient computation
3. **Visualize multiple samples**: Don't rely on single sample
4. **Compare classes**: Look at patterns across different predicted classes
5. **Save visualizations**: Generate PNG/PDF for reports
6. **Document findings**: Keep records of interesting patterns discovered

---

## Citation & References

If you use this explainability framework, please cite:

```bibtex
@article{vaswani2017attention,
  title={Attention is All You Need},
  author={Vaswani, A. and others},
  journal={NeurIPS},
  year={2017}
}
```

For more on interpretable ML:
- Molnar, C. (2022). Interpretable Machine Learning. Online book.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks.

---

## Questions & Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review [Usage Examples](#usage-examples)
3. Examine example output from `src/explainability/example_analysis.py`
4. Review attention tensors directly:
   ```python
   print(results['attention_dict']['eeg_to_spec'].shape)
   print(results['attention_stats']['eeg_to_spec'])
   ```
