# Cross-Modal Attention Explainability - Quick Reference

## One-Page Cheat Sheet

### Import Everything
```python
from src.explainability import (
    ExplainabilityCapture,
    create_explainability_report,
)
from src.explainability.visualizations import (
    plot_cross_modal_attention,
    plot_head_contributions,
    plot_modality_alignment,
)
from src.explainability.attention_analysis import (
    AttentionAnalyzer,
    extract_attention_statistics,
    compute_modality_alignment,
    head_wise_attention_analysis,
)
```

### Basic Workflow

```python
# 1. Setup
model = HMSLightningModule.load_from_checkpoint("model.ckpt")
model.eval()
explainer = ExplainabilityCapture(model, num_heads=8)

# 2. Analyze batch
with torch.no_grad():
    batch = next(iter(test_loader))
    results = explainer.explain_batch(batch)

# 3. Extract results
predictions = results['predictions']           # Predicted classes
logits = results['logits']                     # Prediction confidence
attention_dict = results['attention_dict']     # Raw attention weights
stats = results['attention_stats']             # Statistics
alignment = results['modality_alignment']      # Alignment metrics
per_sample = results['per_sample_explanations'] # Per-sample detail
```

### Metrics Quick Reference

| Metric | Range | Good | Bad |
|--------|-------|------|-----|
| **Modality Agreement** | [-1, 1] | > 0.5 | < 0 |
| **Entropy** | [0, 1] | < 0.5 (focused) | > 0.8 (diffuse) |
| **Sparsity** | [0, 100%] | > 70% | < 20% |
| **Max Attention** | [0, 1] | > 0.7 | < 0.3 |

### Visualization Gallery

#### 1. Cross-Modal Heatmaps
```python
fig = plot_cross_modal_attention(results['attention_dict'], sample_idx=0)
```
Shows: EEG↔Spec attention connections

#### 2. Per-Head Strengths
```python
fig = plot_head_contributions(results['attention_dict'])
```
Shows: Which heads are active

#### 3. Modality Alignment
```python
fig = plot_modality_alignment(results['attention_dict'], sample_idx=0)
```
Shows: Forward/backward attention agreement

### Per-Sample Analysis

```python
# Get single sample explanation
sample_expl = results['per_sample_explanations'][0]

print(f"Predicted: {sample_expl['predicted_class']}")
print(f"Confidence: {sample_expl['predicted_logit']:.4f}")
print(f"Ground Truth: {sample_expl['true_class']}")
print(f"Correct: {sample_expl['correct']}")
print(f"Modality Agreement: {sample_expl['modality_agreement']:.4f}")
print(f"Top EEG→Spec: {sample_expl['top_attention_pairs']['eeg_to_spec'][0]}")
print(f"Top Spec→EEG: {sample_expl['top_attention_pairs']['spec_to_eeg'][0]}")
```

### Generate Report

```python
report = create_explainability_report(results, output_path="report.txt")
print(report)
```

Output:
```
BATCH-LEVEL STATISTICS
- EEG → Spectrogram Attention: mean=0.52, std=0.23, entropy=0.65
- Spectrogram → EEG Attention: mean=0.49, std=0.21, entropy=0.62
- Modality Agreement: 0.6543

PER-SAMPLE ANALYSIS
Sample 0:
  Predicted: 2 (LPD, confidence: 0.8234)
  Ground Truth: 2 ✓ CORRECT
  Modality Agreement: 0.7234
  ...
```

### Advanced: Custom Analysis

```python
# Get analyzer
analyzer = AttentionAnalyzer(num_heads=8)

# Top attention pairs
eeg_to_spec = results['attention_dict']['eeg_to_spec']
top_pairs = analyzer.get_top_attention_pairs(eeg_to_spec, top_k=5)
# Returns: [(query_idx, key_idx, weight), ...]

# Per-head specialization
head_stats = head_wise_attention_analysis(results['attention_dict'])
for head_idx, stats in head_stats['eeg_to_spec'].items():
    print(f"Head {head_idx}: {stats['statistics']}")

# Modality agreement for this sample
agreement = analyzer.compare_modality_agreement(
    results['attention_dict']['eeg_to_spec'],
    results['attention_dict']['spec_to_eeg']
)
```

### Batch Processing All Test Data

```python
all_results = []
for batch in test_loader:
    with torch.no_grad():
        results = explainer.explain_batch(batch)
    all_results.append(results)

# Aggregate statistics
import numpy as np

all_agreements = []
all_entropies = []
all_correct = []

for results in all_results:
    all_agreements.extend([
        e['modality_agreement'] 
        for e in results['per_sample_explanations']
    ])
    all_correct.extend([
        e['correct'] 
        for e in results['per_sample_explanations']
    ])

print(f"Average modality agreement: {np.mean(all_agreements):.4f}")
print(f"Accuracy: {np.mean(all_correct):.4f}")
```

### Compare Correct vs Incorrect

```python
correct_agreements = []
incorrect_agreements = []

for results in all_results:
    for expl in results['per_sample_explanations']:
        if expl['correct']:
            correct_agreements.append(expl['modality_agreement'])
        else:
            incorrect_agreements.append(expl['modality_agreement'])

print(f"Correct avg agreement: {np.mean(correct_agreements):.4f}")
print(f"Incorrect avg agreement: {np.mean(incorrect_agreements):.4f}")
# Hypothesis: Incorrect predictions have lower agreement
```

### Class-Wise Analysis

```python
classes = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

for class_idx in range(6):
    class_samples = [
        e for results in all_results 
        for e in results['per_sample_explanations']
        if e['predicted_class'] == class_idx
    ]
    
    if class_samples:
        avg_agree = np.mean([s['modality_agreement'] for s in class_samples])
        avg_entropy = np.mean([
            np.mean(list(s['head_entropies'].values()))
            for s in class_samples
        ])
        
        print(f"{classes[class_idx]:10s}: agreement={avg_agree:.4f}, entropy={avg_entropy:.4f}")
```

### Save Everything

```python
import pickle
import json

# Save raw results
with open("results.pkl", "wb") as f:
    pickle.dump(all_results, f)

# Save report
with open("report.txt", "w") as f:
    report = create_explainability_report(results)
    f.write(report)

# Save visualizations
for i, batch_results in enumerate(all_results):
    fig = plot_cross_modal_attention(
        batch_results['attention_dict'],
        sample_idx=0
    )
    fig.savefig(f"batch_{i}_attention.png", dpi=150, bbox_inches='tight')

# Summary statistics (JSON)
summary = {
    'num_samples': len(all_explanations),
    'avg_modality_agreement': float(np.mean(all_agreements)),
    'avg_entropy_eeg_to_spec': float(np.mean([
        s['attention_stats']['eeg_to_spec'].entropy
        for r in all_results
        for s in r['per_sample_explanations']
    ])),
}
with open("summary.json", "w") as f:
    json.dump(summary, f, indent=2)
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `model.eval()` forgotten | Add `model.eval()` before analysis |
| Outside `no_grad()` | Wrap in `with torch.no_grad():` |
| Wrong device | Ensure batch is on same device as model |
| NaN in attention | Check for invalid inputs |
| Out of memory | Process smaller batches or individual samples |
| Stats not found | Ensure `explain_batch()` was called first |

### Key Patterns to Look For

**Pattern ✓ Focused Attention**
- High max (> 0.7)
- Low entropy (< 0.4)
- High sparsity (> 80%)
- → Confident, interpretable decisions

**Pattern ⚠ Diffuse Attention**
- Low max (< 0.4)
- High entropy (> 0.8)
- Low sparsity (< 30%)
- → Uncertain, many competing signals

**Pattern ✓ Modality Agreement**
- Agreement score > 0.6
- → EEG and Spec patterns align
- → Strong evidence

**Pattern ⚠ Modality Conflict**
- Agreement score < 0.2
- → EEG and Spec disagree
- → Risky predictions

### How to Interpret Results

```
Modality Agreement = 0.75  ✓ Strong consensus
Head Entropy = 0.3         ✓ Focused attention
Max Attention = 0.92       ✓ Clear pathways
Sparsity = 85%             ✓ Interpretable

Prediction: CORRECT

---

Modality Agreement = 0.05  ⚠ Weak consensus
Head Entropy = 0.9         ⚠ Diffuse attention
Max Attention = 0.38       ⚠ No clear pathways
Sparsity = 12%             ⚠ Noisy

Prediction: INCORRECT
```

### Complete Example Script

```python
# Complete end-to-end example
import torch
from src.lightning_trainer import HMSLightningModule
from src.explainability import ExplainabilityCapture, create_explainability_report
from src.explainability.visualizations import plot_cross_modal_attention
import numpy as np

# Setup
model = HMSLightningModule.load_from_checkpoint("checkpoints/best.ckpt")
model.eval()
explainer = ExplainabilityCapture(model)

# Process test set
all_results = []
for batch in test_loader:
    with torch.no_grad():
        results = explainer.explain_batch(batch)
    all_results.append(results)

# Analysis
all_agreements = []
all_predictions = []
all_targets = []

for results in all_results:
    all_predictions.extend(results['predictions'].tolist())
    all_targets.extend(results['targets'].tolist())
    
    for expl in results['per_sample_explanations']:
        all_agreements.append(expl['modality_agreement'])

# Report
accuracy = np.mean([p == t for p, t in zip(all_predictions, all_targets)])
avg_agreement = np.mean(all_agreements)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Avg Modality Agreement: {avg_agreement:.4f}")

# Visualize first batch
fig = plot_cross_modal_attention(all_results[0]['attention_dict'], sample_idx=0)
fig.savefig("first_sample_attention.png")

# Generate report
report = create_explainability_report(all_results[0])
print(report)
```

---

## Where to Find More Info

- **Full Guide**: `CROSS_MODAL_ATTENTION_EXPLAINABILITY.md`
- **Implementation Details**: `EXPLAINABILITY_IMPLEMENTATION_SUMMARY.md`
- **Example Script**: `src/explainability/example_analysis.py`
- **Source Code**: `src/explainability/`
