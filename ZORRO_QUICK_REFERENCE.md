# ZORRO Explainer - Quick Reference

## Installation & Setup

No additional dependencies needed! Uses existing: PyTorch, PyTorch Geometric

## Import

```python
from src.models import HMSMultiModalGNN, ZORROExplainer
from examples.zorro_explainer_example import explain_hms_predictions
```

## Minimal Working Example

```python
import torch
from src.models import HMSMultiModalGNN, ZORROExplainer

# 1. Load model
model = HMSMultiModalGNN()
model.load_state_dict(torch.load("best_model.pt"))
model.to("cuda")

# 2. Create explainer
explainer = ZORROExplainer(model, device="cuda")

# 3. Explain one sample
explanation = explainer.explain_sample(
    graphs=eeg_graphs,
    modality="eeg",
    sample_idx=0,
    top_k=10,
    n_samples=5
)

# 4. View results
print("Top 5 nodes:", explanation.top_k_nodes[:5])
print("Feature importance:", explanation.feature_importance)
```

## Common Tasks

### Explain Entire Batch

```python
results = explainer.explain_batch(
    eeg_graphs=eeg_graphs,
    spec_graphs=spec_graphs,
    sample_indices=[0, 1, 2],
    top_k=10,
    return_eeg=True,
    return_spec=True,
)
```

### Get Top-k Important Nodes

```python
explanation = explainer.explain_sample(...)
top_nodes = explanation.top_k_nodes[:10]  # List of (node_idx, importance)
```

### Get Feature Importance

```python
feature_scores = explanation.feature_importance  # Tensor of shape (num_features,)
```

### Compare Modalities

```python
from examples.zorro_explainer_example import compare_modalities

eeg_exp = results[0]["eeg"]
spec_exp = results[0]["spec"]
compare_modalities(eeg_exp, spec_exp)
```

### Save Results

```python
from scripts.zorro_workflow import ZORROWorkflow

workflow = ZORROWorkflow("best_model.pt", output_dir="./results")
explanations = workflow.explain_batch(eeg_graphs, spec_graphs)
workflow.save_batch_explanations(explanations)
```

## Parameters

### ZORROExplainer.__init__
```python
explainer = ZORROExplainer(
    model=model,                    # Trained model
    target_class=None,             # None = use predicted class
    device="cuda",                 # Device for computation
    perturbation_mode="zero",      # "zero", "noise", or "mean"
    noise_std=0.1,                # For noise mode
)
```

### explain_sample()
```python
explanation = explainer.explain_sample(
    graphs=eeg_graphs,             # List of 9 Batch objects
    modality="eeg",                # "eeg" or "spec"
    sample_idx=0,                  # Which sample in batch
    top_k=10,                      # Return top-k nodes
    n_samples=5,                   # Monte Carlo samples (3-10)
    pbar=True,                     # Show progress bar
)
```

### explain_batch()
```python
explanations = explainer.explain_batch(
    eeg_graphs=eeg_graphs,         # List of 9 Batch objects
    spec_graphs=spec_graphs,       # List of 119 Batch objects
    sample_indices=[0, 1],         # Samples to explain
    top_k=10,                      # Top-k nodes
    return_eeg=True,               # Explain EEG?
    return_spec=True,              # Explain spectrogram?
    n_samples=5,                   # Monte Carlo samples
    pbar=True,                     # Progress bar
)
```

## Output Format

### ZORROExplanation Object

```python
explanation.node_importance        # Tensor(num_nodes, num_features) - importance scores
explanation.node_indices           # List[int] - node indices
explanation.top_k_nodes            # List[(node_idx, importance)] - sorted top-k
explanation.feature_importance     # Tensor(num_features) - aggregated per feature
explanation.prediction_original    # Tensor(num_classes) - original logits
explanation.modality               # str - "eeg" or "spec"
```

## Interpretation Guide

### What high node importance means
- That electrode/frequency region critical for prediction
- Removing that node significantly changes output
- Model attends strongly to that location

### What high feature importance means
- That feature type (e.g., Delta band) is broadly important
- All or most nodes rely on this feature

### Comparing modalities
- **EEG >> Spectrogram**: Model focuses on temporal dynamics
- **Spectrogram >> EEG**: Model focuses on frequency content
- **Balanced**: Multi-modal decision-making

## Visualization Examples

```python
import matplotlib.pyplot as plt

# Feature importance bar chart
fig, ax = plt.subplots()
ax.bar(range(len(explanation.feature_importance)), 
       explanation.feature_importance.cpu().numpy())
plt.show()

# Top-k nodes bar chart
top_10 = explanation.top_k_nodes[:10]
nodes, importance = zip(*top_10)
ax.barh(range(len(nodes)), importance)
ax.set_yticks(range(len(nodes)))
ax.set_yticklabels([f"Node {n}" for n in nodes])
plt.show()

# Heatmap of node-feature importance
import seaborn as sns
sns.heatmap(explanation.node_importance.cpu().numpy(), 
            cmap='Blues', xticklabels=True, yticklabels=True)
plt.show()
```

## Performance Tips

| Optimization | Speedup | Notes |
|---|---|---|
| Use GPU | 10x | Huge difference |
| Reduce n_samples | 2-3x | Trade stability for speed |
| Reduce top_k | - | Doesn't affect speed |
| Batch samples | - | Can parallelize |

## Troubleshooting

### "Low importance scores"
→ Check if model predictions are stable (not random)

### "Uniform importance everywhere"
→ Model may not attend to specific nodes; verify training

### "Out of memory"
→ Reduce n_samples or batch_size

### "Inconsistent results"
→ Increase n_samples for Monte Carlo stability

## Files

- **Tutorial**: `notebooks/zorro_explainer_tutorial.ipynb`
- **Full docs**: `src/models/ZORRO_README.md`
- **Implementation**: `src/models/zorro_explainer.py`
- **Utils**: `examples/zorro_explainer_example.py`
- **Workflow**: `scripts/zorro_workflow.py`

## Full Example

```python
import torch
from src.models import HMSMultiModalGNN, ZORROExplainer
from examples.zorro_explainer_example import print_explanation, compare_modalities

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HMSMultiModalGNN().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))

# Create explainer
explainer = ZORROExplainer(model=model, device=device)

# Explain samples
for sample_idx in [0, 1, 2]:
    eeg_exp = explainer.explain_sample(
        graphs=eeg_graphs,
        modality="eeg",
        sample_idx=sample_idx,
        top_k=10,
        n_samples=5,
    )
    
    spec_exp = explainer.explain_sample(
        graphs=spec_graphs,
        modality="spec",
        sample_idx=sample_idx,
        top_k=10,
        n_samples=5,
    )
    
    # Print and compare
    print_explanation(eeg_exp, "EEG")
    print_explanation(spec_exp, "Spectrogram")
    compare_modalities(eeg_exp, spec_exp)
```

## Need Help?

1. **Quick start**: See minimal example above
2. **Interactive tutorial**: Run `notebooks/zorro_explainer_tutorial.ipynb`
3. **Detailed guide**: Read `src/models/ZORRO_README.md`
4. **Production use**: Check `scripts/zorro_workflow.py`
