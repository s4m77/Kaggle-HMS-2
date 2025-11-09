# ZORRO Explainer for HMS Multi-Modal GNN

## Overview

ZORRO (Zero-Order Rank-based Relative Output) is a post-hoc explainability method designed for Graph Neural Networks (GNNs). This implementation provides a tool to understand which nodes and node features are most responsible for the HMS multi-modal GNN's predictions.

## What is ZORRO?

ZORRO identifies important nodes and node features through **perturbation analysis**:

1. **Baseline Prediction**: Get the model's original prediction
2. **Perturbation**: Perturb individual node features (zero-out, add noise, or set to mean)
3. **Measure Change**: Compute how much the prediction changes
4. **Importance Ranking**: Nodes/features with larger prediction changes are more important

### Key Properties

- **Post-hoc**: Works with any trained model without modification
- **Model-agnostic**: Doesn't require knowledge of model internals
- **Node-level explanations**: Identifies critical graph nodes
- **Feature-level explanations**: Identifies critical node attributes
- **Scalable**: Handles large graphs through Monte Carlo sampling

## Files

### Core Implementation

- **`src/models/zorro_explainer.py`**: Main ZORRO explainer implementation
  - `ZORROExplainer`: Main explainer class
  - `ZORROExplanation`: Data class for storing explanation results

- **`examples/zorro_explainer_example.py`**: Utility functions and examples
  - `explain_hms_predictions()`: High-level API for explaining batches
  - `print_explanation()`: Pretty-print explanations
  - `compare_modalities()`: Compare EEG and spectrogram modalities

- **`notebooks/zorro_explainer_tutorial.ipynb`**: Complete tutorial notebook

## Usage

### Basic Usage

```python
from src.models import HMSMultiModalGNN, ZORROExplainer
import torch

# Load your trained model
model = HMSMultiModalGNN(...)
model.load_state_dict(torch.load("checkpoint.pt"))
model.to(device)

# Create explainer
explainer = ZORROExplainer(
    model=model,
    device=device,
    perturbation_mode="zero",  # or "noise", "mean"
)

# Explain a sample
explanation = explainer.explain_sample(
    graphs=eeg_graphs,
    modality="eeg",
    sample_idx=0,
    top_k=10,
    n_samples=5,  # Number of perturbation samples
)

# Access results
print(f"Top important nodes: {explanation.top_k_nodes}")
print(f"Feature importance: {explanation.feature_importance}")
```

### Advanced Usage - Batch Explanation

```python
# Explain entire batch for both modalities
explanations = explainer.explain_batch(
    eeg_graphs=eeg_graphs,
    spec_graphs=spec_graphs,
    sample_indices=[0, 1, 2],
    top_k=15,
    return_eeg=True,
    return_spec=True,
    n_samples=5,
)

# Results for sample 0
eeg_exp = explanations[0]["eeg"]
spec_exp = explanations[0]["spec"]

# Compare modalities
print(f"EEG total importance: {eeg_exp.node_importance.sum()}")
print(f"Spec total importance: {spec_exp.node_importance.sum()}")
```

### Using High-Level API

```python
from examples.zorro_explainer_example import explain_hms_predictions, print_explanation

# Get explanations for multiple samples
all_explanations = explain_hms_predictions(
    model=model,
    eeg_graphs=eeg_graphs,
    spec_graphs=spec_graphs,
    sample_indices=[0, 1],
    top_k=10,
)

# Pretty print results
for sample_idx, modality_exps in all_explanations.items():
    print_explanation(modality_exps["eeg"], "EEG")
    print_explanation(modality_exps["spec"], "Spectrogram")
```

## Output Format

The `ZORROExplanation` dataclass contains:

- **`node_importance`** (Tensor): Shape (num_nodes, num_features)
  - Importance scores for each node-feature pair
  - Computed via perturbation sensitivity

- **`node_indices`** (List[int]): Global indices of nodes in graph

- **`top_k_nodes`** (List[Tuple[int, float]]): Top-k nodes by aggregate importance
  - Format: [(node_idx, importance_score), ...]
  - Sorted by importance descending

- **`feature_importance`** (Tensor): Shape (num_features,)
  - Aggregated importance per feature
  - Computed by averaging across nodes

- **`prediction_original`** (Tensor): Original model logits

- **`modality`** (str): "eeg" or "spec"

## Perturbation Modes

### 1. Zero-Out (Default)
```python
perturbation_mode="zero"
```
- Sets node feature to 0
- Simplest, shows impact of feature removal
- Recommended for initial analysis

### 2. Gaussian Noise
```python
perturbation_mode="noise"
noise_std=0.1  # Adjust based on feature scale
```
- Adds Gaussian noise: feature → feature + N(0, std²)
- Smoother perturbation, less artificial
- Good for gradual feature degradation

### 3. Mean Value
```python
perturbation_mode="mean"
```
- Replaces feature with batch mean
- Neutral perturbation, preserves statistics
- Useful for feature importance disambiguation

## Parameters

### ZORROExplainer Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | - | Trained GNN model |
| `target_class` | Optional[int] | None | Class to explain (None = predicted class) |
| `device` | Optional[torch.device] | None | Computation device |
| `perturbation_mode` | str | "zero" | How to perturb: "zero", "noise", "mean" |
| `noise_std` | float | 0.1 | Std for Gaussian noise |

### explain_sample() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graphs` | List[Batch] | - | Temporal graphs for one modality |
| `modality` | str | - | "eeg" or "spec" |
| `sample_idx` | int | - | Sample index in batch |
| `top_k` | int | 10 | Number of top nodes to return |
| `n_samples` | int | 5 | Monte Carlo perturbation samples |
| `pbar` | bool | True | Show progress bar |

## Architecture Details

### For Multi-Modal Model

The explainer handles the HMS multi-modal architecture:

1. **EEG Branch**: 9 temporal graphs × 9 nodes/graph = 81 total nodes
2. **Spec Branch**: 119 temporal graphs × 9 nodes/graph = 1071 total nodes
3. **Cross-Modal Fusion**: Combines EEG and spectrogram representations
4. **Classification**: 6-class brain activity prediction

When explaining one modality, the explainer:
- Perturbs that modality's graphs
- Uses original features for other modality (to isolate effects)
- Computes importance through prediction changes

### Node Indexing

Nodes are indexed sequentially:
- **Temporal dimension**: Graph 0 nodes 0-8, Graph 1 nodes 9-17, etc.
- **Sample dimension**: Batch samples are processed independently
- **Global indices**: Used consistently within one `explain_sample()` call

## Interpretation Guide

### Node Importance

High node importance means:
- That electrode channel (EEG) or frequency bin (spectrogram) region is critical
- Removing that node significantly affects prediction
- Model attends to that location in brain

### Feature Importance

High feature importance means:
- That feature type (band power type) is critical across the graph
- Example: Delta band power more important than Alpha

### Modality Comparison

- If EEG importance >> Spectrogram: Model relies more on temporal dynamics
- If Spectrogram importance >> EEG: Model relies more on frequency characteristics
- Balanced: Multi-modal decision-making

## Tutorial & Examples

See **`notebooks/zorro_explainer_tutorial.ipynb`** for:

1. Loading model and data
2. Computing node importance
3. Extracting feature importance
4. Visualizing explanations:
   - Feature importance bar charts
   - Top-k node rankings
   - Node-feature heatmaps
5. Comparing modalities
6. Quality evaluation (fidelity, sparsity)
7. Practical interpretation examples

## Performance Considerations

### Computational Cost

- **Time Complexity**: O(num_nodes × num_features × n_samples)
- For full graphs: ~81 nodes × 5 features × 5 samples = ~2000 forward passes/sample
- **Typical time**: 2-10 seconds per sample on GPU

### Optimization Tips

1. **Reduce n_samples**: Trade accuracy for speed (3-5 usually sufficient)
2. **Use GPU**: ~10x faster than CPU
3. **Batch processing**: Process multiple samples simultaneously
4. **Sparse graphs**: Fewer nodes = faster computation
5. **Caching**: Reuse explanations if input unchanged

### Memory Usage

- Minimal: Stores node_importance tensor (num_nodes × num_features floats)
- ~10KB per explanation (for 80 nodes × 5 features)

## Limitations & Considerations

1. **Perturbation Dependency**:
   - Results depend on perturbation strategy
   - Try multiple modes for robustness

2. **Independence Assumption**:
   - Treats features as independent
   - Doesn't capture feature interactions

3. **Local Explanations**:
   - Explains one prediction at a time
   - Not global model behavior

4. **Baseline Dependence**:
   - Importance is relative to original prediction
   - Different for different samples

## Extending ZORRO

### Custom Perturbation

```python
class CustomExplainer(ZORROExplainer):
    def _perturb_node_feature(self, ...):
        # Implement custom perturbation logic
        pass
```

### Aggregate Explanations

```python
def aggregate_explanations(explanations_list):
    # Average importance across samples
    node_imp = torch.stack([e.node_importance for e in explanations_list]).mean(0)
    # ... create new ZORROExplanation
```

### Cross-Modality Analysis

```python
def compare_modalities_statistical(eeg_exp, spec_exp):
    # Hypothesis testing on importance differences
    # Spearman correlation of important nodes
    # Statistical significance tests
```

## References

- ZORRO: Zero-Order Rank-based Relative Output
- Paper: https://arxiv.org/abs/2305.02783
- GNN Explainability: https://arxiv.org/abs/2104.13375

## Citation

If you use this ZORRO implementation, please cite:

```bibtex
@inproceedings{zorro2023,
  title={ZORRO: Zero-Order Rank-based Relative Output},
  author={...},
  year={2023}
}
```

## Troubleshooting

### Issue: Low importance scores

- Check if model predictions are stable (not random)
- Verify n_samples is sufficient
- Ensure perturbation mode appropriate for features

### Issue: Uniform importance

- May indicate model doesn't strongly attend to specific nodes
- Try different target classes
- Verify model is actually trained (not random)

### Issue: Out of memory

- Reduce n_samples
- Reduce batch size
- Use CPU (slower but less memory)

### Issue: Inconsistent results

- Increase n_samples for Monte Carlo stability
- Check random seed
- Verify data preprocessing consistency
