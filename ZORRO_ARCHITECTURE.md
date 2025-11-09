# ZORRO Explainer Architecture & Workflow

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ZORRO EXPLAINER SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT: Trained Model + Graph Data                                     │
│  ─────────────────────────────────────────────────────────────────────  │
│     HMSMultiModalGNN                                                   │
│     ├── EEG Branch (9 graphs × 9 nodes)                               │
│     └── Spec Branch (119 graphs × 9 nodes)                           │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │            ZORRO EXPLAINER (src/models/zorro_explainer.py)       │  │
│  │                                                                  │  │
│  │  explain_batch()                                               │  │
│  │  ├── explain_sample() × N samples                             │  │
│  │  └── For each sample:                                         │  │
│  │      ├── Get original prediction                              │  │
│  │      ├── For each node n and feature f:                       │  │
│  │      │   ├── Perturb node_n.feature_f                         │  │
│  │      │   ├── Get perturbed prediction                         │  │
│  │      │   └── Compute delta = |original - perturbed|           │  │
│  │      └── Rank nodes by importance                             │  │
│  │                                                                  │  │
│  │  OUTPUT: ZORROExplanation(s)                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │         UTILITIES & ANALYSIS (examples/zorro_explainer_example.py)  │  │
│  │  ├── explain_hms_predictions()  - High-level batch API         │  │
│  │  ├── print_explanation()         - Pretty printing              │  │
│  │  └── compare_modalities()        - Modality comparison          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │        PRODUCTION WORKFLOW (scripts/zorro_workflow.py)           │  │
│  │  ├── ZORROWorkflow class                                       │  │
│  │  ├── Model loading                                             │  │
│  │  ├── Batch explanation                                         │  │
│  │  ├── JSON export                                               │  │
│  │  └── Report generation                                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  OUTPUT: Explanations                                                  │
│  ──────────────────────────────────────────────────────────────────   │
│     ZORROExplanation object(s):                                       │
│     ├── node_importance      (Tensor: num_nodes × num_features)      │
│     ├── top_k_nodes          (List of ranked important nodes)        │
│     ├── feature_importance   (Tensor: num_features)                  │
│     ├── prediction_original  (Model logits)                          │
│     └── modality             ("eeg" or "spec")                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Explanation Generation Pipeline

```
Step 1: Sample Input
────────────────────────────────────
  eeg_graphs: List[Batch]  (9 temporal graphs)
  spec_graphs: List[Batch] (119 temporal graphs)
  sample_idx: 0
  
          ⬇
          
Step 2: Initialize Explainer
────────────────────────────────────
  explainer = ZORROExplainer(model)
  
          ⬇
          
Step 3: Get Baseline Prediction
────────────────────────────────────
  output_baseline = model(eeg_graphs, spec_graphs)
  pred_baseline = output_baseline[sample_idx]
  
  Example: [0.1, 0.2, 3.5, 0.4, 0.5, 0.2] → class 2 (seizure)
  
          ⬇
          
Step 4: Compute Node Importance (for each node & feature)
────────────────────────────────────
  For node_n, feature_f:
    
    importance_scores = []
    For i in 1..n_samples:
      1. Perturb: node_n.feature_f → 0 (or noise/mean)
      2. Forward pass: pred_perturbed = model(graphs_perturbed)
      3. Compute delta: δᵢ = |pred_baseline[class] - pred_perturbed[class]|
      4. Store: importance_scores.append(δᵢ)
    
    importance(node_n, feature_f) = mean(importance_scores)
  
  Result: node_importance tensor (num_nodes × num_features)
  
          ⬇
          
Step 5: Aggregate & Rank
────────────────────────────────────
  feature_importance = node_importance.sum(dim=0) / num_nodes
  top_k_nodes = sorted by node_importance.sum(dim=1)
  
          ⬇
          
Step 6: Return ZORROExplanation
────────────────────────────────────
  {
    node_importance: Tensor(81, 5),     # For EEG: 9 graphs × 9 nodes
    top_k_nodes: [(5, 0.234), (12, 0.198), ...],
    feature_importance: Tensor(5),
    prediction_original: Tensor(6),
    modality: "eeg"
  }
```

## Data Flow for Multi-Modal Model

```
BATCH INPUT
├─ eeg_graphs (9 temporal steps)
│  └─ Graph_t=0 (81 nodes from batch)
│     ├─ Sample 0: nodes 0-8
│     ├─ Sample 1: nodes 9-17
│     └─ ...
│
└─ spec_graphs (119 temporal steps)
   └─ Graph_t=0 (1071 nodes from batch)
      ├─ Sample 0: nodes 0-8
      ├─ Sample 1: nodes 9-17
      └─ ...

EXPLANATION PROCESS (sample_idx=0)
├─ Extract sample 0 from all graphs
│  ├─ EEG: 9 graphs × 9 nodes = 81 nodes
│  └─ Spec: 119 graphs × 9 nodes = 1071 nodes
│
├─ For EEG explanation:
│  ├─ Perturb EEG nodes
│  ├─ Use original Spec graphs
│  └─ Get EEG importance
│
└─ For Spec explanation:
   ├─ Use original EEG graphs
   ├─ Perturb Spec nodes
   └─ Get Spec importance

OUTPUT
├─ ZORROExplanation (EEG)
│  └─ node_importance: (81, 5)
│     ├─ Row 0-8: Graph 0 nodes
│     ├─ Row 9-17: Graph 1 nodes
│     └─ Row 72-80: Graph 8 nodes
│
└─ ZORROExplanation (Spec)
   └─ node_importance: (1071, 5)
      ├─ Row 0-8: Graph 0 nodes
      ├─ Row 9-17: Graph 1 nodes
      └─ Row 1062-1070: Graph 118 nodes
```

## Perturbation Modes

```
ORIGINAL FEATURE
─────────────────────────────
  x = [0.5, -0.2, 1.2]
  
        ⬇
        ⬇ Perturbation Mode

┌─────────────────────────────────────────────────────────────┐
│ MODE 1: ZERO-OUT (Default)                                 │
├─────────────────────────────────────────────────────────────┤
│  x_perturbed = [0.5, 0.0, 1.2]  (feature 1 set to 0)      │
│  ✓ Clear interpretation                                     │
│  ✗ Slightly artificial                                      │
│  Use case: Initial analysis, debugging                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MODE 2: GAUSSIAN NOISE (noise_std=0.1)                     │
├─────────────────────────────────────────────────────────────┤
│  noise ~ N(0, 0.1²)                                         │
│  x_perturbed = [0.5, -0.2 + noise, 1.2]                    │
│  ✓ Realistic degradation                                    │
│  ✓ Smoother perturbation                                    │
│  Use case: Robustness analysis                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MODE 3: MEAN REPLACEMENT                                   │
├─────────────────────────────────────────────────────────────┤
│  mean_val = batch_mean(feature 1) = 0.1                    │
│  x_perturbed = [0.5, 0.1, 1.2]                             │
│  ✓ Statistical baseline                                     │
│  ✓ Feature distribution preserved                          │
│  Use case: Feature attribution                             │
└─────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
src/models/
├── __init__.py
│   └── exports: ZORROExplainer, ZORROExplanation
│
├── zorro_explainer.py ← Main implementation
│   ├── imports: torch, torch.nn, torch_geometric, numpy, dataclasses
│   ├── ZORROExplainer class
│   └── ZORROExplanation dataclass
│
└── hms_model.py (uses ZORROExplainer)
    └── HMSMultiModalGNN (target model)

examples/
└── zorro_explainer_example.py
    ├── imports: ZORROExplainer, ZORROExplanation from src.models
    ├── explain_hms_predictions()
    ├── print_explanation()
    └── compare_modalities()

scripts/
└── zorro_workflow.py
    ├── imports: ZORROExplainer from src.models
    ├── ZORROWorkflow class
    └── main() entry point

notebooks/
└── zorro_explainer_tutorial.ipynb
    ├── imports all modules
    ├── Interactive examples
    └── Visualizations
```

## Typical Workflow

```
USER CODE
───────────

1. Load Data
   eeg_graphs, spec_graphs = load_data()

2. Load Model
   model = HMSMultiModalGNN()
   model.load_state_dict(torch.load("best.pt"))

3. Create Explainer
   explainer = ZORROExplainer(model=model, device="cuda")

4. Explain Batch
   for sample_idx in range(batch_size):
       eeg_exp = explainer.explain_sample(
           graphs=eeg_graphs,
           modality="eeg",
           sample_idx=sample_idx,
           n_samples=5
       )
       spec_exp = explainer.explain_sample(
           graphs=spec_graphs,
           modality="spec",
           sample_idx=sample_idx,
           n_samples=5
       )

5. Analyze Results
   print(eeg_exp.top_k_nodes)
   print(eeg_exp.feature_importance)

6. Visualize
   plot_importance(eeg_exp)
   compare_modalities(eeg_exp, spec_exp)

7. Export
   save_to_json(eeg_exp)
```

## Computational Complexity

```
For one sample explanation:

Time Complexity:
─────────────────
  O(num_nodes × num_features × n_samples × T_forward)
  
  = 81 nodes × 5 features × 5 samples × 0.05s
  ≈ 101 seconds (CPU)
  ≈ 10 seconds (GPU)

Space Complexity:
──────────────────
  O(num_nodes × num_features)
  = 81 × 5 × 4 bytes (float32)
  ≈ 2 KB per sample

Memory Usage:
─────────────
  Model: ~100 MB
  Graph data: ~50 MB (batch)
  Explanations: ~10 KB per sample
  Total: ~150 MB
```

## Quality Metrics

```
FIDELITY
─────────
Measures: How much model relies on identified nodes
Formula:  conf_with_topk / conf_original
Range:    0 to 1
Target:   > 0.7 (70% confidence retained with top-20% nodes)

SPARSITY
─────────
Measures: Percentage of nodes needed for 80% importance
Formula:  num_topk_nodes / total_nodes
Range:    0 to 1
Target:   < 0.2 (< 20% nodes needed)

Interpretation:
  Low sparsity (< 0.1):  Model very focused on few nodes
  Medium sparsity (0.1-0.3): Distributed attention
  High sparsity (> 0.3):  Uniform importance across nodes
```

## Performance Characteristics

```
SPEED
──────
GPU:  2-10 seconds per sample
CPU:  20-100 seconds per sample
Batch: Linear scaling (N samples = N × time_per_sample)

Optimization Opportunities:
  ✓ Reduce n_samples (5 often sufficient)
  ✓ Use GPU (10x faster)
  ✓ Cache perturbations
  ✓ Parallel processing across samples

MEMORY
──────
Peak memory: ~150 MB
Per sample: ~10 KB explanation
Scales linearly with num_nodes

STABILITY
─────────
Monte Carlo variance decreases with n_samples:
  n_samples=3:  High variance
  n_samples=5:  Good balance
  n_samples=10: Very stable
  n_samples=20: Over-sampling
```

## Integration Points

```
Your HMS Project
├── Existing: HMSMultiModalGNN
│   ├── Model checkpoint
│   ├── Training pipeline
│   └── Data loaders
│
├── New: ZORRO Explainer
│   ├── Zero-cost integration
│   ├── No model modification
│   ├── Post-hoc analysis
│   └── Interpretability layer
│
└── Outputs:
    ├── Explanation objects
    ├── Visualizations
    ├── JSON reports
    └── Insights for downstream use
```

---

This architecture ensures:
- ✅ Modular design
- ✅ Easy integration
- ✅ Scalable processing
- ✅ Clear data flow
- ✅ Production-ready
