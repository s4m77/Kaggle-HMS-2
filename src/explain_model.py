from src.models.explainer_wrappers import EEGExplainerWrapper, SpecExplainerWrapper
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Batch
from src.lightning_trainer.graph_lightning_module import HMSLightningModule
from omegaconf import OmegaConf
import torch
import argparse
import sys
import pandas as pd
import os

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explainer for EEG Graph")
    p.add_argument("--config", default="configs/train.yaml", help="Training config path")
    p.add_argument("--checkpoint", required=True, help="Path to Lightning checkpoint")
    p.add_argument("--graphs", default="data/processed/patient_graphs.pt", help="Graphs file path")
    p.add_argument("--train_csv", default="data/raw/train.csv", help="Path to train.csv file")
    p.add_argument("--target_class", required=True, type=str, 
                   choices=['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other'],
                   help="Target class to explain")
    p.add_argument("--output_dir", default="explanations", help="Directory to save explanation results")
    return p.parse_args()


def get_prediction(model, eeg_graph, spec_graph):
    """Get model prediction for a single graph pair."""
    model.eval()
    with torch.no_grad():
        eeg_batch = Batch.from_data_list([eeg_graph])
        spec_batch = Batch.from_data_list([spec_graph])
        
        output = model.model(eeg_batch, spec_batch)
        
        # Handle different output types
        if isinstance(output, dict):
            # If output is a dict, look for 'logits' or 'out' key
            if 'logits' in output:
                logits = output['logits']
            elif 'out' in output:
                logits = output['out']
            else:
                # Print keys to help debug
                print(f"Model output keys: {output.keys()}")
                raise ValueError(f"Could not find logits in model output. Available keys: {output.keys()}")
        else:
            # Output is already a tensor
            logits = output
        
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
    return predicted_class, probabilities[0]


def find_examples(model, patient_graphs, filtered_df, target_class_idx, class_names):
    """Find true positive, false negative, and false positive examples."""
    examples = {
        'true_positive': None,
        'false_negative': None,
        'false_positive': None
    }
    
    print(f"\nSearching for examples (target class index: {target_class_idx})...")
    
    for idx, row in filtered_df.iterrows():
        patient_id = str(row['patient_id'])  # Ensure it's a string
        label_id = str(row['label_id'])  # Ensure it's a string
        
        if patient_id not in patient_graphs:
            continue
        if label_id not in patient_graphs[patient_id]:
            continue
            
        patient_data = patient_graphs[patient_id]
        label_data = patient_data[label_id]
        
        # Check if both graphs exist
        if not (label_data.get("eeg_graphs") and label_data["eeg_graphs"] and
                label_data.get("spectrogram_graphs") and label_data["spectrogram_graphs"]):
            continue
        
        eeg_graph = label_data["eeg_graphs"][0]
        spec_graph = label_data["spectrogram_graphs"][0]
        
        # Get prediction
        predicted_class, probabilities = get_prediction(model, eeg_graph, spec_graph)
        
        # Determine example type
        ground_truth_is_target = (row['expert_consensus'] == class_names[target_class_idx])
        predicted_is_target = (predicted_class == target_class_idx)
        
        example_info = {
            'patient_id': patient_id,
            'label_id': label_id,
            'eeg_graph': eeg_graph,
            'spec_graph': spec_graph,
            'predicted_class': predicted_class,
            'predicted_class_name': class_names[predicted_class],
            'probabilities': probabilities,
            'ground_truth': row['expert_consensus']
        }
        
        # True Positive: ground truth is target AND prediction is target
        if ground_truth_is_target and predicted_is_target and examples['true_positive'] is None:
            examples['true_positive'] = example_info
            print(f"Found TRUE POSITIVE: Patient {patient_id}, Label {label_id}")
        
        # False Negative: ground truth is target BUT prediction is NOT target
        elif ground_truth_is_target and not predicted_is_target and examples['false_negative'] is None:
            examples['false_negative'] = example_info
            print(f"Found FALSE NEGATIVE: Patient {patient_id}, Label {label_id} (predicted as {example_info['predicted_class_name']})")
        
        # False Positive: ground truth is NOT target BUT prediction IS target
        elif not ground_truth_is_target and predicted_is_target and examples['false_positive'] is None:
            examples['false_positive'] = example_info
            print(f"Found FALSE POSITIVE: Patient {patient_id}, Label {label_id} (ground truth: {row['expert_consensus']})")
        
        # Check if we found all examples
        if all(v is not None for v in examples.values()):
            print("All examples found!")
            break
    
    return examples


def run_explanation(model, eeg_graph, spec_graph, target_class_idx):
    """Run GNNExplainer on a single graph."""
    # Create batches
    eeg_batch = Batch.from_data_list([eeg_graph])
    spec_batch = Batch.from_data_list([spec_graph])
    
    # Create wrapped model
    wrapped_model_eeg = EEGExplainerWrapper(model.model, spec_batch)
    wrapped_model_spec = SpecExplainerWrapper(model.model, eeg_batch)
    
    # Set up explainers
    explainer_eeg = Explainer(
        model=wrapped_model_eeg,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='phenomenon',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
        node_mask_type='attributes',
        edge_mask_type='object',
    )

    explainer_spec = Explainer(
        model=wrapped_model_spec,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='phenomenon',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
        node_mask_type='attributes',
        edge_mask_type='object',
    )
    
    target = torch.tensor([target_class_idx])
    
    print("  Running GNNExplainer on EEG graphs...")
    
    # Run explanations
    explanation_eeg = explainer_eeg(
        x=eeg_batch.x,
        edge_index=eeg_batch.edge_index,
        batch=eeg_batch.batch,
        edge_attr=eeg_batch.edge_attr,
        target=target
    )

    print("  Running GNNExplainer on Spectrogram graphs...")
    
    # Run explanations
    explanation_spec = explainer_spec(
        x=spec_batch.x,
        edge_index=spec_batch.edge_index,
        batch=spec_batch.batch,
        edge_attr=spec_batch.edge_attr,
        target=target
    )

    return explanation_eeg, explanation_spec


def save_results(example_type, example_info, explanation, output_dir, target_class_name, graph_type):
    """Save explanation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename_prefix = f"{example_type}_{target_class_name}"
    
    # Save graphs
    torch.save({
        'eeg_graph': example_info['eeg_graph'],
        'spec_graph': example_info['spec_graph'],
        'patient_id': example_info['patient_id'],
        'label_id': example_info['label_id'],
        'predicted_class': example_info['predicted_class'],
        'predicted_class_name': example_info['predicted_class_name'],
        'ground_truth': example_info['ground_truth'],
        'probabilities': example_info['probabilities']
    }, os.path.join(output_dir, f"{filename_prefix}_{graph_type}_graphs.pt"))
    
    # Save explanation
    torch.save({
        'node_mask': explanation.node_mask,
        'edge_mask': explanation.edge_mask,
        'example_type': example_type,
        'target_class': target_class_name
    }, os.path.join(output_dir, f"{filename_prefix}_{graph_type}_explanation.pt"))
    
    print(f"  Saved results to {output_dir}/{filename_prefix}_{graph_type}_*.pt")

def print_explanation(explanation, batch):
    # Print explanation summary
    node_importance = explanation.node_mask.sum(axis=1)
    most_important_node = torch.argmax(node_importance).item()
    feature_importance = explanation.node_mask.sum(axis=0)
    most_important_feature = torch.argmax(feature_importance).item()
    
    print(f"\n--- Explanation Results ---")
    print(f"Node mask:\n{explanation.node_mask}")
    print(f"Most important node: {most_important_node}")
    print(f"Most important feature: {most_important_feature}")
    
    top_k_scores, top_k_indices = torch.topk(explanation.edge_mask, k=5)
    most_important_edges = batch.edge_index[:, top_k_indices]
    print(f"Edge mask:\n{explanation.edge_mask}")
    print(f"Top 5 most important edges:")
    print(most_important_edges)
    

def main():
    args = parse_args()
    
    # Load config and model
    cfg = OmegaConf.load(args.config)
    model = HMSLightningModule(cfg)
    
    state = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    
    model.eval()
    
    # Load patient graphs
    print(f"Loading patient graphs from {args.graphs}...")
    patient_graphs = torch.load(args.graphs)
    
    # Load train.csv
    print(f"Loading train.csv from {args.train_csv}...")
    train_df = pd.read_csv(args.train_csv)
    
    # Define class names (order matters!)
    class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
    
    target_class_name = args.target_class
    target_class_idx = class_names.index(target_class_name)
    
    print(f"\n=== Target Class: {target_class_name} (index {target_class_idx}) ===")
    
    # Get available patient_ids and label_ids from patient_graphs
    available_patient_ids = set(str(k) for k in patient_graphs.keys())
    print(f"Available patient_ids in patient_graphs: {len(available_patient_ids)}")
    
    # Convert IDs in train_df to strings for comparison
    train_df['patient_id'] = train_df['patient_id'].astype(str)
    train_df['label_id'] = train_df['label_id'].astype(str)
    
    # For True Positive and False Negative: filter by expert_consensus == target_class
    tp_fn_filtered = train_df[
        (train_df['expert_consensus'] == target_class_name) & 
        (train_df['patient_id'].isin(available_patient_ids))
    ].copy()
    
    # For False Positive: filter by expert_consensus != target_class
    fp_filtered = train_df[
        (train_df['expert_consensus'] != target_class_name) & 
        (train_df['patient_id'].isin(available_patient_ids))
    ].copy()
    
    print(f"Found {len(tp_fn_filtered)} entries with expert_consensus == {target_class_name}")
    print(f"Found {len(fp_filtered)} entries with expert_consensus != {target_class_name}")
    
    # Further filter both for label_ids that exist for each patient
    def filter_valid_labels(df):
        valid_rows = []
        for idx, row in df.iterrows():
            patient_id = str(row['patient_id'])
            label_id = str(row['label_id'])
            if patient_id in patient_graphs and label_id in patient_graphs[patient_id]:
                valid_rows.append(idx)
        return df.loc[valid_rows]
    
    tp_fn_filtered = filter_valid_labels(tp_fn_filtered)
    fp_filtered = filter_valid_labels(fp_filtered)
    
    print(f"After filtering for available label_ids:")
    print(f"  TP/FN candidates: {len(tp_fn_filtered)} entries")
    print(f"  FP candidates: {len(fp_filtered)} entries")
    
    if len(tp_fn_filtered) == 0 and len(fp_filtered) == 0:
        print("Error: No valid entries found after filtering.", file=sys.stderr)
        sys.exit(1)
    
    # Combine both dataframes for searching
    filtered_df = pd.concat([tp_fn_filtered, fp_filtered], ignore_index=True)
    
    # Find examples
    examples = find_examples(model, patient_graphs, filtered_df, target_class_idx, class_names)
    
    # Check which examples were found
    found_examples = {k: v for k, v in examples.items() if v is not None}
    if not found_examples:
        print("\nError: No examples found!", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nFound {len(found_examples)} example(s): {list(found_examples.keys())}")
    
    # Run explanations for each found example
    for example_type, example_info in found_examples.items():
        print(f"\n{'='*60}")
        print(f"Processing {example_type.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        print(f"Patient ID: {example_info['patient_id']}")
        print(f"Label ID: {example_info['label_id']}")
        print(f"Ground Truth: {example_info['ground_truth']}")
        print(f"Predicted: {example_info['predicted_class_name']} (class {example_info['predicted_class']})")
        print(f"Probabilities: {example_info['probabilities']}")
        
        # Run explanation
        explanation_eeg, explanation_spec = run_explanation(
            model, 
            example_info['eeg_graph'], 
            example_info['spec_graph'], 
            target_class_idx
        )
        print_explanation(explanation_eeg, Batch.from_data_list([example_info['eeg_graph']]))
        save_results(example_type, example_info, explanation_eeg, args.output_dir, target_class_name, "EEG")
        print_explanation(explanation_spec, Batch.from_data_list([example_info['spec_graph']]))
        save_results(example_type, example_info, explanation_spec, args.output_dir, target_class_name, "Spec")
        
    
    print(f"\n{'='*60}")
    print("All explanations completed!")
    print(f"Results saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()