import argparse
import sys
from pathlib import Path
import torch
from torch_geometric.data import Batch
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer, AttentionExplainer
from torch_geometric.explain.metric import (fidelity, characterization_score, fidelity_curve_auc, unfaithfulness)
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple, Dict
from src.data.graph_datamodule import HMSDataModule 
from src.lightning_trainer.graph_lightning_module import HMSLightningModule
from src.models.hms_model import HMSMultiModalGNN
from src.models.explainer_wrappers import ExplanationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GNNExplainer on the HMSMultiModalGNN model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint file (model.ckpt)')
    parser.add_argument('--data_dir', type=str, default="data/processed", help='Directory containing preprocessed patient files')
    parser.add_argument('--train_csv', type=str, default="data/raw/train_unique.csv", help='Path to train_unique.csv with metadata')
    parser.add_argument('--n_folds', type=int, default=5, help='Total number of folds used in training')
    parser.add_argument('--current_fold', type=int, default=0, help='The fold to use for the test/validation set')
    parser.add_argument('--num_explain', type=int, default=5, help='Number of samples from the test set to explain. Set to -1 for all.')
    parser.add_argument('--output_dir', type=str, default='explanations', help='Directory to save explanation results')
    parser.add_argument('--modality', type=str, choices=['eeg', 'spec'], default=None, help='Specific modality to explain (eeg or spec). If omitted, explains all.')
    parser.add_argument('--index', type=int, default=None, help='Specific graph index *within a sample* to explain. If omitted, explains all graphs.')
    return parser.parse_args()

def explain_single_graph(
    model: HMSMultiModalGNN,
    eeg_graphs_sample: List[Batch],
    spec_graphs_sample: List[Batch],
    modality: str,
    index: int,
    use_edge_attr_bool: bool,
    prediction: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Helper function to run GNNExplainer on a single graph.
    Now returns the node_mask tensor and a dictionary of all metrics.
    """
    if modality == 'eeg':
        graph_to_explain_batch = eeg_graphs_sample[index]
    else:
        graph_to_explain_batch = spec_graphs_sample[index]
        
    graph_to_explain_data = graph_to_explain_batch.to_data_list()[0]

    # Instantiate the wrapper model for this specific graph
    wrapper_model = ExplanationWrapper(
        model,
        eeg_graphs_sample,
        spec_graphs_sample,
        graph_to_explain_data,
        index,
        modality,
        use_edge_attr=use_edge_attr_bool
    )

    # Configure the Explainer
    explainer = Explainer(
        model=wrapper_model,
        algorithm=GNNExplainer(epochs=100, lr=0.01),
        explanation_type='model',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
        node_mask_type='attributes',
        edge_mask_type='object',
    )

    # Get the inputs for the explainer call
    x = graph_to_explain_data.x
    edge_index = graph_to_explain_data.edge_index
    edge_attr = graph_to_explain_data.edge_attr
    
    print(f"Running GNNExplainer for this graph (predicts class {prediction.item()})...")

    # Run the explanation
    if edge_attr is not None and wrapper_model.use_edge_attr:
        explanation = explainer(x=x, edge_index=edge_index, edge_attr=edge_attr)
    else:
        explanation = explainer(x=x, edge_index=edge_index)

    # Print the results for this graph
    print(f"--- Results for {modality.upper()} Graph {index} ---")
    print(f"Node Feature Mask (shape: {explanation.node_mask.shape}):")
    print(explanation.node_mask)
    print(f"Edge Feature Mask (shape: {explanation.edge_mask.shape}):")
    print(explanation.edge_mask)
    
    print(f"Computing explanation metrics...")
    try:
        is_valid = explanation.validate()
        print(f"Explanation validation: {is_valid}")
        # Fidelity
        fid_pos, fid_neg = fidelity(explainer, explanation)
        print(f"Fidelity - Positive: {fid_pos}, Negative: {fid_neg}")
        # Unfaithfulness
        unfaith = unfaithfulness(explainer, explanation)
        print(f"Unfaithfulness: {unfaith}")
        # Characterization score
        char_score = characterization_score(fid_pos, fid_neg, pos_weight=0.7, neg_weight=0.3)
        print(f"Characterization Score: {char_score}")
        # Fidelity curve AUC
        pos_fidelity = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]) 
        neg_fidelity = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]) 

        # x-axis values (e.g., thresholds or steps), sorted in ascending order
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        # Call the fidelity_curve_auc function
        fidelity_auc = fidelity_curve_auc(pos_fidelity, neg_fidelity, x)
        print(f"Fidelity Curve AUC: {fidelity_auc}")
        metrics_dict = {
            'fidelity_pos': fid_pos.item(),
            'fidelity_neg': fid_neg.item(),
            'char_score': char_score.item(),
            'fidelity_auc': fidelity_auc.item(),
            'unfaithfulness': unfaith.item(),
        }
        print(f"Metrics: {metrics_dict}")
        
    except Exception as e:
        print(f"Warning: Could not compute metrics. Error: {e}")
        metrics_dict = {
            'fidelity_pos': -1.0,
            'fidelity_neg': -1.0,
            'char_score': -1.0,
            'fidelity_auc': -1.0,
            'unfaithfulness': -1.0,
        }
    print("--------------------------------------")

    return explanation.node_mask, explanation.edge_mask, metrics_dict

def run_explanation(
    model_path: str,
    data_dir: str,
    train_csv: str,
    n_folds: int,
    current_fold: int,
    num_explain: int,
    output_dir: str,
    modality_to_explain: Optional[str] = None,
    graph_to_explain_idx: Optional[int] = None,
):
    """
    Loads a model and data *from the datamodule*, then runs GNNExplainer
    on samples from the test set and saves the results and metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    save_path_dir = Path(output_dir)
    save_path_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving explanation results to: {save_path_dir.resolve()}")

    model = HMSLightningModule.load_from_checkpoint(model_path)
    model_config = model.hparams['model_config']
    model.to(device)
    model.eval()

    print("Setting up DataModule to get test set...")
    datamodule = HMSDataModule(
        data_dir=data_dir,
        train_csv=train_csv,
        n_folds=n_folds,
        current_fold=current_fold,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    datamodule.setup(stage='fit') 
    
    test_dataset = datamodule.val_dataset
    if test_dataset is None:
        print("Error: Validation dataset was not initialized. Exiting.")
        return

    def simple_collate(batch_list):
        return batch_list[0]

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate
    )
    
    total_samples = len(test_dataset)
    print(f"Loaded test/validation set for fold {current_fold}. Total samples: {total_samples}")
    
    if num_explain == -1:
        samples_to_run = total_samples
        print("Running explanation for ALL samples in the test set.")
    else:
        samples_to_run = min(num_explain, total_samples)
        print(f"Running explanation for {samples_to_run} samples.")

    samples_explained = 0
    for i, sample_data in enumerate(test_loader):
        
        if samples_explained >= samples_to_run:
            print(f"Reached explanation limit ({samples_to_run}). Stopping.")
            break
        
        try:
            eeg_data_list = sample_data['eeg_graphs']
            spec_data_list = sample_data['spec_graphs']
            target_label = sample_data['target']
            
            sample_id_key = 'eeg_id'
            if sample_id_key not in sample_data:
                id_keys = [k for k in sample_data.keys() if str(k).endswith('_id')]
                if id_keys:
                    sample_id_key = id_keys[0]
                else:
                    sample_id_key = 'index'
                    sample_data['index'] = i
                    
            sample_id = sample_data.get(sample_id_key, f"sample_index_{i}")
            
            print(f"\n{'='*80}")
            print(f"Processing sample {i+1}/{total_samples} (ID: {sample_id})")

            eeg_graphs_sample = [Batch.from_data_list([data]).to(device) for data in eeg_data_list]
            
            spec_graphs_sample = []
            for data in spec_data_list:
                data.x = data.x[:, :4]
                spec_graphs_sample.append(Batch.from_data_list([data]).to(device))
            
            print("Data sample loaded and converted to List[Batch] successfully.")

        except Exception as e:
            print(f"Error loading/preparing data for sample {i} (ID: {sample_id}): {e}")
            print("Skipping this sample.")
            continue

        with torch.no_grad():
            logits = model(eeg_graphs_sample, spec_graphs_sample)
            prediction = logits.argmax(dim=-1)
            print(f"Original model prediction: class {prediction.item()} (True label: {target_label})")

        explanation_results = {
            "sample_id": sample_id,
            "prediction": prediction.item(),
            "target": target_label,
            "eeg_masks": [],
            "spec_masks": [],
            "eeg_metrics": [],
            "spec_metrics": [],
        }

        # Explain EEG Graphs
        if modality_to_explain in [None, 'eeg']:
            eeg_edge_attr_bool = model_config["eeg_encoder"].get("use_edge_attr", True)
            num_eeg_graphs = len(eeg_graphs_sample)
            
            indices_key = 'eeg_indices'
            if indices_key not in sample_data:
                sample_data[indices_key] = list(range(num_eeg_graphs))

            if graph_to_explain_idx is None:
                eeg_indices_to_run = range(num_eeg_graphs)
            else:
                eeg_indices_to_run = [graph_to_explain_idx]

            for j in eeg_indices_to_run:
                if j >= num_eeg_graphs:
                    print(f"Warning: Index {j} is out of bounds for EEG (size={num_eeg_graphs}). Skipping.")
                    continue
                    
                print(f"\n--- Explaining EEG Graph {j} ---")

                node_mask, edge_mask, metrics = explain_single_graph(
                    model=model.model,
                    eeg_graphs_sample=eeg_graphs_sample,
                    spec_graphs_sample=spec_graphs_sample,
                    modality='eeg',
                    index=j,
                    use_edge_attr_bool=eeg_edge_attr_bool,
                    prediction=prediction
                )
                explanation_results["eeg_masks"].append(node_mask.cpu())
                explanation_results["eeg_masks"].append(edge_mask.cpu())
                explanation_results["eeg_metrics"].append(metrics)

        # Explain Spectrogram Graphs
        if modality_to_explain in [None, 'spec']:
            spec_edge_attr_bool = model_config["spec_encoder"].get("use_edge_attr", False)
            num_spec_graphs = len(spec_graphs_sample)
            
            indices_key = 'spec_indices'
            if indices_key not in sample_data:
                sample_data[indices_key] = list(range(num_spec_graphs))

            if graph_to_explain_idx is None:
                spec_indices_to_run = range(num_spec_graphs)
            else:
                spec_indices_to_run = [graph_to_explain_idx]

            for j in spec_indices_to_run:
                if j >= num_spec_graphs:
                    print(f"Warning: Index {j} is out of bounds for Spectrogram (size={num_spec_graphs}). Skipping.")
                    continue

                print(f"\n--- Explaining Spectrogram Graph {j} ---")

                node_mask, edge_mask, metrics = explain_single_graph(
                    model=model.model,
                    eeg_graphs_sample=eeg_graphs_sample,
                    spec_graphs_sample=spec_graphs_sample,
                    modality='spec',
                    index=j,
                    use_edge_attr_bool=spec_edge_attr_bool,
                    prediction=prediction
                )
                explanation_results["spec_masks"].append(node_mask.cpu())
                explanation_results["spec_masks"].append(edge_mask.cpu())
                explanation_results["spec_metrics"].append(metrics)
        
        # Save the results 
        try:
            save_file_path = save_path_dir / f"explanation_{sample_id}.pt"
            torch.save(explanation_results, save_file_path)
            print(f"\nSuccessfully saved explanations and metrics for sample {sample_id} to:")
            print(f"{save_file_path}")
        except Exception as e:
            print(f"\nError saving explanation file for sample {sample_id}: {e}")

        samples_explained += 1
    
    print(f"\n{'='*80}")
    print(f"All explanations complete. Total samples explained: {samples_explained}")

if __name__ == "__main__":
    args = parse_args()
    print(f"Starting explanation with config:")
    print(f"  Model: {args.model_path}")
    print(f"  Data Dir: {args.data_dir}")
    print(f"  CSV: {args.train_csv}")
    print(f"  Fold: {args.current_fold}/{args.n_folds - 1}")
    print(f"  Samples to explain: {'ALL' if args.num_explain == -1 else args.num_explain}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Modality: {args.modality or 'All'}")
    print(f"  Graph Index: {args.index or 'All'}")
    run_explanation(
        model_path=args.model_path,
        data_dir=args.data_dir,
        train_csv=args.train_csv,
        n_folds=args.n_folds,
        current_fold=args.current_fold,
        num_explain=args.num_explain,
        output_dir=args.output_dir,
        modality_to_explain=args.modality,
        graph_to_explain_idx=args.index
    )