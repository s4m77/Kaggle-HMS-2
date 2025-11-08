import argparse
import sys
from pathlib import Path
import torch
from torch_geometric.data import Batch
from torch_geometric.explain import Explainer, GNNExplainer
from typing import List, Optional

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.hms_model import HMSMultiModalGNN
from src.models.explainer_wrappers import ExplanationWrapper
from src.lightning_trainer.graph_lightning_module import HMSLightningModule

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GNNExplainer on the HMSMultiModalGNN model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint file (model.ckpt)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data sample file (patient_*.pt)')
    parser.add_argument('--modality', type=str, choices=['eeg', 'spec'], default=None, help='Specific modality to explain (eeg or spec). If omitted, explains all.')
    parser.add_argument('--index', type=int, default=None, help='Specific graph index to explain. If omitted, explains all graphs in the chosen modality.')
    return parser.parse_args()

def explain_single_graph(
    model: HMSMultiModalGNN,
    eeg_graphs_sample: List[Batch],
    spec_graphs_sample: List[Batch],
    modality: str,
    index: int,
    use_edge_attr_bool: bool,
    prediction: torch.Tensor
):
    """
    Helper function to run GNNExplainer on a single graph.
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
        edge_mask_type=None,
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
    print("--------------------------------------")

def run_explanation(
    model_path: str,
    data_path: str,
    modality_to_explain: Optional[str] = None,
    graph_to_explain_idx: Optional[int] = None,
):
    """
    Loads a model and data sample, then runs GNNExplainer.
    
    Controls behavior based on arguments:
    - modality_to_explain: 'eeg', 'spec', or None (for all)
    - graph_to_explain_idx: Integer index or None (for all)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the Lightning module from checkpoint
    print(f"Loading checkpoint from: {model_path}")
    print(f"Device: {device}")
    
    try:
        lightning_module = HMSLightningModule.load_from_checkpoint(model_path)
        print("Lightning module loaded successfully.")
        
        # Extract the underlying model
        model = lightning_module.model
        model.to(device)
        model.eval()
        print("Model extracted and set to eval mode.")
        
    except Exception as e:
        print(f"Error loading checkpoint with Lightning: {e}")
        print(f"\nTrying alternative loading with raw torch.load()...")
        try:
            # Try loading with raw torch.load as fallback
            checkpoint = torch.load(model_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                print("Found 'state_dict' in checkpoint. Attempting to create model and load weights...")
                
                # Create a new model instance
                model_config = {
                    "eeg_config": {"use_edge_attr": True},
                    "spec_config": {"use_edge_attr": False}
                }
                model = HMSMultiModalGNN(**model_config)
                
                state_dict = checkpoint["state_dict"]
                clean_state_dict = {}
                prefix_to_strip = "model."
                
                for k, v in state_dict.items():
                    if k.startswith(prefix_to_strip): 
                        clean_state_dict[k[len(prefix_to_strip):]] = v
                    else:
                        clean_state_dict[k] = v
                
                model.load_state_dict(clean_state_dict, strict=False)
                model.to(device)
                model.eval()
                print("Model loaded successfully with fallback method.")
            else:
                print("Checkpoint structure not recognized.")
                return
                
        except Exception as fallback_error:
            print(f"Fallback loading also failed: {fallback_error}")
            print(f"\nCheckpoint file may be corrupted. Please verify the file exists and is valid.")
            print(f"File path: {model_path}")
            print(f"File size: {Path(model_path).stat().st_size if Path(model_path).exists() else 'File not found'} bytes")
            return

    # Load one full data sample
    print(f"Loading data sample from {data_path}...")
    try:
        data_dict = torch.load(data_path, map_location='cpu')
        sample_id = list(data_dict.keys())[0]
        sample_data = data_dict[sample_id]
        
        eeg_data_list = sample_data['eeg_graphs']
        spec_data_list = sample_data['spec_graphs']
        target_label = sample_data['target']
        
        print(f"Loaded sample for ID: {sample_id} with target: {target_label}")

        eeg_graphs_sample = [Batch.from_data_list([data]).to(device) for data in eeg_data_list]
        
        spec_graphs_sample = []
        for data in spec_data_list:
            data.x = data.x[:, :4]
            spec_graphs_sample.append(Batch.from_data_list([data]).to(device))
        
        print("Data sample loaded and converted to List[Batch] successfully.")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Get the model's overall prediction ONCE
    with torch.no_grad():
        logits = model(eeg_graphs_sample, spec_graphs_sample)
        prediction = logits.argmax(dim=-1)
        print(f"Original model prediction: class {prediction.item()} (True label: {target_label})")

    # --- LOOP LOGIC BASED ON ARGUMENTS ---
    
    # Loop 1: Explain EEG Graphs
    if modality_to_explain in [None, 'eeg']:
        # EEG uses edge attributes by default
        eeg_edge_attr_bool = True
        num_eeg_graphs = len(eeg_graphs_sample)
        
        # Determine which indices to run
        if graph_to_explain_idx is None:
            eeg_indices_to_run = range(num_eeg_graphs)
            print(f"\n===== STARTING ALL EEG GRAPH EXPLANATIONS ({num_eeg_graphs} graphs) =====")
        else:
            eeg_indices_to_run = [graph_to_explain_idx]
            print(f"\n===== STARTING EEG GRAPH {graph_to_explain_idx} EXPLANATION =====")

        for i in eeg_indices_to_run:
            if i >= num_eeg_graphs:
                print(f"Warning: Index {i} is out of bounds for EEG (size={num_eeg_graphs}). Skipping.")
                continue
                
            print(f"\n--- Explaining EEG Graph {i} ---")
            explain_single_graph(
                model=model,
                eeg_graphs_sample=eeg_graphs_sample,
                spec_graphs_sample=spec_graphs_sample,
                modality='eeg',
                index=i,
                use_edge_attr_bool=eeg_edge_attr_bool,
                prediction=prediction
            )

    # Loop 2: Explain Spectrogram Graphs
    if modality_to_explain in [None, 'spec']:
        # Spectrogram does not use edge attributes by default
        spec_edge_attr_bool = False
        num_spec_graphs = len(spec_graphs_sample)
        
        # Determine which indices to run
        if graph_to_explain_idx is None:
            spec_indices_to_run = range(num_spec_graphs)
            print(f"\n===== STARTING ALL SPECTROGRAM GRAPH EXPLANATIONS ({num_spec_graphs} graphs) =====")
        else:
            spec_indices_to_run = [graph_to_explain_idx]
            print(f"\n===== STARTING SPECTROGRAM GRAPH {graph_to_explain_idx} EXPLANATION =====")

        for i in spec_indices_to_run:
            if i >= num_spec_graphs:
                print(f"Warning: Index {i} is out of bounds for Spectrogram (size={num_spec_graphs}). Skipping.")
                continue

            print(f"\n--- Explaining Spectrogram Graph {i} ---")
            explain_single_graph(
                model=model,
                eeg_graphs_sample=eeg_graphs_sample,
                spec_graphs_sample=spec_graphs_sample,
                modality='spec',
                index=i,
                use_edge_attr_bool=spec_edge_attr_bool,
                prediction=prediction
            )
    
    print("\n===== All explanations complete. =====")

if __name__ == "__main__":

    args = parse_args()

    print(f"Starting explanation with config: Modality={args.modality}, Index={args.index}")
    
    run_explanation(
        model_path=args.model_path,
        data_path=args.data_path,
        modality_to_explain=args.modality,
        graph_to_explain_idx=args.index
    )