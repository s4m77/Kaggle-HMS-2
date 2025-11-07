import torch
from src.models.hms_model import HMSMultiModalGNN
from torch_geometric.data import Data, Batch
import torch.nn as nn
from typing import List, Optional

class ExplanationWrapper(nn.Module):
    """
    Wraps the HMSMultiModalGNN for GNNExplainer.
    ...
    """
    def __init__(
        self,
        model: HMSMultiModalGNN,
        eeg_graphs_sample: List[Batch],
        spec_graphs_sample: List[Batch],
        original_data_to_explain: Data,
        graph_to_explain_idx: int,
        modality: str = 'eeg',
        use_edge_attr: bool = True
    ):
        super().__init__()
        self.model = model
        self.eeg_graphs_sample = eeg_graphs_sample
        self.spec_graphs_sample = spec_graphs_sample
        self.idx = graph_to_explain_idx
        self.modality = modality
        self.use_edge_attr = use_edge_attr
        self.original_data_to_explain = original_data_to_explain

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This forward pass is what GNNExplainer will call.
        """
        
        self.model.train()

        # Create the new perturbed Data object by cloning the original
        perturbed_data = self.original_data_to_explain.clone()
        
        # Replace attributes with the perturbed ones
        perturbed_data.x = x
        perturbed_data.edge_index = edge_index
        
        if self.use_edge_attr:
            perturbed_data.edge_attr = edge_attr
        else:
            if hasattr(perturbed_data, "edge_attr"):
                del perturbed_data.edge_attr

        # Convert it back to a Batch (for B=1)
        perturbed_batch = Batch.from_data_list([perturbed_data])

        # Re-assemble the full input sequence
        if self.modality == 'eeg':
            eeg_graphs = list(self.eeg_graphs_sample) 
            eeg_graphs[self.idx] = perturbed_batch
            spec_graphs = self.spec_graphs_sample
        else:
            eeg_graphs = self.eeg_graphs_sample
            spec_graphs = list(self.spec_graphs_sample)
            spec_graphs[self.idx] = perturbed_batch

        # Call the original model with the modified input
        logits = self.model(eeg_graphs, spec_graphs)

        # GNNExplainer's 'raw' return_type expects logits
        return logits