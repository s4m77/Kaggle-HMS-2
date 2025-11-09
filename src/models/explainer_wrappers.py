import torch
from src.models.hms_model import HMSMultiModalGNN
from torch_geometric.data import Data, Batch
import torch.nn as nn
from typing import List, Optional

class ExplanationWrapper(nn.Module):
    """
    Wraps the HMSMultiModalGNN for GNNExplainer.
    
    This wrapper is "explanation-aware". It isolates the forward pass
    to only the modality being explained, preventing explainer hooks
    from interfering with the other modality's encoder.
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
        self.device = next(model.parameters()).device 

        # Pre-compute and cache the features for the other modality.
        # We do this once, with no gradients, so the explainer's hooks
        # (which are active during the explainer's training loop)
        # do not see or interfere with this computation.
        with torch.no_grad():
            self.model.eval()  # Use eval mode for no_grad forward pass
            if self.modality == 'eeg':
                # We are explaining EEG, so we cache the original SPEC features
                self.other_features = self.model.spec_encoder(
                    self.spec_graphs_sample, return_sequence=False
                )
            else:
                # We are explaining SPEC, so we cache the original EEG features
                self.other_features = self.model.eeg_encoder(
                    self.eeg_graphs_sample, return_sequence=False
                )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This forward pass is what GNNExplainer will call.
        It now only runs the relevant encoder.
        """
        # We have to call .train() for the CUDNN RNN backward pass to work.
        self.model.train() 
        
        # but, we also manually disable dropout/norm layers
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.AlphaDropout, nn.FeatureAlphaDropout, 
                                   nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                   nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)):
                module.eval()
        
        perturbed_data = self.original_data_to_explain.clone()
        perturbed_data.x = x
        perturbed_data.edge_index = edge_index
        
        if self.use_edge_attr:
            if edge_attr is not None:
                perturbed_data.edge_attr = edge_attr
            elif hasattr(perturbed_data, "edge_attr"):
                 del perturbed_data.edge_attr
        else:
            if hasattr(perturbed_data, "edge_attr"):
                del perturbed_data.edge_attr

        perturbed_batch = Batch.from_data_list([perturbed_data]).to(self.device)

        # Run ONLY the necessary parts of the model ---
        if self.modality == 'eeg':

            eeg_graphs = list(self.eeg_graphs_sample) 
            eeg_graphs[self.idx] = perturbed_batch
            
            current_features = self.model.eeg_encoder(eeg_graphs, return_sequence=False)

            other_features = self.other_features
            
            fused_features = self.model.fusion(current_features, other_features)
            
        else: # modality == 'spec'
            spec_graphs = list(self.spec_graphs_sample)
            spec_graphs[self.idx] = perturbed_batch
            
            current_features = self.model.spec_encoder(spec_graphs, return_sequence=False)
            
            other_features = self.other_features
            
            fused_features = self.model.fusion(other_features, current_features)

        logits = self.model.classifier(fused_features)

        return logits