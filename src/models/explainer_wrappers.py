from torch_geometric.data import Batch
from src.models.hms_model import HMSMultiModalGNN
import torch.nn as nn

class EEGExplainerWrapper(nn.Module):
    """
    Wraps the HMSMultiModalGNN to explain the EEG modality.
    It holds a fixed spectrogram graph and passes the perturbed
    EEG inputs to the full model.
    """
    def __init__(self, model: HMSMultiModalGNN, spectrogram_batch: Batch):
        super().__init__()
        self.model = model
        # Store the fixed spectrogram batch
        # This graph will NOT be perturbed
        self.spectrogram_batch = spectrogram_batch

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        # Reconstruct the EEG graph from the perturbed inputs
        # The Explainer will pass new (x, edge_index, etc.) here
        eeg_batch = Batch(x=x, 
                          edge_index=edge_index, 
                          batch=batch)
        
        # Add edge attributes if the model uses them
        if edge_attr is not None:
            eeg_batch.edge_attr = edge_attr
        
        # Get the fixed spectrogram graph, ensuring it's on the same device
        spec_batch = self.spectrogram_batch.to(x.device)

        # Manually disable explanations on the spec_encoder's modules
        # to prevent them from consuming the EEG edge_mask.
        original_states = {}
        spec_encoder = self.model.spec_encoder
        for module in spec_encoder.modules():
            if hasattr(module, "explain"):
                original_states[module] = module.explain
                module.explain = False  # Disable explanations

        try:
            # Call the original model
            # The eeg_encoder will run with explain=True (default)
            # The spec_encoder will run with explain=False
            output = self.model(eeg_graph=eeg_batch, 
                                spectrogram_graph=spec_batch)
        finally:
            # Restore original states
            for module, state in original_states.items():
                if module in original_states:
                    module.explain = state
        
        # We only return logits, as required by the Explainer
        return output["logits"]
    
class SpecExplainerWrapper(nn.Module):
    """
    Wraps the HMSMultiModalGNN to explain the Spectrogram modality.
    It holds a fixed eeg graph and passes the perturbed
    spectrogram inputs to the full model.
    """
    def __init__(self, model: HMSMultiModalGNN, eeg_batch: Batch):
        super().__init__()
        self.model = model
        # Store the fixed eeg batch
        # This graph will NOT be perturbed
        self.eeg_batch = eeg_batch

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        # Reconstruct the Spec graph from the perturbed inputs
        # The Explainer will pass new (x, edge_index, etc.) here
        spec_batch = Batch(x=x, 
                          edge_index=edge_index, 
                          batch=batch)
        
        # Add edge attributes if the model uses them
        if edge_attr is not None:
            spec_batch.edge_attr = edge_attr
        
        # Get the fixed spectrogram graph, ensuring it's on the same device
        eeg_batch = self.eeg_batch.to(x.device)

        # Manually disable explanations on the eeg_encoder's modules
        # to prevent them from consuming the spec edge_mask.
        original_states = {}
        eeg_encoder = self.model.eeg_encoder
        for module in eeg_encoder.modules():
            if hasattr(module, "explain"):
                original_states[module] = module.explain
                module.explain = False  # Disable explanations

        try:
            # Call the original model
            # The spec_encoder will run with explain=True (default)
            # The eeg_encoder will run with explain=False
            output = self.model(eeg_graph=eeg_batch, 
                                spectrogram_graph=spec_batch)
        finally:
            # Restore original states
            for module, state in original_states.items():
                if module in original_states:
                    module.explain = state
        
        # Return logits, as required by the Explainer
        return output["logits"]