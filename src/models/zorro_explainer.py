"""ZORRO (Zero-Order Rank-based Relative Output) explainer for Graph Neural Networks.

ZORRO is a post-hoc explanation method that identifies important nodes and node features
by analyzing how model predictions change when nodes/features are perturbed.

Reference: https://arxiv.org/abs/2305.02783
"""

from __future__ import annotations

import warnings
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Batch
from tqdm import tqdm


@dataclass
class ZORROExplanation:
    """Container for ZORRO explanation results."""
    
    node_importance: torch.Tensor  # (num_nodes, num_features) - importance scores
    node_indices: List[int]  # Global node indices in the graph
    top_k_nodes: List[Tuple[int, float]]  # [(node_idx, importance), ...] sorted by importance
    feature_importance: torch.Tensor  # (num_features,) - aggregated feature importance
    prediction_original: torch.Tensor  # Original model prediction
    modality: str  # "eeg" or "spec"
    
    def __repr__(self) -> str:
        return (
            f"ZORROExplanation(\n"
            f"  node_importance: {self.node_importance.shape},\n"
            f"  num_nodes: {len(self.node_indices)},\n"
            f"  top_k_nodes (k=5): {self.top_k_nodes[:5]},\n"
            f"  feature_importance: {self.feature_importance.shape},\n"
            f"  prediction_original: {self.prediction_original.shape},\n"
            f"  modality: {self.modality}\n"
            f")"
        )


class ZORROExplainer:
    """ZORRO explainer for multi-modal GNN models.
    
    This class explains predictions from a trained GNN model by identifying which
    nodes and node features are most important for the model's decisions.
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained GNN model to explain
    target_class : int, optional
        The class to explain. If None, explains the predicted class.
    device : torch.device, optional
        Device to run computations on. Defaults to model's device.
    perturbation_mode : str
        How to perturb features: "zero" (set to 0), "noise" (add Gaussian noise),
        or "mean" (set to feature mean)
    noise_std : float
        Standard deviation for noise perturbations (if perturbation_mode="noise")
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_class: Optional[int] = None,
        device: Optional[torch.device] = None,
        perturbation_mode: str = "zero",
        noise_std: float = 0.1,
    ) -> None:
        self.model = model
        self.target_class = target_class
        self.perturbation_mode = perturbation_mode
        self.noise_std = noise_std
        
        if device is None:
            device = next(model.parameters()).device
        self.device = device
        
        self.model.eval()
    
    def explain_batch(
        self,
        eeg_graphs: List[Batch],
        spec_graphs: List[Batch],
        sample_indices: Optional[List[int]] = None,
        top_k: int = 10,
        return_eeg: bool = True,
        return_spec: bool = True,
        n_samples: int = 5,
        pbar: bool = True,
    ) -> Dict[str, ZORROExplanation]:
        """Explain predictions for a batch of samples.
        
        Parameters
        ----------
        eeg_graphs : List[Batch]
            List of 9 EEG graphs
        spec_graphs : List[Batch]
            List of 119 spectrogram graphs
        sample_indices : List[int], optional
            Indices of samples in batch to explain. If None, explains all.
        top_k : int
            Number of top nodes to return
        return_eeg : bool
            Whether to explain EEG modality
        return_spec : bool
            Whether to explain spectrogram modality
        n_samples : int
            Number of perturbation samples for Monte Carlo estimation
        pbar : bool
            Whether to show progress bar
        
        Returns
        -------
        Dict[str, ZORROExplanation]
            Dictionary with keys "eeg" and/or "spec" containing explanations
        """
        batch_size = eeg_graphs[0].num_graphs
        
        if sample_indices is None:
            sample_indices = list(range(batch_size))
        
        explanations = {}
        
        if return_eeg:
            eeg_exp = self._explain_modality(
                graphs=eeg_graphs,
                modality="eeg",
                sample_indices=sample_indices,
                top_k=top_k,
                n_samples=n_samples,
                pbar=pbar,
            )
            explanations["eeg"] = eeg_exp
        
        if return_spec:
            spec_exp = self._explain_modality(
                graphs=spec_graphs,
                modality="spec",
                sample_indices=sample_indices,
                top_k=top_k,
                n_samples=n_samples,
                pbar=pbar,
            )
            explanations["spec"] = spec_exp
        
        return explanations
    
    def _explain_modality(
        self,
        graphs: List[Batch],
        modality: str,
        sample_indices: List[int],
        top_k: int,
        n_samples: int,
        pbar: bool,
    ) -> Dict[int, ZORROExplanation]:
        """Explain one modality for given samples.
        
        Returns
        -------
        Dict[int, ZORROExplanation]
            Mapping from sample index to explanation
        """
        explanations = {}
        iterator = tqdm(sample_indices, desc=f"Explaining {modality}", disable=not pbar)
        
        for sample_idx in iterator:
            exp = self.explain_sample(
                graphs=graphs,
                modality=modality,
                sample_idx=sample_idx,
                top_k=top_k,
                n_samples=n_samples,
                pbar=False,
            )
            explanations[sample_idx] = exp
        
        return explanations
    
    def explain_sample(
        self,
        graphs: List[Batch],
        modality: str,
        sample_idx: int,
        top_k: int = 10,
        n_samples: int = 5,
        pbar: bool = True,
        other_modality_graphs: Optional[List[Batch]] = None,
    ) -> ZORROExplanation:
        """Explain prediction for a single sample and modality.
        
        Parameters
        ----------
        graphs : List[Batch]
            List of temporal graphs (9 for EEG, 119 for spec)
        modality : str
            "eeg" or "spec"
        sample_idx : int
            Index of sample to explain in batch
        top_k : int
            Number of top nodes to return
        n_samples : int
            Number of perturbation samples for Monte Carlo
        pbar : bool
            Whether to show progress bar
        other_modality_graphs : Optional[List[Batch]]
            Graphs for the other modality (e.g., spec graphs if explaining EEG).
            If not provided, dummy graphs will be created.
        
        Returns
        -------
        ZORROExplanation
            Explanation for the sample
        """
        with torch.no_grad():
            # Get original prediction
            if modality == "eeg":
                eeg_graphs = graphs
                if other_modality_graphs is not None:
                    spec_graphs = other_modality_graphs
                else:
                    # Create dummy spec graphs - need to match spec feature dimension (4)
                    spec_graphs = self._create_dummy_graphs(
                        graphs, num_graphs=119, num_nodes_per_graph=9, num_features=4
                    )
            else:  # spec
                spec_graphs = graphs
                if other_modality_graphs is not None:
                    eeg_graphs = other_modality_graphs
                else:
                    # Create dummy EEG graphs - need to match EEG feature dimension (5)
                    eeg_graphs = self._create_dummy_graphs(
                        graphs, num_graphs=9, num_nodes_per_graph=9, num_features=5
                    )
            
            # Get original output
            with torch.no_grad():
                logits = self.model(eeg_graphs, spec_graphs)
            
            original_pred = logits[sample_idx]
            target_class = self.target_class or original_pred.argmax().item()
            
            # Extract sample from graphs
            graph_list = eeg_graphs if modality == "eeg" else spec_graphs
            sample_graphs = [self._extract_sample(g, sample_idx) for g in graph_list]
            
            # Compute node importance scores
            node_importance, node_indices = self._compute_node_importance(
                sample_graphs=sample_graphs,
                modality=modality,
                sample_idx=sample_idx,
                target_class=target_class,
                eeg_graphs=eeg_graphs,
                spec_graphs=spec_graphs,
                n_samples=n_samples,
                pbar=pbar,
            )
            
            # Get feature-level importance by aggregating over nodes
            feature_importance = node_importance.mean(dim=0)  # (num_features,)
            
            # Get top-k nodes
            node_importance_sum = node_importance.sum(dim=1)  # (num_nodes,)
            top_k_indices = torch.topk(node_importance_sum, k=min(top_k, len(node_indices)))[1]
            top_k_nodes = [
                (node_indices[idx], node_importance_sum[idx].item())
                for idx in top_k_indices
            ]
            
            return ZORROExplanation(
                node_importance=node_importance,
                node_indices=node_indices,
                top_k_nodes=top_k_nodes,
                feature_importance=feature_importance,
                prediction_original=original_pred,
                modality=modality,
            )
    
    def _compute_node_importance(
        self,
        sample_graphs: List[Batch],
        modality: str,
        sample_idx: int,
        target_class: int,
        eeg_graphs: List[Batch],
        spec_graphs: List[Batch],
        n_samples: int,
        pbar: bool,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Compute importance scores for each node using perturbations.
        
        Returns
        -------
        Tuple[torch.Tensor, List[int]]
            (node_importance scores, global node indices)
        """
        # Collect all nodes from sample graphs
        all_nodes = []
        node_indices = []
        node_offset = 0
        
        for g in sample_graphs:
            num_nodes = g.x.shape[0]
            all_nodes.append(g.x)
            node_indices.extend(list(range(node_offset, node_offset + num_nodes)))
            node_offset += num_nodes
        
        all_nodes_tensor = torch.cat(all_nodes, dim=0)  # (total_nodes, num_features)
        num_nodes = all_nodes_tensor.shape[0]
        num_features = all_nodes_tensor.shape[1]
        
        # Get baseline prediction
        baseline_logits = self.model(eeg_graphs, spec_graphs)
        baseline_output = baseline_logits[sample_idx, target_class].item()
        
        # Compute importance for each node and feature
        node_importance = torch.zeros(num_nodes, num_features, device=self.device)
        
        iterator = tqdm(
            range(num_nodes),
            desc=f"Computing node importance ({modality})",
            disable=not pbar,
        )
        
        for node_idx in iterator:
            for feat_idx in range(num_features):
                # Run multiple perturbation samples
                deltas = []
                
                for _ in range(n_samples):
                    # Perturb the node feature
                    perturbed_graphs = self._perturb_node_feature(
                        sample_graphs=sample_graphs,
                        node_idx=node_idx,
                        feat_idx=feat_idx,
                        sample_graphs_list=sample_graphs,
                    )
                    
                    # Reconstruct full batch
                    if modality == "eeg":
                        test_eeg = perturbed_graphs
                        test_spec = spec_graphs
                    else:
                        test_eeg = eeg_graphs
                        test_spec = perturbed_graphs
                    
                    # Get perturbed prediction
                    with torch.no_grad():
                        perturbed_logits = self.model(test_eeg, test_spec)
                    
                    perturbed_output = perturbed_logits[sample_idx, target_class].item()
                    
                    # Delta: change in model output
                    delta = baseline_output - perturbed_output
                    deltas.append(delta)
                
                # Average importance over samples
                importance = np.mean(deltas)
                node_importance[node_idx, feat_idx] = abs(importance)
        
        return node_importance, node_indices
    
    def _perturb_node_feature(
        self,
        sample_graphs: List[Batch],
        node_idx: int,
        feat_idx: int,
        sample_graphs_list: List[Batch],
    ) -> List[Batch]:
        """Create perturbed graphs with one node feature perturbed.
        
        Returns
        -------
        List[Batch]
            Perturbed graphs
        """
        perturbed_graphs = []
        current_node_offset = 0
        
        for g in sample_graphs_list:
            g_perturbed = self._copy_batch(g)
            num_nodes = g.x.shape[0]
            
            # Check if target node is in this graph
            if current_node_offset <= node_idx < current_node_offset + num_nodes:
                local_idx = node_idx - current_node_offset
                
                if self.perturbation_mode == "zero":
                    g_perturbed.x[local_idx, feat_idx] = 0.0
                elif self.perturbation_mode == "mean":
                    # Use mean of this feature across all nodes
                    mean_val = g.x[:, feat_idx].mean()
                    g_perturbed.x[local_idx, feat_idx] = mean_val
                elif self.perturbation_mode == "noise":
                    # Add Gaussian noise
                    noise = torch.randn_like(g_perturbed.x[local_idx:local_idx+1, feat_idx:feat_idx+1])
                    noise = noise * self.noise_std
                    g_perturbed.x[local_idx, feat_idx] += noise.squeeze()
            
            perturbed_graphs.append(g_perturbed)
            current_node_offset += num_nodes
        
        return perturbed_graphs
    
    def _extract_sample(self, batch: Batch, sample_idx: int) -> Batch:
        """Extract a single sample from a batched graph.
        
        Returns
        -------
        Batch
            Single sample graph
        """
        # Get the mask for this sample
        mask = batch.batch == sample_idx
        
        new_batch = Batch(
            x=batch.x[mask].clone(),
            edge_index=None,
            batch=torch.zeros(mask.sum(), dtype=torch.long, device=batch.x.device),
        )
        
        # Filter edges
        if batch.edge_index is not None:
            edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
            if edge_mask.any():
                # Remap edge indices
                node_mapping = torch.full((batch.x.shape[0],), -1, dtype=torch.long)
                node_mapping[mask] = torch.arange(mask.sum())
                
                filtered_edges = batch.edge_index[:, edge_mask]
                new_batch.edge_index = node_mapping[filtered_edges]
        
        # Copy edge attributes if present
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            if batch.edge_index is not None:
                edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
                new_batch.edge_attr = batch.edge_attr[edge_mask].clone()
        
        return new_batch
    
    def _copy_batch(self, batch: Batch) -> Batch:
        """Create a deep copy of a batch."""
        new_batch = Batch(
            x=batch.x.clone(),
            edge_index=batch.edge_index.clone() if batch.edge_index is not None else None,
            batch=batch.batch.clone() if hasattr(batch, 'batch') else None,
        )
        
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            new_batch.edge_attr = batch.edge_attr.clone()
        
        return new_batch
    
    def _create_dummy_graphs(
        self,
        sample_graphs: List[Batch],
        num_graphs: int,
        num_nodes_per_graph: int,
        num_features: Optional[int] = None,
    ) -> List[Batch]:
        """Create dummy graphs to match model input expectations.
        
        Parameters
        ----------
        sample_graphs : List[Batch]
            Sample graphs to use as reference
        num_graphs : int
            Number of dummy graphs to create
        num_nodes_per_graph : int
            Nodes per graph
        num_features : Optional[int]
            Number of features per node. If None, uses sample_graphs[0].x.shape[1]
        
        Returns
        -------
        List[Batch]
            Dummy graphs with random features
        """
        dummy_graphs = []
        batch_size = sample_graphs[0].num_graphs
        total_nodes = batch_size * num_nodes_per_graph
        
        if num_features is None:
            num_features = sample_graphs[0].x.shape[1]
        
        for _ in range(num_graphs):
            # Create dummy graph with random features
            dummy_x = torch.randn(
                total_nodes,
                num_features,
                device=self.device,
            )
            
            # Create edge_index (self-loops for all nodes)
            dummy_edge_index = torch.arange(total_nodes, device=self.device).unsqueeze(0)
            dummy_edge_index = torch.cat([dummy_edge_index, dummy_edge_index], dim=0)
            
            # Create is_center mask (mark first node of each batch as center)
            dummy_is_center = torch.zeros(total_nodes, dtype=torch.bool, device=self.device)
            for i in range(batch_size):
                dummy_is_center[i * num_nodes_per_graph] = True
            
            dummy_batch = Batch(
                x=dummy_x,
                edge_index=dummy_edge_index,
                batch=torch.arange(batch_size, device=self.device).repeat_interleave(num_nodes_per_graph),
                is_center=dummy_is_center,
            )
            
            dummy_graphs.append(dummy_batch)
        
        return dummy_graphs


__all__ = ["ZORROExplainer", "ZORROExplanation"]
