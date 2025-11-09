"""
NOT WORKING 
GNNExplainer for spatial pattern analysis in EEG graphs.

This script loads a trained model checkpoint and explains which electrodes and
connections are most important for predictions. Focuses on the center window
where the label is applied.

Usage:
    python src/explainers/gnn_explainer.py --checkpoint path/to/checkpoint.ckpt
    python src/explainers/gnn_explainer.py --checkpoint path/to/checkpoint.ckpt --sample_idx 0
    python src/explainers/gnn_explainer.py --checkpoint path/to/checkpoint.ckpt --output_dir outputs/explanations
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from torch_geometric.data import Batch
from torch_geometric.explain import Explainer, GNNExplainer as GNNExplainerAlgo

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.lightning_trainer import HMSLightningModule
from src.data import HMSDataModule


# 10-20 electrode positions for visualization (approximate 2D projection)
ELECTRODE_POSITIONS = {
    'Fp1': (0.3, 0.9),
    'Fp2': (0.7, 0.9),
    'F7': (0.1, 0.7),
    'F3': (0.35, 0.7),
    'Fz': (0.5, 0.7),
    'F4': (0.65, 0.7),
    'F8': (0.9, 0.7),
    'T3': (0.05, 0.5),
    'C3': (0.35, 0.5),
    'Cz': (0.5, 0.5),
    'C4': (0.65, 0.5),
    'T4': (0.95, 0.5),
    'T5': (0.1, 0.3),
    'P3': (0.35, 0.3),
    'Pz': (0.5, 0.3),
    'P4': (0.65, 0.3),
    'T6': (0.9, 0.3),
    'O1': (0.3, 0.1),
    'O2': (0.7, 0.1),
}

CLASS_NAMES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']


class EEGGNNExplainer:
    """Explains EEG graph predictions using GNNExplainer."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize explainer with trained model.
        
        Args:
            checkpoint_path: Path to Lightning checkpoint (.ckpt)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load model from checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.lightning_module = HMSLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device
        )
        self.lightning_module.eval()
        self.lightning_module.to(device)
        
        # Extract the model
        self.model = self.lightning_module.model
        
        # Get EEG encoder's GAT encoder (what we want to explain)
        self.eeg_gat = self.model.eeg_encoder.gat_encoder
        
        print(f"✓ Model loaded successfully")
        print(f"  Device: {device}")
        print(f"  Model type: {self.model.__class__.__name__}")
    
    def explain_sample(
        self,
        eeg_graphs: list,
        spec_graphs: list,
        target_class: Optional[int] = None,
        epochs: int = 200
    ) -> Dict[str, Any]:
        """
        Explain a single sample's prediction.
        
        Args:
            eeg_graphs: List of EEG graphs (9 temporal windows)
            spec_graphs: List of spectrogram graphs (119 temporal windows)
            target_class: Class to explain (if None, uses model's prediction)
            epochs: Number of optimization epochs for GNNExplainer
        
        Returns:
            Dictionary with:
                - 'edge_mask': Importance of each edge (shape: n_edges)
                - 'node_mask': Importance of each node (shape: n_nodes, n_features)
                - 'prediction': Model's prediction (shape: n_classes)
                - 'predicted_class': Predicted class index
                - 'target_class': Class being explained
        """
        # Prepare batches (each window wrapped as Batch) and move to device
        # Batch.from_data_list returns a Batch; move contained tensors manually
        eeg_graphs_device = []
        for g in eeg_graphs:
            b = Batch.from_data_list([g])
            # Move tensor attributes to device
            for attr, val in b.__dict__.items():
                if isinstance(val, torch.Tensor):
                    b.__dict__[attr] = val.to(self.device)
            eeg_graphs_device.append(b)

        spec_graphs_device = []
        for g in spec_graphs:
            b = Batch.from_data_list([g])
            for attr, val in b.__dict__.items():
                if isinstance(val, torch.Tensor):
                    b.__dict__[attr] = val.to(self.device)
            spec_graphs_device.append(b)
        
        with torch.no_grad():
            output = self.model(eeg_graphs_device, spec_graphs_device)
            prediction = torch.softmax(output, dim=-1)[0]  # (n_classes,)
            predicted_class = int(torch.argmax(prediction).item())
        
        # Use predicted class if target not specified
        if target_class is None:
            target_class = predicted_class
        
        print(f"\nExplaining prediction:")
        print(f"  Predicted class: {CLASS_NAMES[predicted_class]} ({prediction[predicted_class]:.3f})")
        print(f"  Explaining class: {CLASS_NAMES[target_class]}")
        
        # Focus on center window (index 4 out of 9 windows)
        center_idx = 4
        center_window = eeg_graphs[center_idx]

        # Precompute fixed components (spectrogram branch + EEG non-center window embeddings)
        with torch.no_grad():
            # Spectrogram features (batch_size=1)
            spec_features = self.model.spec_encoder(spec_graphs_device, return_sequence=False)  # (1, spec_dim)

            # EEG window embeddings without running LSTM yet
            # We'll replicate temporal_encoder forward but skip GAT for center (will be dynamic)
            eeg_window_embeddings = []
            for i, g_batch in enumerate(eeg_graphs_device):
                if i == center_idx:
                    eeg_window_embeddings.append(None)  # placeholder
                else:
                    node_emb = self.eeg_gat(g_batch)  # (num_nodes, gat_out_dim)
                    # Pool nodes (mirror TemporalGraphEncoder._pool_nodes logic: use mean pooling default)
                    from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
                    pooling_method = self.model.eeg_encoder.pooling_method
                    if pooling_method == 'mean':
                        win_emb = global_mean_pool(node_emb, g_batch.batch)
                    elif pooling_method == 'max':
                        win_emb = global_max_pool(node_emb, g_batch.batch)
                    else:
                        win_emb = global_add_pool(node_emb, g_batch.batch)
                    eeg_window_embeddings.append(win_emb)  # (1, gat_out_dim)

            # Store static tensors for wrapper
            static_non_center_embeddings = eeg_window_embeddings  # list with one None at center_idx
            layer_norm = self.model.eeg_encoder.layer_norm
            lstm = self.model.eeg_encoder.lstm
            center_attention = self.model.eeg_encoder.center_attention
            bidirectional = self.model.eeg_encoder.bidirectional

        class CenterEEGWrapper(torch.nn.Module):
            """Wrapper that only recomputes center window GAT embedding and reconstructs EEG sequence."""
            def __init__(
                self,
                gat,
                static_embeddings,
                layer_norm_module,
                lstm_module,
                center_attention_module,
                pooling_method: str,
                fusion_module,
                classifier_module,
                spec_features_tensor,
                center_index: int,
            ):
                super().__init__()
                self.gat = gat
                self.static_embeddings = static_embeddings
                self.layer_norm = layer_norm_module
                self.lstm = lstm_module
                self.center_attention = center_attention_module
                self.spec_features = spec_features_tensor  # (1, spec_dim)
                self.center_idx = center_index
                self.pooling_method = pooling_method
                self.fusion = fusion_module
                self.classifier = classifier_module

            def forward(self, x, edge_index, batch=None):
                from torch_geometric.data import Data
                from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
                device = x.device
                # Build Data object for center window
                data = Data(x=x, edge_index=edge_index, batch=batch if batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=device))
                # Run GAT on center graph
                node_emb = self.gat(data)
                # Pool
                if self.pooling_method == 'mean':
                    center_emb = global_mean_pool(node_emb, data.batch)
                elif self.pooling_method == 'max':
                    center_emb = global_max_pool(node_emb, data.batch)
                else:
                    center_emb = global_add_pool(node_emb, data.batch)

                # Reconstruct sequence
                seq_list = []
                for i, emb in enumerate(self.static_embeddings):
                    if i == self.center_idx:
                        seq_list.append(center_emb)
                    else:
                        seq_list.append(emb)
                sequence = torch.stack(seq_list, dim=1)  # (1, seq_len, gat_out_dim)
                sequence = self.layer_norm(sequence)
                lstm_out, _ = self.lstm(sequence)
                # Attention weighting (replicate original logic)
                attention_scores = self.center_attention(lstm_out).squeeze(-1)  # (1, seq_len)
                # Boost center
                center_mask = torch.zeros(sequence.size(1), dtype=torch.float, device=device)
                center_mask[self.center_idx] = 1.0
                attention_scores = attention_scores + center_mask * 0.3
                attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
                eeg_features = (lstm_out * attention_weights).sum(dim=1)  # (1, output_dim)

                # Fuse with spec features
                fused = self.fusion(eeg_features, self.spec_features)
                logits = self.classifier(fused)  # (1, num_classes)
                return logits

        wrapper = CenterEEGWrapper(
            gat=self.eeg_gat,
            static_embeddings=static_non_center_embeddings,
            layer_norm_module=layer_norm,
            lstm_module=lstm,
            center_attention_module=center_attention,
            pooling_method=self.model.eeg_encoder.pooling_method,
            fusion_module=self.model.fusion,
            classifier_module=self.model.classifier,
            spec_features_tensor=spec_features,
            center_index=center_idx,
        ).to(self.device)
        
        # Create explainer
        explainer = Explainer(
            model=wrapper,
            algorithm=GNNExplainerAlgo(epochs=epochs),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='raw',
            ),
        )
        
        print(f"  Running GNNExplainer ({epochs} epochs)...")
        
        # Get explanation
        explanation = explainer(
            x=center_window.x.to(self.device),
            edge_index=center_window.edge_index.to(self.device),
        )
        
        return {
            'edge_mask': explanation.edge_mask,
            'node_mask': explanation.node_mask,
            'prediction': prediction.cpu(),
            'predicted_class': predicted_class,
            'target_class': target_class,
            # Also return edge_index for visualization without relying on Data object typing
            'center_edge_index': center_window.edge_index,
        }
    
    def visualize_explanation(
        self,
        explanation: Dict[str, torch.Tensor],
        channels: list,
        output_path: Optional[Path] = None,
        top_k_edges: int = 10,
    ):
        """
        Visualize explanation on a brain topographic map.
        
        Args:
            explanation: Output from explain_sample()
            channels: List of channel names (19 electrodes)
            output_path: Where to save the figure (if None, just show)
            top_k_edges: Number of top edges to highlight
        """
        edge_mask = explanation['edge_mask'].cpu().numpy()
        node_mask = explanation['node_mask'].cpu().numpy()
        edge_index = explanation['center_edge_index'].cpu().numpy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Edge importance map
        ax = axes[0]
        ax.set_title('Edge Importance\n(Connectivity)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Plot edges (connections between electrodes)
        top_k_indices = np.argsort(edge_mask)[-top_k_edges:]
        
        reds_cmap = cm.get_cmap('Reds')
        for idx in range(len(edge_mask)):
            src_idx, tgt_idx = edge_index[:, idx]
            src_ch = channels[src_idx]
            tgt_ch = channels[tgt_idx]
            
            if src_ch not in ELECTRODE_POSITIONS or tgt_ch not in ELECTRODE_POSITIONS:
                continue
            
            src_pos = ELECTRODE_POSITIONS[src_ch]
            tgt_pos = ELECTRODE_POSITIONS[tgt_ch]
            
            importance = edge_mask[idx]
            
            # Highlight top-k edges
            if idx in top_k_indices:
                alpha = 0.8
                linewidth = 3
                color = reds_cmap(importance)
            else:
                alpha = 0.1
                linewidth = 0.5
                color = 'gray'
            
            ax.plot(
                [src_pos[0], tgt_pos[0]],
                [src_pos[1], tgt_pos[1]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                zorder=1
            )
        
        # Plot electrode positions
        for ch in channels:
            if ch in ELECTRODE_POSITIONS:
                pos = ELECTRODE_POSITIONS[ch]
                ax.scatter(pos[0], pos[1], c='black', s=100, zorder=2)
                ax.text(pos[0], pos[1] + 0.05, ch, ha='center', fontsize=8)
        
        # 2. Node feature importance (frequency bands)
        ax = axes[1]
        ax.set_title('Node Feature Importance\n(Frequency Bands)', fontsize=14, fontweight='bold')
        
        # Average importance across nodes for each frequency band
        band_importance = node_mask.mean(axis=0)
        band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        viridis_cmap = cm.get_cmap('viridis')
        colors = viridis_cmap(np.linspace(0.3, 0.9, len(band_names)))
        bars = ax.barh(band_names, band_importance, color=colors)
        ax.set_xlabel('Average Importance')
        ax.grid(axis='x', alpha=0.3)
        
        # 3. Electrode-specific importance (average across bands)
        ax = axes[2]
        ax.set_title('Electrode Importance\n(Avg across bands)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Average importance per electrode
        electrode_importance = node_mask.mean(axis=1)
        
        # Normalize for color mapping
        importance_norm = (electrode_importance - electrode_importance.min()) / (
            electrode_importance.max() - electrode_importance.min() + 1e-8
        )
        
        # Plot electrodes with color indicating importance
        for idx, ch in enumerate(channels):
            if ch in ELECTRODE_POSITIONS:
                pos = ELECTRODE_POSITIONS[ch]
                importance = importance_norm[idx]
                
                ax.scatter(
                    pos[0], pos[1],
                    c=[reds_cmap(importance)],
                    s=500,
                    edgecolors='black',
                    linewidths=2,
                    zorder=2
                )
                ax.text(pos[0], pos[1], ch, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Add colorbar
        sm = ScalarMappable(cmap=reds_cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Importance', rotation=270, labelpad=15)
        
        # Overall title
        pred_class = explanation['predicted_class']
        target_class = explanation['target_class']
        pred_prob = explanation['prediction'][pred_class]
        
        fig.suptitle(
            f"GNN Explanation - Predicted: {CLASS_NAMES[pred_class]} ({pred_prob:.2%}), "
            f"Explaining: {CLASS_NAMES[target_class]}",
            fontsize=16,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved explanation to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def explain_multiple_samples(
        self,
        datamodule: HMSDataModule,
        num_samples: int = 5,
        output_dir: Optional[Path] = None,
    ):
        """
        Explain multiple samples and save visualizations.
        
        Args:
            datamodule: DataModule for loading samples
            num_samples: Number of samples to explain
            output_dir: Directory to save explanations
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get validation dataloader
        val_loader = datamodule.val_dataloader()
        
        print(f"\n{'='*70}")
        print(f"Explaining {num_samples} samples from validation set")
        print(f"{'='*70}")
        
        sample_count = 0
        for batch_idx, batch in enumerate(val_loader):
            eeg_graphs = batch['eeg_graphs']
            spec_graphs = batch['spec_graphs']
            
            # Each element in eeg_graphs/spec_graphs is already a Batch
            # We need to extract individual graphs from the batch
            batch_size = eeg_graphs[0].num_graphs
            
            for sample_idx in range(batch_size):
                if sample_count >= num_samples:
                    return
                
                # Extract single sample from each window's batch
                # Use .get_example(idx) to extract individual graphs from Batch
                eeg_sample = []
                for window_batch in eeg_graphs:
                    # Get individual graph from batch
                    single_graph = window_batch.get_example(sample_idx)
                    eeg_sample.append(single_graph)
                
                spec_sample = []
                for window_batch in spec_graphs:
                    single_graph = window_batch.get_example(sample_idx)
                    spec_sample.append(single_graph)
                
                print(f"\n--- Sample {sample_count + 1}/{num_samples} ---")
                
                # Explain
                explanation = self.explain_sample(eeg_sample, spec_sample)
                
                # Visualize
                output_path = None
                if output_dir:
                    output_path = output_dir / f"explanation_sample_{sample_count:03d}.png"
                
                # Channel list for visualization (use standard 10-20 montage fallback)
                channel_list = list(ELECTRODE_POSITIONS.keys())[:19]

                self.visualize_explanation(
                    explanation,
                    channels=channel_list,
                    output_path=output_path,
                )
                
                sample_count += 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Explain GNN predictions using GNNExplainer'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--train_csv',
        type=str,
        default='data/raw/train_unique.csv',
        help='Path to training CSV'
    )
    parser.add_argument(
        '--sample_idx',
        type=int,
        default=None,
        help='Specific sample index to explain (if None, explains multiple samples)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of samples to explain if sample_idx not specified'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/explanations',
        help='Directory to save explanations'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='Which fold to use for validation data'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of epochs for GNNExplainer optimization'
    )
    
    args = parser.parse_args()
    
    # Create explainer
    explainer = EEGGNNExplainer(
        checkpoint_path=args.checkpoint,
    )
    
    # Create datamodule
    print("\nLoading data...")
    datamodule = HMSDataModule(
        data_dir=args.data_dir,
        train_csv=args.train_csv,
        batch_size=1,  # Process one at a time for explanation
        n_folds=5,
        current_fold=args.fold,
        num_workers=0,  # No multiprocessing for explanation
    )
    datamodule.setup()
    
    # Explain samples
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if args.sample_idx is not None:
        # Explain specific sample
        print(f"\nExplaining sample {args.sample_idx}...")
        val_loader = datamodule.val_dataloader()
        
        # Find the sample
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx == args.sample_idx:
                eeg_graphs = batch['eeg_graphs']
                spec_graphs = batch['spec_graphs']
                
                # Extract the single graph from each temporal window's Batch
                eeg_sample = [g.get_example(0) for g in eeg_graphs]
                spec_sample = [g.get_example(0) for g in spec_graphs]
                explanation = explainer.explain_sample(
                    eeg_sample,
                    spec_sample,
                    epochs=args.epochs,
                )
                
                output_path = None
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"explanation_sample_{args.sample_idx:03d}.png"
                
                # Use the same fallback for channels
                channel_list = list(ELECTRODE_POSITIONS.keys())[:19]
                explainer.visualize_explanation(
                    explanation,
                    channels=channel_list,
                    output_path=output_path,
                )
                break
    else:
        # Explain multiple samples
        explainer.explain_multiple_samples(
            datamodule=datamodule,
            num_samples=args.num_samples,
            output_dir=output_dir,
        )
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
