"""
Example script demonstrating cross-modal attention explainability.

This script shows how to:
1. Load a trained model
2. Extract attention weights from test data
3. Visualize cross-modal attention patterns
4. Generate explainability reports
5. Analyze per-head specialization

Usage:
    python src/explainability/example_analysis.py \
        --checkpoint checkpoints/hms-model.ckpt \
        --output_dir explainability_results/
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import torch
import matplotlib.pyplot as plt

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.lightning_trainer import HMSLightningModule
from src.explainability import (
    ExplainabilityCapture,
    create_explainability_report,
)
from src.explainability.visualizations import (
    plot_cross_modal_attention,
    plot_attention_heatmap,
    plot_head_contributions,
    plot_modality_alignment,
)


def load_model(checkpoint_path: str) -> HMSLightningModule:
    """Load trained Lightning model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = HMSLightningModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    print("✓ Model loaded successfully")
    return model


def analyze_batch(
    model: HMSLightningModule,
    batch: dict,
    output_dir: Path,
    batch_name: str = "batch",
) -> dict:
    """
    Analyze a batch of samples and generate visualizations.
    
    Parameters
    ----------
    model : HMSLightningModule
        Trained model
    batch : dict
        Batch with 'eeg_graphs', 'spec_graphs', and optionally 'targets'
    output_dir : Path
        Directory to save outputs
    batch_name : str
        Prefix for output files
    
    Returns
    -------
    dict
        Explanation results
    """
    print(f"\nAnalyzing {batch_name}...")
    
    # Create explainability capture
    explainer = ExplainabilityCapture(model, num_heads=8)
    
    # Explain batch
    print("  Extracting attention weights...")
    results = explainer.explain_batch(batch)
    
    # Generate text report
    print("  Creating report...")
    report = create_explainability_report(results)
    report_path = output_dir / f"{batch_name}_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  ✓ Report saved to {report_path}")
    
    # Generate visualizations
    print("  Creating visualizations...")
    attention_dict = results['attention_dict']
    
    # 1. Cross-modal attention heatmaps
    for sample_idx in range(min(2, attention_dict['eeg_to_spec'].shape[0])):
        fig = plot_cross_modal_attention(
            attention_dict,
            sample_idx=sample_idx,
            figsize=(15, 5),
        )
        fig_path = output_dir / f"{batch_name}_sample_{sample_idx}_cross_attention.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved {fig_path}")
    
    # 2. Per-head contributions
    fig = plot_head_contributions(attention_dict, figsize=(12, 5))
    fig_path = output_dir / f"{batch_name}_head_contributions.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved {fig_path}")
    
    # 3. Modality alignment
    for sample_idx in range(min(2, attention_dict['eeg_to_spec'].shape[0])):
        fig = plot_modality_alignment(
            attention_dict,
            sample_idx=sample_idx,
            figsize=(14, 6),
        )
        fig_path = output_dir / f"{batch_name}_sample_{sample_idx}_alignment.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved {fig_path}")
    
    print(f"✓ Analysis complete for {batch_name}")
    return results


def print_summary(results: dict) -> None:
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("EXPLAINABILITY SUMMARY")
    print("=" * 80)
    
    alignment = results['modality_alignment']
    print(f"\nModality Alignment Score: {alignment['modality_agreement']:.4f}")
    print(f"  (Range: [-1, 1]. Higher = better agreement between modalities)")
    
    print(f"\nEEG → Spectrogram Entropy: {alignment['eeg_to_spec_entropy']:.4f}")
    print(f"  (Higher = more diffuse attention, Lower = more focused)")
    
    print(f"\nSpectrogram → EEG Entropy: {alignment['spec_to_eeg_entropy']:.4f}")
    
    stats = results['attention_stats']
    print(f"\nEEG → Spectrogram Attention Statistics:")
    print(f"  Mean: {stats['eeg_to_spec'].mean:.4f}")
    print(f"  Max: {stats['eeg_to_spec'].max:.4f}")
    print(f"  Sparsity: {stats['eeg_to_spec'].sparsity:.2f}%")
    
    print(f"\nSpectrogram → EEG Attention Statistics:")
    print(f"  Mean: {stats['spec_to_eeg'].mean:.4f}")
    print(f"  Max: {stats['spec_to_eeg'].max:.4f}")
    print(f"  Sparsity: {stats['spec_to_eeg'].sparsity:.2f}%")
    
    predictions = results['predictions']
    targets = results['targets']
    if targets is not None:
        accuracy = (predictions == targets).float().mean()
        print(f"\nBatch Accuracy: {accuracy:.4f}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cross-modal attention explainability"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="explainability_results",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to analyze",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint)
    device = torch.device(args.device)
    model = model.to(device)
    
    # Note: In a real scenario, you would load actual test data
    print("\n" + "=" * 80)
    print("NOTE: This is a demonstration script.")
    print("To use with real data, replace the dummy batch creation with actual data loading.")
    print("=" * 80)
    
    # Create dummy batch for demonstration
    print("\nCreating dummy batch for demonstration...")
    from torch_geometric.data import Batch, Data
    
    batch_size = 2
    dummy_eeg_graphs = [
        Batch.from_data_list([
            Data(
                x=torch.randn(19, 5),  # 19 EEG channels, 5 band features
                edge_index=torch.randint(0, 19, (2, 50)),
                edge_attr=torch.rand(50, 1),
            )
            for _ in range(batch_size)
        ])
        for _ in range(9)  # 9 temporal windows
    ]
    
    dummy_spec_graphs = [
        Batch.from_data_list([
            Data(
                x=torch.randn(4, 5),  # 4 regions, 5 band features
                edge_index=torch.tensor([[0, 1], [1, 0]]).t().contiguous(),
            )
            for _ in range(batch_size)
        ])
        for _ in range(119)  # 119 temporal windows
    ]
    
    dummy_batch = {
        'eeg_graphs': dummy_eeg_graphs,
        'spec_graphs': dummy_spec_graphs,
        'targets': torch.randint(0, 6, (batch_size,)),
    }
    
    # Move to device
    for key in ['eeg_graphs', 'spec_graphs']:
        dummy_batch[key] = [
            g.to(device) if hasattr(g, 'to') else g
            for g in dummy_batch[key]
        ]
    dummy_batch['targets'] = dummy_batch['targets'].to(device)
    
    # Analyze batch
    with torch.no_grad():
        results = analyze_batch(
            model,
            dummy_batch,
            output_dir,
            batch_name="demonstration",
        )
    
    # Print summary
    print_summary(results)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
