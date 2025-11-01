"""
Lightning-compatible explainability wrapper for capturing attention weights during inference.

Provides utilities to extract cross-modal attention explanations during model evaluation.
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
from pytorch_lightning import LightningModule
from torch_geometric.data import Batch

from src.explainability.attention_analysis import (
    AttentionAnalyzer,
    extract_attention_statistics,
    compute_modality_alignment,
)


class ExplainabilityCapture:
    """
    Captures and stores attention weights during model inference.
    
    Usage:
        ```python
        model = HMSLightningModule.load_from_checkpoint(...)
        explainer = ExplainabilityCapture(model, num_heads=8)
        
        with torch.no_grad():
            batch = next(iter(test_loader))
            results = explainer.explain_batch(batch)
        ```
    """
    
    def __init__(
        self,
        model: LightningModule,
        num_heads: int = 8,
    ):
        """
        Parameters
        ----------
        model : LightningModule
            Trained HMS Lightning module
        num_heads : int
            Number of attention heads in fusion module
        """
        self.model = model
        self.num_heads = num_heads
        self.analyzer = AttentionAnalyzer(num_heads=num_heads)
        self.device = next(model.parameters()).device
    
    def explain_batch(
        self,
        batch: Dict,
    ) -> Dict[str, Any]:
        """
        Explain predictions for an entire batch.
        
        Parameters
        ----------
        batch : Dict
            Batch dictionary with 'eeg_graphs', 'spec_graphs', 'targets'
        
        Returns
        -------
        Dict[str, Any]
            Explanation results including:
            - 'logits': Model predictions
            - 'predictions': Predicted class indices
            - 'targets': Ground truth (if available)
            - 'attention_dict': Raw attention weights
            - 'attention_stats': Statistical analysis
            - 'modality_alignment': Cross-modal alignment metrics
            - 'per_sample_explanations': Per-sample detailed explanations
        """
        eeg_graphs = batch['eeg_graphs']
        spec_graphs = batch['spec_graphs']
        targets = batch.get('targets', None)
        
        # Forward pass with attention capture
        with torch.no_grad():
            logits, attention_dict = self._forward_with_attention(eeg_graphs, spec_graphs)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=1)
        
        # Compute global statistics
        attention_stats = extract_attention_statistics(attention_dict, self.num_heads)
        modality_alignment = compute_modality_alignment(attention_dict, self.num_heads)
        
        # Per-sample explanations
        per_sample = []
        batch_size = logits.shape[0]
        
        for sample_idx in range(batch_size):
            sample_explanation = self._explain_single_sample(
                sample_idx,
                logits,
                predictions,
                targets,
                attention_dict,
            )
            per_sample.append(sample_explanation)
        
        return {
            'logits': logits.cpu(),
            'predictions': predictions.cpu(),
            'targets': targets.cpu() if targets is not None else None,
            'attention_dict': attention_dict,
            'attention_stats': attention_stats,
            'modality_alignment': modality_alignment,
            'per_sample_explanations': per_sample,
        }
    
    def explain_single_sample(
        self,
        batch: Dict,
        sample_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Explain prediction for a single sample in a batch.
        
        Parameters
        ----------
        batch : Dict
            Batch dictionary
        sample_idx : int
            Index of sample to explain
        
        Returns
        -------
        Dict[str, Any]
            Single-sample explanation
        """
        with torch.no_grad():
            results = self.explain_batch(batch)
        
        return results['per_sample_explanations'][sample_idx]
    
    def _forward_with_attention(
        self,
        eeg_graphs: List[Batch],
        spec_graphs: List[Batch],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass that captures attention from fusion module.
        
        Parameters
        ----------
        eeg_graphs : List[Batch]
            EEG graph batch
        spec_graphs : List[Batch]
            Spectrogram graph batch
        
        Returns
        -------
        Tuple[torch.Tensor, Dict]
            (logits, attention_dict)
        """
        # Forward through encoders
        eeg_features = self.model.model.eeg_encoder(eeg_graphs, return_sequence=False)
        spec_features = self.model.model.spec_encoder(spec_graphs, return_sequence=False)
        
        # Forward through fusion with attention capture
        fused_features, attention_dict = self.model.model.fusion(
            eeg_features,
            spec_features,
            return_attention=True,
        )
        
        # Forward through classifier
        logits = self.model.model.classifier(fused_features)
        
        return logits, attention_dict
    
    def _explain_single_sample(
        self,
        sample_idx: int,
        logits: torch.Tensor,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor],
        attention_dict: Dict,
    ) -> Dict[str, Any]:
        """
        Generate explanation for single sample.
        
        Parameters
        ----------
        sample_idx : int
            Sample index in batch
        logits : torch.Tensor
            Model logits
        predictions : torch.Tensor
            Predicted classes
        targets : torch.Tensor, optional
            Ground truth classes
        attention_dict : Dict
            Attention weights from fusion module
        
        Returns
        -------
        Dict[str, Any]
            Per-sample explanation
        """
        # Get sample-level attention
        sample_eeg_to_spec = attention_dict['eeg_to_spec'][sample_idx]
        sample_spec_to_eeg = attention_dict['spec_to_eeg'][sample_idx]
        
        # Compute sample-level statistics
        eeg_to_spec_attn_dict = {'eeg_to_spec': sample_eeg_to_spec.unsqueeze(0)}
        spec_to_eeg_attn_dict = {'spec_to_eeg': sample_spec_to_eeg.unsqueeze(0)}
        
        eeg_to_spec_stats = self.analyzer.analyze_attention_weights(
            sample_eeg_to_spec.unsqueeze(0).unsqueeze(0)
        )
        spec_to_eeg_stats = self.analyzer.analyze_attention_weights(
            sample_spec_to_eeg.unsqueeze(0).unsqueeze(0)
        )
        
        # Get top attention patterns
        eeg_to_spec_top = self.analyzer.get_top_attention_pairs(
            sample_eeg_to_spec.unsqueeze(0).unsqueeze(0),
            top_k=5,
        )
        spec_to_eeg_top = self.analyzer.get_top_attention_pairs(
            sample_spec_to_eeg.unsqueeze(0).unsqueeze(0),
            top_k=5,
        )
        
        # Modality agreement for this sample
        sample_agreement = self.analyzer.compare_modality_agreement(
            sample_eeg_to_spec.unsqueeze(0).unsqueeze(0),
            sample_spec_to_eeg.unsqueeze(0).unsqueeze(0),
        )
        
        explanation = {
            'sample_idx': sample_idx,
            'predicted_class': int(predictions[sample_idx]),
            'predicted_logit': float(logits[sample_idx].max()),
            'true_class': int(targets[sample_idx]) if targets is not None else None,
            'correct': (predictions[sample_idx] == targets[sample_idx]).item() if targets is not None else None,
            'attention_stats': {
                'eeg_to_spec': eeg_to_spec_stats,
                'spec_to_eeg': spec_to_eeg_stats,
            },
            'top_attention_pairs': {
                'eeg_to_spec': eeg_to_spec_top,
                'spec_to_eeg': spec_to_eeg_top,
            },
            'modality_agreement': sample_agreement,
            'head_entropies': self.analyzer.compute_attention_entropy(
                sample_eeg_to_spec.unsqueeze(0).unsqueeze(0),
                per_head=True,
            ),
        }
        
        return explanation


def create_explainability_report(
    explanation_results: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Create a human-readable report from explanation results.
    
    Parameters
    ----------
    explanation_results : Dict[str, Any]
        Results from ExplainabilityCapture.explain_batch()
    output_path : str, optional
        Path to save report. If None, only returns string.
    
    Returns
    -------
    str
        Formatted report
    """
    report = []
    report.append("=" * 80)
    report.append("CROSS-MODAL ATTENTION EXPLAINABILITY REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Batch-level statistics
    report.append("BATCH-LEVEL STATISTICS")
    report.append("-" * 40)
    
    stats = explanation_results['attention_stats']
    report.append(f"EEG → Spectrogram Attention:")
    report.append(f"  {stats['eeg_to_spec']}")
    report.append(f"\nSpectrogram → EEG Attention:")
    report.append(f"  {stats['spec_to_eeg']}")
    
    alignment = explanation_results['modality_alignment']
    report.append(f"\nModality Alignment:")
    report.append(f"  Agreement Score: {alignment['modality_agreement']:.4f}")
    report.append(f"  EEG→Spec Entropy: {alignment['eeg_to_spec_entropy']:.4f}")
    report.append(f"  Spec→EEG Entropy: {alignment['spec_to_eeg_entropy']:.4f}")
    report.append("")
    
    # Per-sample analysis
    report.append("PER-SAMPLE ANALYSIS")
    report.append("-" * 40)
    
    predictions = explanation_results['predictions']
    targets = explanation_results['targets']
    per_sample = explanation_results['per_sample_explanations']
    
    for i, sample_expl in enumerate(per_sample):
        report.append(f"\nSample {i}:")
        report.append(f"  Predicted: {sample_expl['predicted_class']} (confidence: {sample_expl['predicted_logit']:.4f})")
        if sample_expl['true_class'] is not None:
            status = "✓ CORRECT" if sample_expl['correct'] else "✗ WRONG"
            report.append(f"  Ground Truth: {sample_expl['true_class']} {status}")
        
        report.append(f"  Modality Agreement: {sample_expl['modality_agreement']:.4f}")
        
        # Top attention pairs
        e2s_top = sample_expl['top_attention_pairs']['eeg_to_spec']
        if e2s_top:
            report.append(f"  Top EEG→Spec attention: {e2s_top[0]}")
        
        s2e_top = sample_expl['top_attention_pairs']['spec_to_eeg']
        if s2e_top:
            report.append(f"  Top Spec→EEG attention: {s2e_top[0]}")
    
    report_str = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_str)
    
    return report_str


__all__ = [
    'ExplainabilityCapture',
    'create_explainability_report',
]
