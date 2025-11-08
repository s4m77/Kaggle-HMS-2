"""Practical script demonstrating ZORRO explainer usage in typical workflows."""

import argparse
import torch
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime

from torch_geometric.data import Batch

from src.models import HMSMultiModalGNN, ZORROExplainer
from examples.zorro_explainer_example import print_explanation, compare_modalities


class ZORROWorkflow:
    """Workflow class for end-to-end ZORRO explanation generation."""
    
    def __init__(
        self,
        model_path: Path,
        device: torch.device = torch.device("cpu"),
        output_dir: Optional[Path] = None,
    ):
        """Initialize workflow.
        
        Parameters
        ----------
        model_path : Path
            Path to trained model checkpoint
        device : torch.device
            Device to use for computation
        output_dir : Path, optional
            Directory to save explanations
        """
        self.model_path = Path(model_path)
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path("./zorro_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        self.explainer = ZORROExplainer(
            model=self.model,
            device=device,
            perturbation_mode="zero",
        )
    
    def _load_model(self) -> HMSMultiModalGNN:
        """Load trained model from checkpoint."""
        print(f"Loading model from: {self.model_path}")
        
        model = HMSMultiModalGNN()
        
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            print("✓ Model loaded successfully")
        else:
            print(f"⚠ Model path not found: {self.model_path}")
            print("  Using randomly initialized model")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def explain_batch(
        self,
        eeg_graphs: List[Batch],
        spec_graphs: List[Batch],
        sample_indices: Optional[List[int]] = None,
        top_k: int = 10,
        n_samples: int = 5,
    ) -> Dict[int, Dict[str, any]]:
        """Explain batch of samples.
        
        Parameters
        ----------
        eeg_graphs : List[Batch]
            List of 9 EEG graphs
        spec_graphs : List[Batch]
            List of 119 spectrogram graphs
        sample_indices : List[int], optional
            Indices of samples to explain
        top_k : int
            Number of top nodes to return
        n_samples : int
            Number of perturbation samples
        
        Returns
        -------
        Dict[int, Dict[str, any]]
            Explanations for each sample
        """
        batch_size = eeg_graphs[0].num_graphs
        
        if sample_indices is None:
            sample_indices = list(range(batch_size))
        
        results = {}
        
        for sample_idx in sample_indices:
            print(f"\nExplaining sample {sample_idx}...")
            
            eeg_exp = self.explainer.explain_sample(
                graphs=eeg_graphs,
                modality="eeg",
                sample_idx=sample_idx,
                top_k=top_k,
                n_samples=n_samples,
                pbar=True,
            )
            
            spec_exp = self.explainer.explain_sample(
                graphs=spec_graphs,
                modality="spec",
                sample_idx=sample_idx,
                top_k=top_k,
                n_samples=n_samples,
                pbar=True,
            )
            
            results[sample_idx] = {
                "eeg": eeg_exp,
                "spec": spec_exp,
            }
            
            # Print summary
            print_explanation(eeg_exp, "EEG")
            print_explanation(spec_exp, "Spectrogram")
            compare_modalities(eeg_exp, spec_exp)
        
        return results
    
    def save_explanation(
        self,
        explanation,
        sample_idx: int,
        modality: str,
    ) -> None:
        """Save explanation to file.
        
        Parameters
        ----------
        explanation : ZORROExplanation
            Explanation object
        sample_idx : int
            Sample index
        modality : str
            "eeg" or "spec"
        """
        # Prepare data for JSON serialization
        top_k_nodes = [
            {"node_idx": int(node_idx), "importance": float(importance)}
            for node_idx, importance in explanation.top_k_nodes
        ]
        
        feature_importance = explanation.feature_importance.cpu().numpy().tolist()
        
        data = {
            "sample_idx": sample_idx,
            "modality": modality,
            "timestamp": datetime.now().isoformat(),
            "top_k_nodes": top_k_nodes,
            "feature_importance": feature_importance,
            "node_importance_shape": list(explanation.node_importance.shape),
            "num_nodes": len(explanation.node_indices),
            "predicted_class": int(explanation.prediction_original.argmax()),
            "prediction_logits": explanation.prediction_original.cpu().numpy().tolist(),
        }
        
        # Save to JSON
        output_path = self.output_dir / f"sample_{sample_idx:03d}_{modality}.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved to: {output_path}")
    
    def save_batch_explanations(
        self,
        explanations: Dict[int, Dict[str, any]],
    ) -> None:
        """Save all explanations in batch.
        
        Parameters
        ----------
        explanations : Dict
            Explanations from explain_batch()
        """
        for sample_idx, modality_exps in explanations.items():
            for modality, exp in modality_exps.items():
                self.save_explanation(exp, sample_idx, modality)
        
        print(f"\n✓ All explanations saved to: {self.output_dir}")
    
    def generate_report(
        self,
        explanations: Dict[int, Dict[str, any]],
    ) -> str:
        """Generate text report of explanations.
        
        Parameters
        ----------
        explanations : Dict
            Explanations from explain_batch()
        
        Returns
        -------
        str
            Formatted report text
        """
        report_lines = [
            "=" * 70,
            "ZORRO EXPLAINER REPORT",
            "=" * 70,
            f"Generated: {datetime.now().isoformat()}",
            f"Model: {self.model_path.name}",
            f"Device: {self.device}",
            "",
        ]
        
        for sample_idx, modality_exps in explanations.items():
            report_lines.extend([
                f"\nSample {sample_idx}:",
                "-" * 50,
            ])
            
            eeg_exp = modality_exps["eeg"]
            spec_exp = modality_exps["spec"]
            
            report_lines.extend([
                "\nEEG:",
                f"  Predicted class: {eeg_exp.prediction_original.argmax().item()}",
                f"  Total node importance: {eeg_exp.node_importance.sum().item():.4f}",
                "  Top-5 nodes:",
            ])
            
            for rank, (node_idx, importance) in enumerate(eeg_exp.top_k_nodes[:5], 1):
                report_lines.append(f"    {rank}. Node {node_idx}: {importance:.4f}")
            
            report_lines.extend([
                "\nSpectrogram:",
                f"  Predicted class: {spec_exp.prediction_original.argmax().item()}",
                f"  Total node importance: {spec_exp.node_importance.sum().item():.4f}",
                "  Top-5 nodes:",
            ])
            
            for rank, (node_idx, importance) in enumerate(spec_exp.top_k_nodes[:5], 1):
                report_lines.append(f"    {rank}. Node {node_idx}: {importance:.4f}")
        
        report_lines.extend([
            "\n" + "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / "explanation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"✓ Report saved to: {report_path}")
        
        return report_text


def main():
    """Example usage as command-line script."""
    parser = argparse.ArgumentParser(
        description="ZORRO explainer workflow for HMS model"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default="./checkpoints/best_model.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./zorro_outputs",
        help="Directory to save explanations",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top nodes to return",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of perturbation samples",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    # Initialize workflow
    device = torch.device(args.device)
    workflow = ZORROWorkflow(
        model_path=args.model_path,
        device=device,
        output_dir=args.output_dir,
    )
    
    print("\nZORRO Explainer Workflow Initialized")
    print(f"Model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    print("\nTo use in your code:")
    print("  workflow = ZORROWorkflow(model_path, device, output_dir)")
    print("  explanations = workflow.explain_batch(eeg_graphs, spec_graphs)")
    print("  workflow.save_batch_explanations(explanations)")


if __name__ == "__main__":
    main()
