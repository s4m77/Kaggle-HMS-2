"""Visualize cross-modal attention weights for the HMS multi-modal GNN."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch


from src.lightning_trainer.graph_lightning_module import HMSLightningModule

DEFAULT_EEG_REGIONS = ["Frontal", "Central", "Parietal", "Occipital"]
DEFAULT_SPEC_REGIONS = ["Left Frontal", "Right Frontal", "Left Parietal", "Right Parietal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate regional attention heatmaps for a saved HMSMultiModalGNN checkpoint."
    )
    parser.add_argument("--model_path", required=True, help="Path to the Lightning checkpoint (model.ckpt).")
    parser.add_argument("--data_path", required=True, help="Path to a serialized data sample (patient_*.pt).")
    parser.add_argument(
        "--sample_id",
        default=None,
        help="Optional sample key inside the data file. Defaults to the first entry.",
    )
    parser.add_argument(
        "--output_dir",
        default="explanations/attention",
        help="Directory where plots and raw weights will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plots interactively instead of just saving them.",
    )
    return parser.parse_args()


def _prepare_batches(graphs: List[Batch], device: torch.device, trim_spec: bool = False) -> List[Batch]:
    batched = []
    for graph in graphs:
        if trim_spec and hasattr(graph, "x"):
            graph.x = graph.x[:, :4]
        batched.append(Batch.from_data_list([graph]).to(device))
    return batched


def load_sample(
    data_path: str,
    device: torch.device,
    sample_id: Optional[str] = None,
) -> Tuple[str, List[Batch], List[Batch], torch.Tensor]:
    path = Path(data_path)
    if path.name.startswith("._"):
        candidate = path.with_name(path.name[2:])
        if candidate.exists():
            print(f"Detected MacOS resource file {path.name}, switching to {candidate.name}.")
            path = candidate
        else:
            raise FileNotFoundError(
                f"{data_path} looks like a MacOS metadata file and the corresponding {candidate.name} was not found."
            )

    try:
        data_dict: Dict[str, Dict] = torch.load(path, map_location="cpu")
    except (pickle.UnpicklingError, RuntimeError) as err:
        if "weights_only" in str(err):
            data_dict = torch.load(path, map_location="cpu", weights_only=False)
        elif "invalid load key" in str(err):
            raise ValueError(
                f"{path} could not be unpickled. Make sure you are pointing to the actual patient file "
                "and not a MacOS metadata stub. Original error: {err}"
            ) from err
        else:
            raise
    if not data_dict:
        raise ValueError(f"No samples found inside {data_path}")

    if sample_id is None:
        sample_key = next(iter(data_dict.keys()))
    else:
        sample_key = sample_id
        if sample_key not in data_dict and isinstance(sample_key, str):
            try:
                alt_key = int(sample_key)
            except ValueError:
                alt_key = None
            if alt_key in data_dict:
                sample_key = alt_key
        if sample_key not in data_dict:
            raise KeyError(f"{sample_id} not found in {path}. Available keys: {list(data_dict.keys())}")

    sample = data_dict[sample_key]
    eeg_graphs = _prepare_batches(sample["eeg_graphs"], device)
    spec_graphs = _prepare_batches(sample["spec_graphs"], device, trim_spec=True)
    target = sample.get("target", torch.tensor([-1]))
    return str(sample_key), eeg_graphs, spec_graphs, target


def _resolve_region_labels(
    cfg: Dict,
    key: str,
    num_regions: int,
    default_names: List[str],
) -> List[str]:
    names = cfg.get("region_names") or []
    if names and len(names) >= num_regions:
        return names[:num_regions]

    if len(default_names) >= num_regions:
        return default_names[:num_regions]

    # Fallback: generic labels
    return [f"{key}_{idx}" for idx in range(num_regions)]


def _tensor_to_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Convert attention tensor to (query, key) matrix for batch size 1."""
    if tensor.dim() == 4:
        # (batch, heads, query, key) -> average over heads
        tensor = tensor.mean(dim=1)
    if tensor.dim() != 3:
        raise ValueError(f"Unexpected attention tensor shape: {tensor.shape}")
    return tensor[0].detach()


def plot_heatmap(
    matrix: torch.Tensor,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    output_path: Path,
    show_plot: bool,
    vlim: Optional[Tuple[float, float]] = None,
    cmap: str = "magma",
    colorbar_label: str = "Attention weight",
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix_np = matrix.cpu().numpy()
    vmin = vlim[0] if vlim else None
    vmax = vlim[1] if vlim else None
    im = ax.imshow(matrix_np, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def save_attention_json(
    matrix: torch.Tensor,
    row_labels: List[str],
    col_labels: List[str],
    output_path: Path,
) -> None:
    data = {
        "rows": row_labels,
        "cols": col_labels,
        "values": matrix.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))


def summarize_attention(
    matrix: torch.Tensor,
    row_labels: List[str],
    col_labels: List[str],
    top_k: int = 3,
) -> Dict:
    """Compute per-region importance and top cross-modal interactions."""
    uniform = 1.0 / matrix.shape[1]

    deviations = matrix - uniform
    source_focus = deviations.abs().sum(dim=1)
    source_focus = source_focus / source_focus.sum().clamp(min=1e-6)

    target_importance = matrix.sum(dim=0)
    target_importance = target_importance / target_importance.sum().clamp(min=1e-6)

    flat = deviations.flatten()
    k = min(top_k, flat.numel())
    top_values, top_indices = torch.topk(flat, k)

    interactions = []
    num_cols = matrix.shape[1]
    for score, idx in zip(top_values, top_indices):
        row_idx = (idx // num_cols).item()
        col_idx = (idx % num_cols).item()
        percent_above = (score / uniform) * 100.0
        interactions.append(
            {
                "pair": f"{row_labels[row_idx]} -> {col_labels[col_idx]}",
                "attention": float(matrix[row_idx, col_idx].item()),
                "above_uniform": float(score.item()),
                "percent_above_uniform": float(percent_above),
            }
        )

    return {
        "source_focus": {row_labels[i]: float(source_focus[i].item()) for i in range(len(row_labels))},
        "target_importance": {col_labels[i]: float(target_importance[i].item()) for i in range(len(col_labels))},
        "top_interactions": interactions,
        "uniform_value": uniform,
    }


def save_summary_json(summary: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lightning_module = HMSLightningModule.load_from_checkpoint(args.model_path)
    model_config = lightning_module.hparams["model_config"]

    lightning_module.to(device)
    lightning_module.eval()

    if not lightning_module.model.use_regional_fusion:
        raise RuntimeError("Attention visualization requires a checkpoint trained with regional fusion enabled.")

    sample_id, eeg_graphs, spec_graphs, target = load_sample(
        data_path=args.data_path,
        device=device,
        sample_id=args.sample_id,
    )

    with torch.no_grad():
        logits = lightning_module(eeg_graphs, spec_graphs)
        pred_class = logits.argmax(dim=-1).item()
        attention_maps = lightning_module.model.fusion.get_last_attention()

    if not attention_maps:
        raise RuntimeError("No attention weights were captured from the fusion layer.")

    eeg_cfg = model_config.get("eeg_encoder", {})
    spec_cfg = model_config.get("spec_encoder", {})
    num_eeg_regions = eeg_cfg.get("num_regions", len(DEFAULT_EEG_REGIONS))
    num_spec_regions = spec_cfg.get("num_regions", len(DEFAULT_SPEC_REGIONS))

    eeg_labels = _resolve_region_labels(eeg_cfg, "EEG", num_eeg_regions, DEFAULT_EEG_REGIONS)
    spec_labels = _resolve_region_labels(spec_cfg, "SPEC", num_spec_regions, DEFAULT_SPEC_REGIONS)

    eeg_to_spec = _tensor_to_matrix(attention_maps["eeg_to_spec"])
    spec_to_eeg = _tensor_to_matrix(attention_maps["spec_to_eeg"])

    output_dir = Path(args.output_dir) / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    eeg_summary = summarize_attention(eeg_to_spec, eeg_labels[: eeg_to_spec.shape[0]], spec_labels[: eeg_to_spec.shape[1]])
    spec_summary = summarize_attention(spec_to_eeg, spec_labels[: spec_to_eeg.shape[0]], eeg_labels[: spec_to_eeg.shape[1]])

    summary = {
        "sample_id": sample_id,
        "target": target.tolist() if torch.is_tensor(target) else target,
        "prediction": pred_class,
        "eeg_to_spec": eeg_summary,
        "spec_to_eeg": spec_summary,
    }
    save_summary_json(summary, output_dir / "attention_summary.json")

    print(f"Sample: {sample_id}")
    print(f"Target label: {target}")
    print(f"Predicted class: {pred_class}")
    print(f"Saving attention visualizations to {output_dir}")

    def _print_summary(title: str, data: Dict) -> None:
        print(f"\n[{title}]")
        top_source = max(data["source_focus"].items(), key=lambda x: x[1])
        top_target = max(data["target_importance"].items(), key=lambda x: x[1])
        print(f"Most selective source region: {top_source[0]} (focus {top_source[1]:.3f})")
        print(f"Most attended target region: {top_target[0]} (share {top_target[1]:.3f})")
        print("Most informative interactions (Δ vs uniform):")
        for interaction in data["top_interactions"]:
            print(
                f"  {interaction['pair']}: {interaction['attention']:.3f} "
                f"(+{interaction['percent_above_uniform']:.1f}% / {interaction['above_uniform']:.3f})"
            )

    _print_summary("EEG → Spectrogram attention", eeg_summary)
    _print_summary("Spectrogram → EEG attention", spec_summary)

    eeg_to_spec_path = output_dir / "eeg_to_spec.png"
    spec_to_eeg_path = output_dir / "spec_to_eeg.png"

    plot_heatmap(
        eeg_to_spec,
        row_labels=eeg_labels[: eeg_to_spec.shape[0]],
        col_labels=spec_labels[: eeg_to_spec.shape[1]],
        title="EEG query → Spectrogram key",
        output_path=eeg_to_spec_path,
        show_plot=args.show,
    )
    plot_heatmap(
        spec_to_eeg,
        row_labels=spec_labels[: spec_to_eeg.shape[0]],
        col_labels=eeg_labels[: spec_to_eeg.shape[1]],
        title="Spectrogram query → EEG key",
        output_path=spec_to_eeg_path,
        show_plot=args.show,
    )

    def _plot_relative(
        matrix: torch.Tensor,
        row_labels: List[str],
        col_labels: List[str],
        title: str,
        filename: str,
    ) -> None:
        uniform = 1.0 / matrix.shape[1]
        relative = (matrix - uniform) / uniform * 100.0  # percent difference vs baseline
        max_abs = float(relative.abs().max().item())
        vlim = (-max_abs, max_abs) if max_abs > 0 else None
        plot_heatmap(
            relative,
            row_labels=row_labels,
            col_labels=col_labels,
            title=title,
            output_path=output_dir / filename,
            show_plot=False,
            vlim=vlim,
            cmap="coolwarm",
            colorbar_label="% vs uniform",
        )

    _plot_relative(
        eeg_to_spec,
        row_labels=eeg_labels[: eeg_to_spec.shape[0]],
        col_labels=spec_labels[: eeg_to_spec.shape[1]],
        title="EEG query → Spectrogram key (Δ% vs uniform)",
        filename="eeg_to_spec_relative.png",
    )
    _plot_relative(
        spec_to_eeg,
        row_labels=spec_labels[: spec_to_eeg.shape[0]],
        col_labels=eeg_labels[: spec_to_eeg.shape[1]],
        title="Spectrogram query → EEG key (Δ% vs uniform)",
        filename="spec_to_eeg_relative.png",
    )

    save_attention_json(eeg_to_spec, eeg_labels, spec_labels, output_dir / "eeg_to_spec.json")
    save_attention_json(spec_to_eeg, spec_labels, eeg_labels, output_dir / "spec_to_eeg.json")


if __name__ == "__main__":
    main()
