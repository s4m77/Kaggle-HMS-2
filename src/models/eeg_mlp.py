"""Simple EEG-only MLP baseline model."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from omegaconf import OmegaConf

from torch import nn

EPS = 1e-6


class EEGMLPBaseline(nn.Module):
    """A lightweight MLP operating directly on raw EEG signals."""

    def __init__(self, config: OmegaConf) -> None:
        super().__init__()

        if config is None or not hasattr(config, "model"):
            raise ValueError("Configuration with a 'model' section is required for the EEG MLP baseline.")

        self.cfg = config
        self.model_cfg = config.model

        classifier_cfg = getattr(self.model_cfg, "classifier", None)
        if classifier_cfg is None or not hasattr(classifier_cfg, "layers"):
            raise KeyError("model.classifier.layers must be defined for the EEG MLP baseline.")

        self.layer_dims: List[int] = [int(dim) for dim in classifier_cfg.layers]
        if not self.layer_dims:
            raise ValueError("model.classifier.layers must contain at least one dimension.")

        self.activation_name: str = str(getattr(classifier_cfg, "activation", "relu"))
        self.dropout_p: float = float(getattr(classifier_cfg, "dropout", 0.0))
        self.normalize_input: bool = bool(getattr(self.model_cfg, "normalize", True))

        prep_cfg = getattr(self.model_cfg, "preprocess", None)
        self.pool_kernel: int = int(getattr(prep_cfg, "pool_kernel", 1)) if prep_cfg else 1
        self.pool_stride: int = int(getattr(prep_cfg, "pool_stride", self.pool_kernel)) if prep_cfg else self.pool_kernel
        if self.pool_kernel < 1 or self.pool_stride < 1:
            raise ValueError("Preprocess pool_kernel and pool_stride must be >= 1.")
        self.use_layernorm: bool = bool(getattr(prep_cfg, "layernorm", False)) if prep_cfg else False

        self.avg_pool: Optional[nn.AvgPool1d] = (
            nn.AvgPool1d(kernel_size=self.pool_kernel, stride=self.pool_stride)
            if self.pool_kernel > 1
            else None
        )

        self.mlp: nn.Sequential = self._build_mlp()

        vote_keys = getattr(self.cfg.data, "vote_keys", None)
        if vote_keys:
            expected_outputs = len(vote_keys)
            if self.layer_dims[-1] != expected_outputs:
                raise ValueError(
                    f"Final classifier dimension ({self.layer_dims[-1]}) must match number of vote keys ({expected_outputs})."
                )

    def forward(self, eeg_signal: torch.Tensor, *, return_aux: bool = False) -> Dict[str, torch.Tensor]:
        if eeg_signal is None:
            raise ValueError("eeg_signal must be provided to the EEG MLP baseline.")
        if eeg_signal.ndim not in {3, 4}:
            raise ValueError(
                f"eeg_signal must be a 3D or 4D tensor (batch x channels x time or batch x segments x channels x time), got shape {tuple(eeg_signal.shape)}."
            )

        x = eeg_signal.float().contiguous()
        if x.ndim == 4:
            x = x.mean(dim=1)

        if self.avg_pool is not None:
            x = self.avg_pool(x)

        batch_size, channels, timesteps = x.shape
        flattened = x.view(batch_size, channels * timesteps)

        if self.normalize_input:
            mean = flattened.mean(dim=-1, keepdim=True)
            std = flattened.std(dim=-1, keepdim=True)
            std = torch.where(std < EPS, torch.ones_like(std), std)
            flattened = (flattened - mean) / std

        if self.mlp is None:
            self._build_mlp(flattened.size(-1))

        logits = self.mlp(flattened)

        outputs: Dict[str, torch.Tensor] = {"logits": logits}
        if return_aux:
            outputs["features"] = flattened
        return outputs

    def reset_params(self) -> None:
        for module in self.mlp:
            reset_fn = getattr(module, "reset_parameters", None)
            if callable(reset_fn):
                lazy_mixin = getattr(module, "has_uninitialized_params", None)
                if callable(lazy_mixin) and lazy_mixin():
                    continue
                reset_fn()

    def _build_mlp(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim: Optional[int] = None

        for idx, out_dim in enumerate(self.layer_dims):
            if idx == 0:
                layers.append(nn.LazyLinear(out_dim))
            else:
                assert in_dim is not None
                layers.append(nn.Linear(in_dim, out_dim))
            if idx == 0 and self.use_layernorm:
                layers.append(nn.LayerNorm(out_dim))
            if idx < len(self.layer_dims) - 1:
                layers.append(self._resolve_activation(self.activation_name))
                if self.dropout_p > 0:
                    layers.append(nn.Dropout(self.dropout_p))
            in_dim = out_dim

        return nn.Sequential(*layers)

    @staticmethod
    def _resolve_activation(name: str) -> nn.Module:
        lookup = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
        }
        try:
            return lookup[name.lower()]()
        except KeyError as exc:
            raise ValueError(f"Unsupported activation '{name}'.") from exc


__all__ = ["EEGMLPBaseline"]
