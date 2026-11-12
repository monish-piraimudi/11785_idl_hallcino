"""Baseline truthfulness probe, after Azaria & Mitchell (2023): a small MLP
(256->128->64->1, ReLU, sigmoid) over intermediate hidden-state features.

`feature_dim` is the one parameter to change once the real J-lens feature
size is known; everything else (num_layers, fusion strategy, hidden sizes)
is a constructor argument so it stays swappable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

FUSION_MODES = ("concat", "scalar_mix")


class LayerFusion(nn.Module):
    """Combines per-layer feature vectors of shape (batch, feature_dim, num_layers)
    into a single vector for the MLP head.

    "concat": flattens all layers, so input width grows with num_layers.
    "scalar_mix": ELMo-style learned softmax-weighted sum (Tenney et al.), so
    input width stays feature_dim regardless of num_layers, and the learned
    per-layer weights are a rough signal of which layers the probe relies on.
    """

    def __init__(self, feature_dim: int, num_layers: int, mode: str = "scalar_mix"):
        super().__init__()
        if mode not in FUSION_MODES:
            raise ValueError(f"unknown fusion mode: {mode!r}, expected one of {FUSION_MODES}")
        self.mode = mode
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        if mode == "scalar_mix":
            self.layer_logits = nn.Parameter(torch.zeros(num_layers))
            self.scale = nn.Parameter(torch.ones(1))

    @property
    def output_dim(self) -> int:
        return self.feature_dim * self.num_layers if self.mode == "concat" else self.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            return x.reshape(x.shape[0], -1)
        weights = torch.softmax(self.layer_logits, dim=0)
        return self.scale * torch.einsum("bfl,l->bf", x, weights)

    def layer_weights(self) -> torch.Tensor:
        """Normalized per-layer weights; only meaningful in scalar_mix mode."""
        if self.mode != "scalar_mix":
            raise RuntimeError("layer_weights() only applies to scalar_mix fusion")
        return torch.softmax(self.layer_logits, dim=0).detach()


class MLPProbe(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_layers: int = 3,
        fusion: str = "scalar_mix",
        hidden_dims: tuple[int, ...] = (256, 128, 64),
    ):
        super().__init__()
        self.fusion = LayerFusion(feature_dim, num_layers, mode=fusion)
        dims = [self.fusion.output_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits, shape (batch,). Use predict_proba for the sigmoid
        output; training uses BCEWithLogitsLoss directly on these logits for
        numerical stability."""
        return self.head(self.fusion(x)).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
