"""Synthetic stand-in for J-lens features, for developing the probe before
the real (n, m) feature shape is settled.

Mirrors the real collection shape: one prompt produces a variable-length
generated token sequence, each token has a j_transport vector per sampled
layer, and every token in a given sequence inherits that sequence's single
ground-truth label. There is no token-level judge anywhere in this repo (see
docs/RESEARCH_NOTES.md) -- only a whole-answer verdict is available -- so the
true/false signal can only be assigned per prompt, then broadcast to every
token row generated for that prompt, e.g.:

    "What's the capital of France?" -> "It is Paris"   (truthful)
      It          -> label 0
      It is       -> label 0
      It is Paris -> label 0

    "What's the capital of France?" -> "It is Tokyo"   (hallucinated)
      It          -> label 1
      It is       -> label 1
      It is Tokyo -> label 1
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MockDataset:
    features: torch.Tensor  # (N, feature_dim, num_layers): one row per generated token
    labels: torch.Tensor  # (N,): ground truth broadcast from the row's prompt
    prompt_ids: torch.Tensor  # (N,): which prompt each row came from
    token_positions: torch.Tensor  # (N,): position of the token within its prompt's sequence


def generate_mock_dataset(
    n_prompts: int,
    feature_dim: int,
    num_layers: int = 3,
    min_tokens: int = 3,
    max_tokens: int = 8,
    hallucination_rate: float = 0.5,
    noise_std: float = 0.5,
    seed: int = 0,
) -> MockDataset:
    """Each prompt gets a random generated length and a single hallucination
    label. A shared random direction in feature space carries that label into
    every token's feature vector, scaled by a depth-dependent layer strength
    (deeper layers carry a cleaner signal) and a position-dependent token
    strength (the signal gets easier to detect later in the generated
    sequence, since that's typically where a wrong token actually lands) --
    then buried in Gaussian noise.
    """
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(feature_dim, generator=generator)
    direction /= direction.norm()
    layer_strength = torch.linspace(0.4, 1.0, num_layers)

    prompt_labels = (torch.rand(n_prompts, generator=generator) < hallucination_rate).float()
    lengths = torch.randint(min_tokens, max_tokens + 1, (n_prompts,), generator=generator)

    features_by_prompt = []
    labels_by_prompt = []
    prompt_ids_by_prompt = []
    positions_by_prompt = []
    for prompt_id in range(n_prompts):
        length = int(lengths[prompt_id])
        label = prompt_labels[prompt_id]
        sign = 2.0 * label - 1.0  # -1 truthful, +1 hallucinated
        token_strength = torch.linspace(0.3, 1.0, length)

        signal = (
            sign * direction[None, :, None] * layer_strength[None, None, :] * token_strength[:, None, None]
        )
        noise = torch.randn(length, feature_dim, num_layers, generator=generator) * noise_std

        features_by_prompt.append(signal + noise)
        labels_by_prompt.append(label.expand(length))
        prompt_ids_by_prompt.append(torch.full((length,), prompt_id, dtype=torch.long))
        positions_by_prompt.append(torch.arange(length))

    return MockDataset(
        features=torch.cat(features_by_prompt, dim=0),
        labels=torch.cat(labels_by_prompt, dim=0),
        prompt_ids=torch.cat(prompt_ids_by_prompt, dim=0),
        token_positions=torch.cat(positions_by_prompt, dim=0),
    )
