"""Per-answer-weighted loss, matching the training objective documented in
hallucinot/docs/CLASSIFIER_DESIGN.md:

    L = (1/N) sum_i (1/T_i) sum_t BCEWithLogits(s_i,t, Y_i)

Every answer gets equal total weight regardless of how many tokens it
generated. A flat mean over all rows (plain BCEWithLogitsLoss on the
concatenated batch) would overweight longer answers, since they contribute
more rows.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def per_answer_bce_loss(logits: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    """Mean-of-means: average BCE within each group (answer), then average
    those group means across groups."""
    per_row = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
    sums = torch.zeros(unique_ids.numel(), dtype=per_row.dtype)
    sums.index_add_(0, inverse, per_row)
    counts = torch.zeros(unique_ids.numel(), dtype=per_row.dtype)
    counts.index_add_(0, inverse, torch.ones_like(per_row))
    return (sums / counts).mean()
