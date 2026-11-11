"""Classifier-level evaluation for hallucination detectors (numpy only; plots need matplotlib).

Label convention: 1 = hallucination, 0 = non-hallucination. Scores are any real-valued
hallucination score where higher means "more likely hallucination" (a probability, a logit,
or a discrete count such as a lexicon-baseline token count).
"""

from .metrics import (
    auroc,
    average_precision,
    classification_report,
    evaluate_classifier,
    select_threshold,
)
from .selection import important_layers, layer_auroc_curve, topk_curve

__all__ = [
    "auroc",
    "average_precision",
    "classification_report",
    "evaluate_classifier",
    "select_threshold",
    "important_layers",
    "layer_auroc_curve",
    "topk_curve",
]
