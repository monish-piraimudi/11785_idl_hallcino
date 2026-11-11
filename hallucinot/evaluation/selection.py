"""Sections 1.4-1.5: per-layer AUROC / important layers, and the top-k sensitivity sweep.

Both are model-selection tools, so they only ever see validation rows. Callers pass the
validation split; the CLI filters to it and refuses to run on test rows.

Optional question-level bootstrap bands (``n_boot > 0``) use one resampler for every layer
and k, so all points of a curve are resampled with the same questions (paired).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from .bootstrap import QuestionResampler, resampler_for
from .metrics import auroc

DEFAULT_LAYER_MARGIN = 0.2
DEFAULT_PLATEAU_TOLERANCE = 0.005


class _Group:
    def __init__(self) -> None:
        self.labels: list[int] = []
        self.scores: list[float] = []
        self.questions: list[str | None] = []

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.labels, dtype=np.int64), np.asarray(self.scores, dtype=np.float64)


def _grouped(rows: Iterable[Mapping[str, Any]], keys: tuple[str, ...]) -> dict[tuple, _Group]:
    groups: dict[tuple, _Group] = defaultdict(_Group)
    for row in rows:
        group = groups[tuple(row[key] for key in keys)]
        group.labels.append(row["label"])
        group.scores.append(row["score"])
        group.questions.append(row.get("question_id"))
    return groups


def _shared_resampler(groups: Mapping[tuple, _Group], n_boot: int, seed: int) -> tuple[QuestionResampler, dict[tuple, np.ndarray]]:
    """One resampler over every group's questions; per-group question codes into it.

    Without question ids on every row, each example is its own resampling unit.
    """
    flat = [(key, question) for key, group in groups.items() for question in group.questions]
    resampler, ids = resampler_for([question for _, question in flat], n_boot, seed)
    codes: dict[tuple, list[str]] = defaultdict(list)
    for (key, _), identifier in zip(flat, ids):
        codes[key].append(identifier)
    return resampler, {key: resampler.codes(value) for key, value in codes.items()}


def _auroc_or_none(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if labels.size == 0 or labels.min() == labels.max():
        return None
    return auroc(labels, scores)


def _non_hallucination_questions(group: _Group) -> int | None:
    if any(question is None for question in group.questions):
        return None
    return len({question for question, label in zip(group.questions, group.labels) if label == 0})


def layer_auroc_curve(
    rows: Iterable[Mapping[str, Any]],
    *,
    margin: float = DEFAULT_LAYER_MARGIN,
    n_boot: int = 0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """AUROC of a one-layer probe per (config, layer), with Eq. (2)'s importance flag.

    A layer is important iff |AUROC_l - 0.5| > margin (strict; default margin 0.2), so a probe
    that is reliably *anti*-correlated also counts as carrying signal. The flag uses the point
    estimate, as Eq. (2) defines it; ``ci_low``/``ci_high`` show how certain that call is.
    """
    groups = _grouped(rows, ("config", "layer"))
    resampler, codes = _shared_resampler(groups, n_boot, seed) if n_boot else (None, {})
    out = []
    for key, group in sorted(groups.items()):
        config, layer = key
        labels, scores = group.arrays()
        value = auroc(labels, scores)
        row = {
            "config": config,
            "layer": layer,
            "n": int(labels.size),
            "n_hallucination": int(labels.sum()),
            "n_questions_with_non_hallucination": _non_hallucination_questions(group),
            "auroc": value,
            "deviation_from_chance": abs(value - 0.5),
            "important": abs(value - 0.5) > margin,
        }
        if resampler is not None:
            group_codes = codes[key]

            def statistic(counts: np.ndarray, labels=labels, scores=scores, group_codes=group_codes):
                index = resampler.expand(group_codes, counts)
                result = _auroc_or_none(labels[index], scores[index])
                return None if result is None else {"auroc": result}

            boot = resampler.intervals(statistic)
            low, high = boot["ci"].get("auroc", [float("nan"), float("nan")])
            row.update({"ci_low": low, "ci_high": high, "bootstrap_unit": boot["unit"], "bootstrap_valid": boot["n_valid"]})
        out.append(row)
    return out


def important_layers(curve: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per config: the peak layer (highest AUROC), the important set I, and |I|."""
    summary: dict[str, dict[str, Any]] = {}
    for config in dict.fromkeys(row["config"] for row in curve):
        points = [row for row in curve if row["config"] == config]
        peak = max(points, key=lambda row: (row["auroc"], -row["layer"]))
        important = [row["layer"] for row in points if row["important"]]
        summary[config] = {
            "peak_layer": peak["layer"],
            "peak_auroc": peak["auroc"],
            "peak_ci": [peak["ci_low"], peak["ci_high"]] if "ci_low" in peak else None,
            "important_layers": important,
            "n_important": len(important),
            "n_layers_evaluated": len(points),
            "n_questions_with_non_hallucination": peak["n_questions_with_non_hallucination"],
        }
    return summary


def topk_curve(
    rows: Iterable[Mapping[str, Any]],
    *,
    tolerance: float = DEFAULT_PLATEAU_TOLERANCE,
    n_boot: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Best-layer validation AUROC for each k, and the k where the curve levels off.

    For each (config, k) the best layer is the one with the highest validation AUROC. The
    chosen k is the smallest k whose best-layer AUROC is within ``tolerance`` of the maximum
    over the sweep: an explicit version of "where the curve levels off" that does not assume
    the curve is monotone. Inspect the plotted curve before freezing k.

    The band for each k re-picks the best layer inside every bootstrap replicate, so it
    reflects the noise in the selection as well as in the AUROC.
    """
    groups = _grouped(rows, ("config", "k", "layer"))
    resampler, codes = _shared_resampler(groups, n_boot, seed) if n_boot else (None, {})
    per_layer = []
    for (config, k, layer), group in sorted(groups.items()):
        labels, scores = group.arrays()
        per_layer.append({"config": config, "k": k, "layer": layer, "n": int(labels.size), "auroc": auroc(labels, scores)})

    curve = []
    for config, k in dict.fromkeys((row["config"], row["k"]) for row in per_layer):
        candidates = [row for row in per_layer if row["config"] == config and row["k"] == k]
        best = max(candidates, key=lambda row: (row["auroc"], -row["layer"]))
        point = {
            "config": config,
            "k": k,
            "best_layer": best["layer"],
            "best_layer_auroc": best["auroc"],
            "n_layers": len(candidates),
        }
        if resampler is not None:
            members = [(groups[(config, k, row["layer"])].arrays(), codes[(config, k, row["layer"])]) for row in candidates]

            def statistic(counts: np.ndarray, members=members):
                values = []
                for (labels, scores), group_codes in members:
                    index = resampler.expand(group_codes, counts)
                    value = _auroc_or_none(labels[index], scores[index])
                    if value is not None:
                        values.append(value)
                return {"best_layer_auroc": max(values)} if values else None

            boot = resampler.intervals(statistic)
            low, high = boot["ci"].get("best_layer_auroc", [float("nan"), float("nan")])
            point.update({"ci_low": low, "ci_high": high, "bootstrap_unit": boot["unit"], "bootstrap_valid": boot["n_valid"]})
        curve.append(point)

    chosen = {}
    for config in dict.fromkeys(row["config"] for row in curve):
        points = sorted((row for row in curve if row["config"] == config), key=lambda row: row["k"])
        top = max(row["best_layer_auroc"] for row in points)
        pick = next(row for row in points if row["best_layer_auroc"] >= top - tolerance)
        chosen[config] = {
            "chosen_k": pick["k"],
            "chosen_k_auroc": pick["best_layer_auroc"],
            "chosen_k_best_layer": pick["best_layer"],
            "max_auroc": top,
            "tolerance": tolerance,
            "monotone_nondecreasing": all(
                b["best_layer_auroc"] >= a["best_layer_auroc"] for a, b in zip(points, points[1:])
            ),
            "best_layer_varies_with_k": len({row["best_layer"] for row in points}) > 1,
        }
    return {"per_layer": per_layer, "curve": curve, "chosen": chosen}
