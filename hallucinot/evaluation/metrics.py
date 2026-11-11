"""Sections 1.1-1.3: AUROC, per-class AUPRC, and F1 at a validation-frozen threshold.

Every function takes ``labels`` (0/1, 1 = hallucination) and ``scores`` (higher = more likely
hallucination). The decision rule everywhere is: predict hallucination iff ``score >= tau``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

THRESHOLD_RULES = ("macro-f1", "wrongful-stop")


def _validate(labels: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or y.shape != s.shape:
        raise ValueError(f"labels and scores must be 1-D and the same length; got {y.shape} and {s.shape}")
    if y.size == 0:
        raise ValueError("no examples to evaluate")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (non-hallucination) or 1 (hallucination)")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or infinity")
    return y.astype(np.int64), s


def _require_both_classes(y: np.ndarray, what: str) -> tuple[int, int]:
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"{what} needs both classes; got {n_pos} hallucination and {n_neg} non-hallucination examples")
    return n_pos, n_neg


def _midranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with tied values sharing the average of their ranks."""
    order = np.argsort(x, kind="mergesort")
    ordered = x[order]
    starts_group = np.empty(x.size, dtype=bool)
    starts_group[0] = True
    starts_group[1:] = ordered[1:] != ordered[:-1]
    group = np.cumsum(starts_group) - 1
    starts = np.flatnonzero(starts_group)
    ends = np.append(starts[1:], x.size)
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = ((starts + 1 + ends) / 2.0)[group]
    return ranks


def auroc(labels: Any, scores: Any) -> float:
    """Eq. (1): normalized Mann-Whitney U; a tied (positive, negative) pair counts one half.

    Computed in O(n log n) from mid-ranks: U = R_+ - n_+(n_+ + 1)/2, AUROC = U / (n_+ n_-).
    """
    y, s = _validate(labels, scores)
    n_pos, n_neg = _require_both_classes(y, "AUROC")
    rank_sum = _midranks(s)[y == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: Any, scores: Any, *, positive: int = 1) -> float:
    """Area under the precision-recall curve as step-wise average precision.

    AP = sum_n (R_n - R_{n-1}) P_n over distinct score thresholds (tied scores enter together),
    the same estimator as scikit-learn's ``average_precision_score``. With ``positive=0`` the
    non-hallucination class is treated as positive and ranked by ``-score``.
    """
    y, s = _validate(labels, scores)
    if positive not in (0, 1):
        raise ValueError("positive must be 0 or 1")
    if positive == 0:
        y, s = 1 - y, -s
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError(f"AUPRC for class {positive} needs at least one example of that class")
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    last_of_group = np.flatnonzero(np.append(s_sorted[1:] != s_sorted[:-1], True))
    tp = np.cumsum(y_sorted)[last_of_group]
    predicted = last_of_group + 1
    precision = tp / predicted
    recall = tp / n_pos
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_report(labels: Any, scores: Any, tau: float) -> dict[str, float | int]:
    """Confusion counts, per-class precision/recall/F1, macro-F1 and class balance at ``tau``.

    F1 for a class with no true and no predicted members is reported as 0 (it is undefined).
    ``wrongful_stop_rate`` is the fraction of non-hallucination examples flagged as
    hallucination (the false-positive rate): stopping generation there would be wrong.
    """
    y, s = _validate(labels, scores)
    predicted = s >= tau
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    f1_hall = _safe_div(2 * tp, 2 * tp + fp + fn)
    f1_non = _safe_div(2 * tn, 2 * tn + fn + fp)
    n = int(y.size)
    return {
        "n": n,
        "n_hallucination": tp + fn,
        "n_non_hallucination": tn + fp,
        "prevalence_hallucination": (tp + fn) / n,
        "tau": float(tau),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_hallucination": _safe_div(tp, tp + fp),
        "recall_hallucination": _safe_div(tp, tp + fn),
        "f1_hallucination": f1_hall,
        "precision_non_hallucination": _safe_div(tn, tn + fn),
        "recall_non_hallucination": _safe_div(tn, tn + fp),
        "f1_non_hallucination": f1_non,
        "macro_f1": (f1_hall + f1_non) / 2.0,
        "wrongful_stop_rate": _safe_div(fp, fp + tn),
        "accuracy": (tp + tn) / n,
    }


def _threshold_sweep(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every distinct decision ``score >= tau``, from predicting nothing (tau=+inf) downwards."""
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    last_of_group = np.flatnonzero(np.append(s_sorted[1:] != s_sorted[:-1], True))
    tp = np.concatenate(([0], np.cumsum(y_sorted)[last_of_group]))
    fp = np.concatenate(([0], last_of_group + 1)) - tp
    taus = np.concatenate(([math.inf], s_sorted[last_of_group]))
    return taus, tp.astype(np.float64), fp.astype(np.float64)


def select_threshold(
    labels: Any,
    scores: Any,
    *,
    rule: str = "macro-f1",
    max_wrongful_stop: float | None = None,
) -> dict[str, Any]:
    """Choose tau on VALIDATION data only; the caller freezes it for the test split.

    ``macro-f1``: the tau maximizing validation macro-F1 (ties -> the highest tau).
    ``wrongful-stop``: the lowest tau whose validation wrongful-stop rate (false-positive rate)
    is at most ``max_wrongful_stop``, i.e. the most hallucinations caught within that budget.
    """
    y, s = _validate(labels, scores)
    n_pos, n_neg = _require_both_classes(y, "threshold selection")
    taus, tp, fp = _threshold_sweep(y, s)
    fn, tn = n_pos - tp, n_neg - fp
    if rule == "macro-f1":
        f1_hall = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
        f1_non = np.divide(2 * tn, 2 * tn + fn + fp, out=np.zeros_like(tn), where=(2 * tn + fn + fp) > 0)
        index = int(np.argmax((f1_hall + f1_non) / 2.0))
    elif rule == "wrongful-stop":
        if max_wrongful_stop is None or not 0.0 <= max_wrongful_stop <= 1.0:
            raise ValueError("the wrongful-stop rule needs max_wrongful_stop in [0, 1]")
        feasible = np.flatnonzero(fp / n_neg <= max_wrongful_stop)
        index = int(feasible[-1])
    else:
        raise ValueError(f"unknown threshold rule {rule!r}; choose from {THRESHOLD_RULES}")
    tau = float(taus[index])
    return {
        "rule": rule,
        "max_wrongful_stop": max_wrongful_stop,
        "tau": tau,
        "validation": classification_report(y, s, tau),
    }


def score_metrics(y: np.ndarray, s: np.ndarray, tau: float) -> dict[str, float]:
    """Every bootstrapped test metric, including each one's margin over its no-skill reference.

    ``*_lift`` = metric minus its no-skill value: AUROC - 0.5, AUPRC - class prevalence, and
    F1 minus the always-"hallucination" predictor. Negative lift means worse than no skill.
    """
    report = classification_report(y, s, tau)
    prevalence = report["prevalence_hallucination"]
    trivial = constant_hallucination_baseline(prevalence)
    values = {
        "auroc": auroc(y, s),
        "auprc_hallucination": average_precision(y, s, positive=1),
        "auprc_non_hallucination": average_precision(y, s, positive=0),
        "macro_f1": report["macro_f1"],
        "f1_hallucination": report["f1_hallucination"],
        "f1_non_hallucination": report["f1_non_hallucination"],
    }
    values.update({
        "auroc_lift": values["auroc"] - 0.5,
        "auprc_hallucination_lift": values["auprc_hallucination"] - prevalence,
        "auprc_non_hallucination_lift": values["auprc_non_hallucination"] - (1.0 - prevalence),
        "macro_f1_lift": values["macro_f1"] - trivial["macro_f1"],
        "f1_hallucination_lift": values["f1_hallucination"] - trivial["f1_hallucination"],
    })
    return values


def evaluate_classifier(
    val_labels: Any,
    val_scores: Any,
    test_labels: Any,
    test_scores: Any,
    *,
    rule: str = "macro-f1",
    max_wrongful_stop: float | None = None,
    test_question_ids: Any = None,
    n_boot: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Sections 1.1-1.3 for one classifier: tau from validation, every number from test.

    With ``n_boot > 0``, adds 95% percentile intervals (``<metric>_ci_low/high``) from a
    bootstrap over test questions (``test_question_ids``; per example if absent). tau stays
    frozen inside the bootstrap, so intervals cover test-sample noise, not tau-selection noise.
    """
    from .bootstrap import flatten_ci, resampler_for

    threshold = select_threshold(val_labels, val_scores, rule=rule, max_wrongful_stop=max_wrongful_stop)
    y, s = _validate(test_labels, test_scores)
    _require_both_classes(y, "test evaluation")
    tau = threshold["tau"]
    report = classification_report(y, s, tau)
    prevalence = report["prevalence_hallucination"]
    result: dict[str, Any] = {
        **report,
        **score_metrics(y, s, tau),
        "no_skill_auprc_hallucination": prevalence,
        "no_skill_auprc_non_hallucination": 1.0 - prevalence,
        "always_hallucination_f1": constant_hallucination_baseline(prevalence)["f1_hallucination"],
        "always_hallucination_macro_f1": constant_hallucination_baseline(prevalence)["macro_f1"],
        "threshold_rule": rule,
        "max_wrongful_stop": max_wrongful_stop,
        "validation_n": threshold["validation"]["n"],
        "validation_prevalence_hallucination": threshold["validation"]["prevalence_hallucination"],
        "validation_macro_f1": threshold["validation"]["macro_f1"],
        "validation_wrongful_stop_rate": threshold["validation"]["wrongful_stop_rate"],
    }
    questions = None if test_question_ids is None else [str(q) for q in test_question_ids]
    if questions is not None:
        if len(questions) != y.size:
            raise ValueError("test_question_ids must have one entry per test example")
        result.update({
            "n_questions": len(set(questions)),
            "n_questions_with_hallucination": len({q for q, label in zip(questions, y) if label == 1}),
            "n_questions_with_non_hallucination": len({q for q, label in zip(questions, y) if label == 0}),
        })
    if n_boot:
        resampler, ids = resampler_for(questions or [None] * y.size, n_boot, seed)
        codes = resampler.codes(ids)

        def statistic(counts: np.ndarray) -> dict[str, float] | None:
            index = resampler.expand(codes, counts)
            yb = y[index]
            if yb.size == 0 or yb.min() == yb.max():
                return None
            return score_metrics(yb, s[index], tau)

        boot = resampler.intervals(statistic)
        result.update(flatten_ci(boot["ci"]))
        result.update({"bootstrap_unit": boot["unit"], "bootstrap_n": boot["n_boot"], "bootstrap_valid": boot["n_valid"]})
    return result


def constant_hallucination_baseline(prevalence: float) -> dict[str, float]:
    """F1 of the useless classifier that always predicts hallucination (Section 1.3's warning)."""
    f1_hall = 2 * prevalence / (1 + prevalence) if prevalence > 0 else 0.0
    return {"f1_hallucination": f1_hall, "f1_non_hallucination": 0.0, "macro_f1": f1_hall / 2.0}
