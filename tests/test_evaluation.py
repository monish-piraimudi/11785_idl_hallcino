from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hallucinot.evaluation import (
    auroc,
    average_precision,
    classification_report,
    evaluate_classifier,
    important_layers,
    layer_auroc_curve,
    select_threshold,
    topk_curve,
)
from hallucinot.evaluation.__main__ import main as eval_main
from hallucinot.evaluation.bootstrap import QuestionResampler
from hallucinot.evaluation.io import read_predictions
from hallucinot.evaluation.metrics import constant_hallucination_baseline


def _pairwise_auroc(labels, scores) -> float:
    """Eq. (1) literally: double sum over (positive, negative) pairs, ties count 1/2."""
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    total = sum((si > sj) + 0.5 * (si == sj) for si in pos for sj in neg)
    return total / (len(pos) * len(neg))


def _brute_macro_f1(labels, scores):
    best = (-1.0, None)
    for tau in [math.inf, *sorted(set(scores), reverse=True)]:
        value = classification_report(labels, scores, tau)["macro_f1"]
        if value > best[0]:
            best = (value, tau)
    return best


def _random_case(seed: int, n: int = 60, discrete: bool = True):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    labels[:2] = [0, 1]
    scores = rng.integers(0, 6, size=n).astype(float) if discrete else rng.normal(size=n) + labels
    return labels.tolist(), scores.tolist()


@pytest.mark.parametrize("seed", range(8))
def test_auroc_matches_pairwise_definition_including_ties(seed):
    labels, scores = _random_case(seed, discrete=seed % 2 == 0)
    assert auroc(labels, scores) == pytest.approx(_pairwise_auroc(labels, scores), abs=1e-12)


def test_auroc_edge_cases():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1, 0, 1], [3, 3, 3, 3]) == 0.5
    with pytest.raises(ValueError, match="both classes"):
        auroc([1, 1], [0.2, 0.3])
    with pytest.raises(ValueError, match="0 .* or 1"):
        auroc([0, 2], [0.2, 0.3])


@pytest.mark.parametrize("seed", range(6))
def test_metrics_agree_with_scikit_learn(seed):
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    labels, scores = _random_case(seed, discrete=seed % 2 == 0)
    y, s = np.array(labels), np.array(scores)
    assert auroc(y, s) == pytest.approx(sklearn_metrics.roc_auc_score(y, s))
    assert average_precision(y, s, positive=1) == pytest.approx(sklearn_metrics.average_precision_score(y, s))
    assert average_precision(y, s, positive=0) == pytest.approx(sklearn_metrics.average_precision_score(1 - y, -s))
    tau = float(np.median(s))
    report = classification_report(y, s, tau)
    predicted = (s >= tau).astype(int)
    assert report["macro_f1"] == pytest.approx(sklearn_metrics.f1_score(y, predicted, average="macro", zero_division=0))
    assert report["f1_hallucination"] == pytest.approx(sklearn_metrics.f1_score(y, predicted, zero_division=0))


def test_constant_hallucination_predictor_looks_good_on_f1_but_not_macro_f1():
    # Section 1.3: at 97% hallucination, always predicting "hallucination" has F1 ~0.985 on that class.
    labels = [1] * 97 + [0] * 3
    report = classification_report(labels, [1.0] * 100, tau=0.5)
    assert report["f1_hallucination"] == pytest.approx(0.98477, abs=1e-4)
    assert report["f1_non_hallucination"] == 0.0
    assert report["macro_f1"] == pytest.approx(0.49239, abs=1e-4)
    assert constant_hallucination_baseline(0.97)["f1_hallucination"] == pytest.approx(report["f1_hallucination"])


@pytest.mark.parametrize("seed", range(6))
def test_macro_f1_threshold_is_the_validation_optimum(seed):
    labels, scores = _random_case(seed, discrete=seed % 2 == 0)
    chosen = select_threshold(labels, scores, rule="macro-f1")
    best_value, _ = _brute_macro_f1(labels, scores)
    assert chosen["validation"]["macro_f1"] == pytest.approx(best_value)


def test_wrongful_stop_threshold_respects_budget_and_maximizes_recall():
    labels, scores = _random_case(3, n=200, discrete=False)
    chosen = select_threshold(labels, scores, rule="wrongful-stop", max_wrongful_stop=0.05)
    assert chosen["validation"]["wrongful_stop_rate"] <= 0.05
    candidates = [t for t in sorted(set(scores)) if classification_report(labels, scores, t)["wrongful_stop_rate"] <= 0.05]
    assert chosen["tau"] == min(candidates)
    with pytest.raises(ValueError, match="max_wrongful_stop"):
        select_threshold(labels, scores, rule="wrongful-stop")


def test_threshold_is_frozen_from_validation_not_refit_on_test():
    val_labels, val_scores = [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9]
    test_labels, test_scores = [0, 0, 1, 1], [0.7, 0.75, 0.8, 0.9]
    result = evaluate_classifier(val_labels, val_scores, test_labels, test_scores)
    assert result["tau"] == 0.6
    assert result["auroc"] == 1.0
    assert result["fp"] == 2  # the frozen τ misfires on test; refitting on test would hide this
    assert result["no_skill_auprc_hallucination"] == result["prevalence_hallucination"] == 0.5


def test_layer_importance_uses_strict_margin_and_both_directions():
    rows = []
    for layer, scores in {0: [0.1, 0.2, 0.3, 0.4], 1: [0.4, 0.3, 0.2, 0.1], 2: [0.1, 0.4, 0.2, 0.3]}.items():
        rows += [{"config": "probe", "layer": layer, "label": y, "score": s} for y, s in zip([0, 0, 1, 1], scores)]
    curve = layer_auroc_curve(rows, margin=0.2)
    by_layer = {row["layer"]: row for row in curve}
    assert by_layer[0]["auroc"] == 1.0 and by_layer[0]["important"]
    assert by_layer[1]["auroc"] == 0.0 and by_layer[1]["important"]  # anti-correlated still carries signal
    assert by_layer[2]["auroc"] == 0.5 and not by_layer[2]["important"]
    summary = important_layers(curve)["probe"]
    assert summary["peak_layer"] == 0
    assert summary["important_layers"] == [0, 1] and summary["n_important"] == 2


def test_layer_margin_boundary_is_exclusive():
    rows = [{"config": "c", "layer": 5, "label": y, "score": s} for y, s in zip([0, 0, 1, 1], [0.1, 0.3, 0.2, 0.4])]
    assert layer_auroc_curve(rows, margin=0.2)[0]["auroc"] == 0.75
    assert layer_auroc_curve(rows, margin=0.2)[0]["important"]
    assert not layer_auroc_curve(rows, margin=0.25)[0]["important"]  # |0.75 - 0.5| = 0.25 is not > 0.25


def test_topk_picks_smallest_k_on_the_plateau_and_flags_non_monotone_curves():
    target = {5: 0.70, 15: 0.80, 30: 0.86, 50: 0.862, 100: 0.85, 200: 0.861}
    rows = []
    for k, value in target.items():
        n_neg = 100
        # 1 positive vs 100 negatives: AUROC = fraction of negatives scored below the positive.
        below = round(value * n_neg)
        scores = [0.0] * below + [2.0] * (n_neg - below)
        rows += [{"config": "mlp", "k": k, "layer": 8, "label": 0, "score": s} for s in scores]
        rows.append({"config": "mlp", "k": k, "layer": 8, "label": 1, "score": 1.0})
        rows += [{"config": "mlp", "k": k, "layer": 16, "label": y, "score": 0.0} for y in (0, 1)]
    result = topk_curve(rows, tolerance=0.005)
    chosen = result["chosen"]["mlp"]
    assert chosen["chosen_k"] == 30
    assert chosen["chosen_k_best_layer"] == 8
    assert not chosen["monotone_nondecreasing"]


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _synthetic_predictions(rng, *, config: str, question_offset: int = 0, signal: float = 1.5) -> list[dict]:
    rows = []
    for split, n_questions, start in (("val", 40, 0), ("test", 60, 1000)):
        for q in range(n_questions):
            label = int(rng.random() < 0.8)
            for position in range(3):
                rows.append({
                    "config": config, "split": split, "question_id": f"q{start + q + question_offset}",
                    "position": position, "label": label,
                    "score": float(1 / (1 + math.exp(-(rng.normal() + signal * (2 * label - 1))))),
                })
    return rows


def test_classifier_cli_end_to_end(tmp_path: Path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(0)
    rows = _synthetic_predictions(rng, config="residual-mlp") + _synthetic_predictions(rng, config="gru", signal=0.5)
    predictions = _write_csv(tmp_path / "preds.csv", rows)
    eval_main(["classifier", "--predictions", str(predictions), "--output", str(tmp_path / "out"), "--n-boot", "200"])
    results = json.loads((tmp_path / "out" / "classifier_metrics.json").read_text())
    assert [row["config"] for row in results] == ["residual-mlp", "gru"]
    assert results[0]["auroc"] > results[1]["auroc"] > 0.5
    for row in results:
        assert row["bootstrap_unit"] == "question" and row["bootstrap_valid"] == 200
        assert row["auroc_ci_low"] <= row["auroc"] <= row["auroc_ci_high"]
        assert row["n_questions"] == 60
    assert (tmp_path / "out" / "classifier_metrics.png").stat().st_size > 0
    assert (tmp_path / "out" / "classifier_metrics.csv").exists()


def test_question_leakage_between_splits_is_rejected(tmp_path: Path):
    rows = [
        {"split": "val", "question_id": "q1", "label": 1, "score": 0.9},
        {"split": "val", "question_id": "q2", "label": 0, "score": 0.1},
        {"split": "test", "question_id": "q1", "label": 1, "score": 0.8},
    ]
    with pytest.raises(ValueError, match="both val and test"):
        read_predictions(_write_csv(tmp_path / "leak.csv", rows))


def test_layer_and_topk_cli_use_validation_rows_only(tmp_path: Path):
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(1)
    rows = []
    for k, layer in itertools.product((5, 50, 100), (4, 8, 16, 32)):
        signal = (layer / 16) * min(k, 50) / 50
        for row in _synthetic_predictions(rng, config="probe", signal=signal):
            rows.append({**row, "k": k, "layer": layer, "question_id": f"{row['question_id']}"})
    predictions = _write_csv(tmp_path / "layers.jsonl.csv", rows)
    eval_main(["layers", "--predictions", str(predictions), "--output", str(tmp_path / "layers"), "--n-boot", "100"])
    summary = json.loads((tmp_path / "layers" / "layer_summary.json").read_text())
    assert summary["split"] == "val"
    assert all(info["peak_ci"][0] <= info["peak_auroc"] <= info["peak_ci"][1] for info in summary["configs"].values())
    eval_main(["topk", "--predictions", str(predictions), "--output", str(tmp_path / "topk"), "--n-boot", "100"])
    topk = json.loads((tmp_path / "topk" / "topk_summary.json").read_text())["configs"]["probe"]
    assert topk["chosen_k"] in (50, 100)
    for name in ("layer_auroc.png", "layer_auroc.csv"):
        assert (tmp_path / "layers" / name).exists()
    assert (tmp_path / "topk" / "topk_auroc.png").exists()

    test_only = _write_csv(tmp_path / "test_only.csv", [{**r, "split": "test"} for r in rows[:20]])
    with pytest.raises(ValueError, match="validation"):
        eval_main(["layers", "--predictions", str(test_only), "--output", str(tmp_path / "bad")])


def test_classifier_cli_rejects_mixed_layers_in_one_config(tmp_path: Path):
    rows = [
        {"split": s, "label": y, "score": v, "layer": layer}
        for s in ("val", "test") for layer in (8, 16) for y, v in ((0, 0.2), (1, 0.8))
    ]
    with pytest.raises(ValueError, match="mixes rows from layers"):
        eval_main(["classifier", "--predictions", str(_write_csv(tmp_path / "p.csv", rows)),
                   "--output", str(tmp_path / "o"), "--no-plots"])


def test_resampler_draws_whole_questions_reproducibly():
    questions = ["a", "a", "a", "b", "c", "c"]
    resampler = QuestionResampler(questions, n_boot=50, seed=3)
    codes = resampler.codes(questions)
    for counts in resampler.counts:
        assert counts.sum() == 3  # three questions drawn per replicate
        index = resampler.expand(codes, counts)
        drawn = [questions[i] for i in index]
        for question, count in zip("abc", counts):
            assert drawn.count(question) == count * questions.count(question)
    again = QuestionResampler(questions, n_boot=50, seed=3)
    assert (again.counts == resampler.counts).all()


def test_question_bootstrap_is_wider_than_naive_example_bootstrap():
    # Five tokens per answer share the answer's score, so there are really only 40 independent units.
    rng = np.random.default_rng(7)
    labels, scores, questions = [], [], []
    for q in range(40):
        label = int(q % 4 != 0)
        score = rng.normal() + 0.8 * label
        for _ in range(5):
            labels.append(label)
            scores.append(score)
            questions.append(f"q{q}")
    val = ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    by_question = evaluate_classifier(*val, labels, scores, test_question_ids=questions, n_boot=400, seed=0)
    by_example = evaluate_classifier(*val, labels, scores, n_boot=400, seed=0)
    width = lambda r: r["auroc_ci_high"] - r["auroc_ci_low"]  # noqa: E731
    assert by_example["bootstrap_unit"] == "example"
    assert width(by_question) > 1.5 * width(by_example)


def test_lift_metrics_are_zero_for_no_skill_predictors():
    labels = [1] * 97 + [0] * 3
    result = evaluate_classifier([0, 1], [0.0, 1.0], labels, [1.0] * 100)
    assert result["f1_hallucination_lift"] == pytest.approx(0.0)
    assert result["auroc_lift"] == pytest.approx(0.0)
    assert result["auprc_hallucination_lift"] == pytest.approx(0.0)
