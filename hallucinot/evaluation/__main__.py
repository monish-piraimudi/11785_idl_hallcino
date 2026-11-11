"""Command line: ``python -m hallucinot.evaluation {classifier,layers,topk} ...``

Deliberately independent of torch/jlens so it runs on a laptop from a predictions file.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .bootstrap import DEFAULT_N_BOOT, DEFAULT_SEED
from .io import read_predictions, write_csv, write_json
from .metrics import THRESHOLD_RULES, evaluate_classifier
from .selection import DEFAULT_LAYER_MARGIN, DEFAULT_PLATEAU_TOLERANCE, important_layers, layer_auroc_curve, topk_curve


def _by_config(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"val": [], "test": []})
    for row in rows:
        grouped[row["config"]][row["split"]].append(row)
    return grouped


def _question_ids(rows: list[dict[str, Any]]) -> list[str] | None:
    if all("question_id" in row for row in rows):
        return [row["question_id"] for row in rows]
    print("note: no question_id on every row; bootstrap CIs resample examples and will be too narrow")
    return None


def _add_bootstrap_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT,
                        help="bootstrap resamples of questions for 95%% CIs (0 disables)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)


def _validation_only(rows: list[dict[str, Any]], command: str) -> list[dict[str, Any]]:
    val = [row for row in rows if row["split"] == "val"]
    ignored = len(rows) - len(val)
    if not val:
        raise ValueError(f"`{command}` selects layers/k and must use validation rows; the file has none")
    if ignored:
        print(f"note: ignored {ignored} test rows; `{command}` uses validation data only")
    return val


def run_classifier(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_predictions(args.predictions)
    results = []
    for config, splits in _by_config(rows).items():
        layers = {row.get("layer") for row in splits["val"] + splits["test"]}
        if len(layers) > 1:
            raise ValueError(
                f"config {config!r} mixes rows from layers {sorted(map(str, layers))}; give each probe its own "
                "config name, or use the `layers` command for per-layer probes"
            )
        if not splits["val"] or not splits["test"]:
            raise ValueError(f"config {config!r} needs both val rows (to choose τ) and test rows")
        result = evaluate_classifier(
            [row["label"] for row in splits["val"]], [row["score"] for row in splits["val"]],
            [row["label"] for row in splits["test"]], [row["score"] for row in splits["test"]],
            rule=args.threshold_rule, max_wrongful_stop=args.max_wrongful_stop,
            test_question_ids=_question_ids(splits["test"]), n_boot=args.n_boot, seed=args.seed,
        )
        results.append({"config": config, **result})
    out = Path(args.output)
    write_csv(out / "classifier_metrics.csv", results)
    write_json(out / "classifier_metrics.json", results)
    files = [str(out / "classifier_metrics.csv"), str(out / "classifier_metrics.json")]
    if not args.no_plots:
        from .plots import plot_classifier_metrics
        files.append(str(plot_classifier_metrics(results, out / "classifier_metrics.png")))
    columns = ("config", "n", "prevalence_hallucination", "auroc", "auroc_ci_low", "auroc_ci_high",
               "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "auprc_non_hallucination",
               "f1_non_hallucination", "auprc_hallucination_lift", "f1_hallucination_lift", "tau")
    return {"results": [{key: row[key] for key in columns if key in row} for row in results], "files": files}


def run_layers(args: argparse.Namespace) -> dict[str, Any]:
    rows = _validation_only(read_predictions(args.predictions, need=("layer",)), "layers")
    curve = layer_auroc_curve(rows, margin=args.margin, n_boot=args.n_boot, seed=args.seed)
    summary = important_layers(curve)
    out = Path(args.output)
    write_csv(out / "layer_auroc.csv", curve)
    write_json(out / "layer_summary.json", {"margin": args.margin, "split": "val", "configs": summary})
    files = [str(out / "layer_auroc.csv"), str(out / "layer_summary.json")]
    if not args.no_plots:
        from .plots import plot_layer_curve
        files.append(str(plot_layer_curve(curve, summary, args.margin, out / "layer_auroc.png")))
    return {"summary": summary, "files": files}


def run_topk(args: argparse.Namespace) -> dict[str, Any]:
    rows = _validation_only(read_predictions(args.predictions, need=("layer", "k")), "topk")
    result = topk_curve(rows, tolerance=args.tolerance, n_boot=args.n_boot, seed=args.seed)
    out = Path(args.output)
    write_csv(out / "topk_by_layer.csv", result["per_layer"])
    write_csv(out / "topk_curve.csv", result["curve"])
    write_json(out / "topk_summary.json", {"split": "val", "configs": result["chosen"]})
    files = [str(out / "topk_by_layer.csv"), str(out / "topk_curve.csv"), str(out / "topk_summary.json")]
    if not args.no_plots:
        from .plots import plot_topk_curve
        files.append(str(plot_topk_curve(result, out / "topk_auroc.png")))
    return {"chosen": result["chosen"], "files": files}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hallucinot.evaluation",
        description="Classifier-level metrics from a predictions table (CSV/JSONL with split, label, score).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    classifier = sub.add_parser("classifier", help="AUROC, per-class AUPRC, F1 at a validation-frozen τ (1.1-1.3)")
    classifier.add_argument("--predictions", required=True)
    classifier.add_argument("--output", required=True)
    classifier.add_argument("--threshold-rule", choices=THRESHOLD_RULES, default="macro-f1")
    classifier.add_argument("--max-wrongful-stop", type=float,
                            help="for --threshold-rule wrongful-stop: max validation false-positive rate")
    classifier.add_argument("--no-plots", action="store_true")
    _add_bootstrap_args(classifier)
    classifier.set_defaults(run=run_classifier)

    layers = sub.add_parser("layers", help="per-layer AUROC, peak layer and important set I (1.4)")
    layers.add_argument("--predictions", required=True)
    layers.add_argument("--output", required=True)
    layers.add_argument("--margin", type=float, default=DEFAULT_LAYER_MARGIN, help="τ_layer in Eq. (2)")
    layers.add_argument("--no-plots", action="store_true")
    _add_bootstrap_args(layers)
    layers.set_defaults(run=run_layers)

    topk = sub.add_parser("topk", help="best-layer AUROC against k and the plateau k (1.5)")
    topk.add_argument("--predictions", required=True)
    topk.add_argument("--output", required=True)
    topk.add_argument("--tolerance", type=float, default=DEFAULT_PLATEAU_TOLERANCE,
                      help="choose the smallest k within this AUROC of the sweep maximum")
    topk.add_argument("--no-plots", action="store_true")
    _add_bootstrap_args(topk)
    topk.set_defaults(run=run_topk)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "threshold_rule", None) == "wrongful-stop" and args.max_wrongful_stop is None:
        raise SystemExit("--threshold-rule wrongful-stop requires --max-wrongful-stop")
    print(json.dumps(args.run(args), indent=2, default=str))


if __name__ == "__main__":
    main()
