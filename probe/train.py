"""Smoke-test training loop: trains the baseline MLP probe on mock data so the
model/fusion/training code can be validated before real J-lens features exist.

    python -m probe.train --feature-dim 128 --num-layers 3 --fusion scalar_mix

Every run writes its mock data, predictions, trained weights, and metrics to
results/probe_<timestamp>/, following the same results/ convention as
src/hallcino's run_baseline.py.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from probe.losses import per_answer_bce_loss
from probe.mock_data import MockDataset, generate_mock_dataset
from probe.model import FUSION_MODES, MLPProbe


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline MLP probe on mock data")
    parser.add_argument("--n-prompts", type=int, default=400)
    parser.add_argument(
        "--feature-dim", type=int, default=128, help="swap this in once the real J-lens feature size is known"
    )
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--fusion", choices=FUSION_MODES, default="scalar_mix")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--weight-decay", type=float, default=1e-2, help="small probes overfit fast on few thousand examples"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="results dir; default results/probe_<timestamp>")
    args = parser.parse_args()

    data = generate_mock_dataset(args.n_prompts, args.feature_dim, args.num_layers, seed=args.seed)

    # Split by prompt, not by row: rows from the same prompt share a label and
    # a correlated signal, so a row-level split would leak across train/val.
    n_train_prompts = int(0.8 * args.n_prompts)
    train_mask = data.prompt_ids < n_train_prompts
    val_mask = ~train_mask
    train_x, train_y = data.features[train_mask], data.labels[train_mask]
    val_x, val_y = data.features[val_mask], data.labels[val_mask]
    train_prompt_ids = data.prompt_ids[train_mask]

    model = MLPProbe(args.feature_dim, args.num_layers, fusion=args.fusion)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    epoch_history = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        # Per-answer-averaged loss, not a flat mean over rows: a flat mean
        # would overweight prompts with more generated tokens.
        loss = per_answer_bce_loss(model(train_x), train_y, train_prompt_ids)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = ((model.predict_proba(val_x) > 0.5).float() == val_y).float().mean()
        epoch_history.append({"epoch": epoch, "train_loss": loss.item(), "val_acc": val_acc.item()})
        print(f"epoch {epoch:02d}  train_loss={loss.item():.4f}  val_acc={val_acc.item():.3f}")

    layer_weights = None
    if args.fusion == "scalar_mix":
        layer_weights = model.fusion.layer_weights().tolist()
        print("layer weights:", layer_weights)

    val_metrics = _eval_metrics(model, val_x, val_y)
    _print_eval_metrics(val_metrics)

    _print_example_table(model, data, val_mask, n_examples=2)
    latency_summary = _detection_latency_summary(model, data, val_mask)
    _print_detection_latency(latency_summary)

    output_dir = Path(args.output or os.path.join("results", f"probe_{_timestamp()}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_dataset(output_dir, data)
    _save_predictions(output_dir, model, data, train_mask)
    _save_model(output_dir, model, args)
    _save_metrics(output_dir, args, epoch_history, layer_weights, val_metrics, latency_summary, n_train_prompts)
    print(f"\nwrote results to {output_dir}")


def _eval_metrics(model: MLPProbe, x: torch.Tensor, y: torch.Tensor) -> dict:
    """AUROC, average precision, and macro-F1 @ 0.5 -- the metrics
    hallucinot/docs/CLASSIFIER_DESIGN.md calls for, as opposed to plain
    accuracy (never mentioned there, since it's a poor signal under class
    imbalance)."""
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(x).numpy()
    labels = y.numpy()
    predicted = (probs > 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
        "macro_f1_at_0.5": float(f1_score(labels, predicted, average="macro")),
    }


def _print_eval_metrics(metrics: dict) -> None:
    print(
        f"\nval metrics: AUROC={metrics['auroc']:.3f}  "
        f"avg_precision={metrics['average_precision']:.3f}  "
        f"macro_f1@0.5={metrics['macro_f1_at_0.5']:.3f}"
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _save_dataset(output_dir: Path, data: MockDataset) -> None:
    save_file(
        {
            "features": data.features.contiguous(),
            "labels": data.labels.contiguous(),
            "prompt_ids": data.prompt_ids.contiguous(),
            "token_positions": data.token_positions.contiguous(),
        },
        str(output_dir / "data.safetensors"),
        metadata={"axes": "features: [row, feature_dim, num_layers]; others: [row]"},
    )


def _save_predictions(output_dir: Path, model: MLPProbe, data: MockDataset, train_mask: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(data.features)
    path = output_dir / "predictions.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for i in range(data.features.shape[0]):
            stream.write(
                json.dumps(
                    {
                        "prompt_id": int(data.prompt_ids[i]),
                        "token_position": int(data.token_positions[i]),
                        "split": "train" if train_mask[i] else "val",
                        "ground_truth_hallucination": bool(data.labels[i]),
                        "predicted_prob": float(probs[i]),
                        "predicted_hallucination": bool(probs[i] > 0.5),
                    }
                )
                + "\n"
            )


def _save_model(output_dir: Path, model: MLPProbe, args: argparse.Namespace) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "feature_dim": args.feature_dim,
                "num_layers": args.num_layers,
                "fusion": args.fusion,
            },
        },
        output_dir / "model.pt",
    )


def _save_metrics(
    output_dir: Path,
    args: argparse.Namespace,
    epoch_history: list[dict],
    layer_weights: list[float] | None,
    val_metrics: dict,
    latency_summary: dict,
    n_train_prompts: int,
) -> None:
    metrics = {
        "args": vars(args),
        "n_train_prompts": n_train_prompts,
        "n_val_prompts": args.n_prompts - n_train_prompts,
        "epoch_history": epoch_history,
        "final_val_acc": epoch_history[-1]["val_acc"] if epoch_history else None,
        "val_metrics": val_metrics,
        "layer_weights": layer_weights,
        "detection_latency": latency_summary,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _detection_latency_summary(model: MLPProbe, data: MockDataset, val_mask: torch.Tensor) -> dict:
    """For each held-out hallucinated prompt, how many tokens in did the probe
    first cross p(hallucination) > 0.5 -- this is the "earlier is better"
    metric; overall accuracy doesn't tell you that."""
    model.eval()
    hallucinated_ids = data.prompt_ids[val_mask & (data.labels == 1)].unique()
    if hallucinated_ids.numel() == 0:
        return {"n_hallucinated_prompts": 0, "never_flagged": 0, "mean_detection_fraction": None}

    never_detected = 0
    detection_fractions = []
    with torch.no_grad():
        for prompt_id in hallucinated_ids.tolist():
            rows = data.prompt_ids == prompt_id
            order = data.token_positions[rows].argsort()
            positions = data.token_positions[rows][order]
            probs = model.predict_proba(data.features[rows])[order]

            hits = (probs > 0.5).nonzero()
            if hits.numel() == 0:
                never_detected += 1
            else:
                first_hit_position = positions[hits[0].item()].item()
                length = positions.max().item() + 1
                detection_fractions.append(first_hit_position / length)

    mean_fraction = sum(detection_fractions) / len(detection_fractions) if detection_fractions else None
    return {
        "n_hallucinated_prompts": int(hallucinated_ids.numel()),
        "never_flagged": never_detected,
        "mean_detection_fraction": mean_fraction,
    }


def _print_detection_latency(summary: dict) -> None:
    n = summary["n_hallucinated_prompts"]
    print(f"\ndetection latency over {n} held-out hallucinated prompts:")
    print(f"  never flagged: {summary['never_flagged']}/{n}")
    if summary["mean_detection_fraction"] is not None:
        print(
            f"  mean earliest-detection point: {summary['mean_detection_fraction']:.2f} "
            "of the way through the sequence (0=first token)"
        )


def _print_example_table(model: MLPProbe, data: MockDataset, val_mask: torch.Tensor, n_examples: int) -> None:
    """Per-token prob-of-hallucination vs. ground truth, for a few held-out
    prompts -- the same shape as the worked example (token position, predicted
    prob, ground truth), minus the actual generated text, which mock data
    doesn't have."""
    model.eval()
    val_prompt_ids = data.prompt_ids[val_mask].unique()[:n_examples]
    print("\nexample prompts (held-out):")
    with torch.no_grad():
        for prompt_id in val_prompt_ids.tolist():
            rows = data.prompt_ids == prompt_id
            probs = model.predict_proba(data.features[rows])
            label = data.labels[rows][0].item()
            print(f"  prompt {prompt_id}  (ground truth hallucination={bool(label)})")
            for position, prob in zip(data.token_positions[rows].tolist(), probs.tolist()):
                print(f"    token {position}  p(hallucination)={prob:.3f}  predicted={prob > 0.5}")


if __name__ == "__main__":
    main()
