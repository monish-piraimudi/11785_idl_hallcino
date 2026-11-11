"""Figures in the style of the SimpleQA baseline's ``make_plots`` (simpleqa_gemma4_quant.py on main)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

# Config colors: the baseline's first two ("baseline", "uncertain"), then seaborn "deep" minus
# red, which is reserved for "worse than the no-skill reference".
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
BELOW_REFERENCE = "#D62728"
STYLE = {"figure.figsize": (9, 5), "axes.grid": True, "grid.alpha": 0.25, "font.size": 10}

# (metric key, panel title, reference key or constant, is_lift)
CLASSIFIER_PANELS = [
    ("auroc", "AUROC (primary)", 0.5, False),
    ("macro_f1", "Macro F1 @ τ", "always_hallucination_macro_f1", False),
    ("auprc_non_hallucination", "AUPRC, non-hallucination as positive", "no_skill_auprc_non_hallucination", False),
    ("f1_non_hallucination", "F1, non-hallucination @ τ", None, False),
    ("auprc_hallucination_lift", "AUPRC, hallucination as positive\nminus no-skill (prevalence)", 0.0, True),
    ("f1_hallucination_lift", "F1, hallucination @ τ\nminus always-'hallucination' F1", 0.0, True),
]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _ci(row: dict[str, Any], key: str) -> tuple[float, float] | None:
    low, high = row.get(f"{key}_ci_low"), row.get(f"{key}_ci_high")
    return (low, high) if _finite(low) and _finite(high) else None


def _reference_value(row: dict[str, Any], reference: Any) -> float | None:
    if reference is None:
        return None
    return float(row[reference]) if isinstance(reference, str) else float(reference)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _bar_panel(axis, results, colors, key, title, reference, is_lift):
    xs = np.arange(len(results))
    values = np.array([row[key] for row in results], dtype=float)
    refs = [_reference_value(row, reference) for row in results]
    below = [ref is not None and value < ref for value, ref in zip(values, refs)]
    bars = axis.bar(
        xs, values, width=0.75,
        color=[BELOW_REFERENCE if flag else color for flag, color in zip(below, colors)],
        hatch=None, edgecolor="white",
    )
    for bar, flag in zip(bars, below):
        if flag:
            bar.set_hatch("//")
            bar.set_edgecolor("white")
    intervals = [_ci(row, key) for row in results]
    if all(interval is not None for interval in intervals):
        lower = values - np.array([interval[0] for interval in intervals])
        upper = np.array([interval[1] for interval in intervals]) - values
        axis.errorbar(xs, values, yerr=[np.maximum(lower, 0), np.maximum(upper, 0)],
                      fmt="none", ecolor="black", elinewidth=1, capsize=4, zorder=3)
    if is_lift:
        axis.axhline(0.0, color="black", linewidth=1)
    elif any(ref is not None for ref in refs):
        for x, ref in zip(xs, refs):
            if ref is not None:
                axis.hlines(ref, x - 0.375, x + 0.375, colors="black", linestyles="--", linewidth=1, zorder=4)

    tops = [interval[1] if interval else value for value, interval in zip(values, intervals)]
    bottoms = [interval[0] if interval else value for value, interval in zip(values, intervals)]
    if is_lift:
        extent = max([abs(v) for v in tops + bottoms] + [0.01]) * 1.45
        axis.set_ylim(-extent, extent)
        axis.set_ylabel("Margin over no skill")
        pad = extent * 0.04
    else:
        axis.set_ylim(0, 1.12)
        axis.set_ylabel("Score")
        pad = 0.02
    for x, value, top, bottom in zip(xs, values, tops, bottoms):
        text = f"{value:+.3f}" if is_lift else f"{value:.3f}"
        if is_lift and value < 0:
            axis.text(x, bottom - pad, text, ha="center", va="top", fontsize=8)
        else:
            axis.text(x, top + pad, text, ha="center", va="bottom", fontsize=8)
    axis.set_xticks(xs)
    axis.set_xticklabels([row["config"] for row in results], fontsize=8)
    axis.set_title(title, fontsize=11)


def _balance_table(axis, results) -> None:
    axis.axis("off")
    has_questions = all("n_questions" in row for row in results)
    header = ["Config", "Test examples", "Hallucination rate"]
    if has_questions:
        header += ["Test questions", "Non-halluc. questions"]
    header += ["τ (frozen)", "Rule", "AUROC [95% CI]"]
    body = []
    for row in results:
        cells = [row["config"], f"{row['n']:,}", f"{row['prevalence_hallucination']:.1%}"]
        if has_questions:
            cells += [f"{row['n_questions']:,}", f"{row['n_questions_with_non_hallucination']:,}"]
        rule = row["threshold_rule"] + (f" ≤{row['max_wrongful_stop']:g}" if row.get("max_wrongful_stop") is not None else "")
        interval = _ci(row, "auroc")
        auroc_text = f"{row['auroc']:.3f}" + (f" [{interval[0]:.3f}, {interval[1]:.3f}]" if interval else "")
        cells += [f"{row['tau']:.4g}", rule, auroc_text]
        body.append(cells)
    table = axis.table(cellText=body, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("0.8")
        if r == 0:
            cell.set_facecolor("0.93")
            cell.set_text_props(weight="bold")


def plot_classifier_metrics(results: list[dict[str, Any]], path: Path) -> Path:
    """Test-set comparison of classifier configs, best AUROC first, with question-level CIs."""
    plt.rcParams.update(STYLE)
    results = sorted(results, key=lambda row: -row["auroc"])
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(results))]
    fig = plt.figure(figsize=(17, 11))
    grid = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.12 + 0.06 * len(results)], hspace=0.38,
                    top=0.9, bottom=0.03)
    for index, (key, title, reference, is_lift) in enumerate(CLASSIFIER_PANELS):
        _bar_panel(fig.add_subplot(grid[index // 3, index % 3]), results, colors, key, title, reference, is_lift)
    _balance_table(fig.add_subplot(grid[2, :]), results)

    first = results[0]
    if "bootstrap_n" in first:
        unit = first["bootstrap_unit"]
        ci_note = f"Error bars: 95% bootstrap CI, {first['bootstrap_n']} resamples of test {unit}s"
        if unit != "question":
            ci_note += " (no question_id column, so CIs are too narrow)"
    else:
        ci_note = "No confidence intervals (run with --n-boot > 0)"
    fig.suptitle("Hallucination classifiers on the test set (sorted by AUROC; τ chosen on validation, then frozen)",
                 fontsize=15, y=0.975)
    fig.text(0.5, 0.945, f"{ci_note}.  Dashed lines: no-skill reference.  "
             "Red hatched bars fall below their reference (dashed line, or 0).",
             ha="center", fontsize=9)
    return _save(fig, path)


def plot_layer_curve(curve: list[dict[str, Any]], summary: dict[str, dict[str, Any]], margin: float, path: Path) -> Path:
    """Validation AUROC by layer with 95% bands; filled markers are important layers."""
    plt.rcParams.update(STYLE)
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.axhspan(0.5 - margin, 0.5 + margin, color="0.85", alpha=0.6, label=f"|AUROC − 0.5| ≤ {margin:g} (not important)")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    note = None
    for index, config in enumerate(summary):
        color = PALETTE[index % len(PALETTE)]
        points = sorted((row for row in curve if row["config"] == config), key=lambda row: row["layer"])
        layers = [row["layer"] for row in points]
        values = [row["auroc"] for row in points]
        info = summary[config]
        label = f"{config}: peak L{info['peak_layer']} ({info['peak_auroc']:.3f}), |I|={info['n_important']}"
        if info.get("n_questions_with_non_hallucination") is not None:
            label += f", {info['n_questions_with_non_hallucination']} non-halluc. val questions"
        axis.plot(layers, values, color=color, linewidth=1.5, label=label)
        if all(_finite(row.get("ci_low")) for row in points):
            axis.fill_between(layers, [row["ci_low"] for row in points], [row["ci_high"] for row in points],
                              color=color, alpha=0.15, linewidth=0)
            note = f"Shaded: 95% bootstrap CI over validation {points[0]['bootstrap_unit']}s"
        axis.scatter(layers, values, facecolors=[color if row["important"] else "white" for row in points],
                     edgecolors=color, zorder=3)
    axis.set_xlabel("Source layer (zero-based)")
    axis.set_ylabel("Validation AUROC (one-layer probe)")
    axis.set_ylim(0, 1)
    axis.set_title("Per-layer AUROC; filled markers = important layers (point estimate, Eq. 2)"
                   + (f"\n{note}" if note else ""), fontsize=11)
    axis.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    return _save(fig, path)


def plot_topk_curve(result: dict[str, Any], path: Path) -> Path:
    """Best-layer validation AUROC against k with 95% bands, and the chosen k."""
    plt.rcParams.update(STYLE)
    fig, axis = plt.subplots(figsize=(9, 5.5))
    all_k = sorted({row["k"] for row in result["curve"]})
    note = None
    for index, config in enumerate(result["chosen"]):
        color = PALETTE[index % len(PALETTE)]
        points = sorted((row for row in result["curve"] if row["config"] == config), key=lambda row: row["k"])
        ks = [row["k"] for row in points]
        values = [row["best_layer_auroc"] for row in points]
        chosen = result["chosen"][config]
        axis.plot(ks, values, marker="o", color=color, label=f"{config}: chosen k={chosen['chosen_k']}")
        if all(_finite(row.get("ci_low")) for row in points):
            axis.fill_between(ks, [row["ci_low"] for row in points], [row["ci_high"] for row in points],
                              color=color, alpha=0.15, linewidth=0)
            note = (f"Shaded: 95% bootstrap CI over validation {points[0]['bootstrap_unit']}s, "
                    "best layer re-picked in each resample")
        for row in points:
            axis.annotate(f"L{row['best_layer']}", (row["k"], row["best_layer_auroc"]),
                          textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
        axis.axvline(chosen["chosen_k"], color=color, linestyle=":", linewidth=1)
    if all_k and min(all_k) > 0:
        axis.set_xscale("log")
        axis.set_xticks(all_k)
        axis.set_xticklabels([str(k) for k in all_k])
        axis.minorticks_off()
    axis.margins(y=0.12)
    axis.set_xlabel("k (top-k readout size)")
    axis.set_ylabel("Best-layer validation AUROC")
    axis.set_title("Top-k sensitivity (labels: best layer at each k)" + (f"\n{note}" if note else ""), fontsize=11)
    axis.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)
