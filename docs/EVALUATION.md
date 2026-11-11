# Classifier evaluation

`hallucinot.evaluation` scores a hallucination classifier from a table of its predictions. It needs only numpy and matplotlib (`pip install -e '.[eval]'`), not torch, jlens, or a GPU, so any classifier (residual MLP, TDNN/CNN, GRU/LSTM, lexicon baseline) can be evaluated the same way by writing one file.

## Input: a predictions table

CSV or JSONL, one row per labeled example (a question, a generation position, and a set of layers):

| column | required | meaning |
|---|---|---|
| `split` | yes | `val` or `test` |
| `label` | yes | `1` = hallucination, `0` = non-hallucination |
| `score` | yes | higher = more likely hallucination (a probability, logit, or count) |
| `config` | no | classifier/run name; one bar per config (default `default`) |
| `question_id` | no, but recommended | if present, a question in both `val` and `test` is an error |
| `layer` | for `layers` and `topk` | zero-based source layer of a one-layer probe |
| `k` | for `topk` | top-k readout size the probe was trained with |

Split by question, never by token position: positions from one answer are strongly correlated.

## Commands

```bash
# 1.1-1.3: AUROC, per-class AUPRC, F1 at a tau chosen on val and frozen for test
python -m hallucinot.evaluation classifier --predictions preds.csv --output results/eval
python -m hallucinot.evaluation classifier --predictions preds.csv --output results/eval \
  --threshold-rule wrongful-stop --max-wrongful-stop 0.05

# 1.4: per-layer AUROC, peak layer and important set I (validation rows only)
python -m hallucinot.evaluation layers --predictions layer_preds.csv --output results/eval --margin 0.2

# 1.5: best-layer AUROC against k and the plateau k (validation rows only)
python -m hallucinot.evaluation topk --predictions topk_preds.csv --output results/eval --tolerance 0.005
```

All three commands take `--n-boot` (default 1000; `0` disables confidence intervals) and `--seed`.

Outputs: `classifier_metrics.{csv,json,png}`, `layer_auroc.{csv,png}` + `layer_summary.json`, `topk_curve.csv`, `topk_by_layer.csv`, `topk_summary.json`, `topk_auroc.png`.

## Figures

- **`classifier_metrics.png`** follows the SimpleQA baseline's `make_plots` style. Configs are sorted by AUROC, and every bar has a 95% CI.
  - The top row shows AUROC, macro-F1, and AUPRC with non-hallucination as positive. The second row starts with non-hallucination F1. Each panel has a dashed no-skill reference.
  - Hallucination-class AUPRC and F1 sit near 1 for any classifier at ~97% prevalence. These two panels show the *margin over no skill* instead.
  - Any bar below its reference is red and hatched.
  - A table below the bars gives test size, questions, non-hallucination questions, hallucination rate, the frozen tau and the threshold rule.
- **`layer_auroc.png`** plots AUROC by layer with a shaded 95% CI. The grey band is |AUROC - 0.5| <= tau_layer; filled markers are important layers. The legend gives the number of non-hallucination validation questions behind each point.
- **`topk_auroc.png`** plots best-layer AUROC against k with a shaded 95% CI, the best layer labelled at each point, and the chosen k marked.

## Definitions as implemented

- **Decision rule:** predict hallucination iff `score >= tau`.
- **AUROC (Eq. 1):** normalized Mann-Whitney U from mid-ranks; a tied (positive, negative) pair counts 1/2. Tested against the literal double sum and scikit-learn.
- **AUPRC:** step-wise average precision, `sum_n (R_n - R_{n-1}) P_n` over distinct thresholds (scikit-learn's `average_precision_score`). Reported with hallucination as positive and with non-hallucination as positive (ranked by `-score`), each next to its no-skill value (that class's prevalence).
- **F1:** per-class F1 and macro-F1 at `tau`. A class with no true and no predicted members gets F1 = 0. The plot also marks the constant "always hallucination" predictor (F1 = 2p/(1+p) on that class, macro-F1 half of that).
- **Threshold:** chosen on validation only, then frozen. `macro-f1` takes the validation-macro-F1 maximizer (ties -> highest tau). `wrongful-stop` takes the lowest tau whose validation wrongful-stop rate is at most the target. The wrongful-stop rate is the fraction of non-hallucination examples flagged (false-positive rate). The test report states the test class balance.
- **Important layers (Eq. 2):** `I = {l : |AUROC_l - 0.5| > tau_layer}`, strict inequality, default 0.2; anti-correlated layers count. Peak layer = highest AUROC (ties -> lower layer).
- **Top-k:** for each k, the best layer's validation AUROC. The chosen k is the smallest k within `--tolerance` of the sweep maximum. This turns "where the curve levels off" into a rule without assuming the curve is monotone. The summary flags non-monotone curves; look at the plot before freezing k.

- **Lift:** each metric minus its no-skill value: AUROC - 0.5, AUPRC - class prevalence, and F1 / macro-F1 minus the always-"hallucination" predictor. Negative means worse than no skill.
- **Confidence intervals:** 95% percentile intervals from a cluster bootstrap. Each resample draws whole *questions* with replacement and keeps all of a drawn question's positions. Resampling single positions would treat correlated tokens as independent and give much narrower, wrong intervals. The tests check this: question-level intervals come out more than 1.5x wider on correlated data. Without a `question_id` on every row the bootstrap falls back to examples, and the figure says the intervals are too narrow.
  - Classifier intervals keep tau frozen and resample only the test set, so they cover test-sample noise, not threshold-selection noise.
  - Per-layer and top-k intervals reuse the same question draws for every layer and k (paired).
  - The top-k interval re-picks the best layer inside each resample, so it includes the noise of choosing that layer.
  - Resamples that contain only one class are skipped. `bootstrap_valid` in the outputs counts the resamples used.

`layers` and `topk` drop test rows and refuse to run without validation rows. Layer and k selection on test data would inflate the final AUROC.

## Caveats

- Examples are token positions, and several come from one answer, so they are not independent. Treat small AUROC differences between configs with care.
- At ~97% hallucination prevalence a validation split has few negatives. Per-layer and top-k curves can then be noisy enough to pick a layer or k by chance. Check how many non-hallucination *questions* back each point.
- If hallucinations come mostly from SimpleQA and non-hallucinations mostly from EasyBall QA, a classifier can score well by detecting the source. Report metrics within each source when both classes exist there.
