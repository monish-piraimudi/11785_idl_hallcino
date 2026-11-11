# Hallucinot

Hallucinot calibrates and collects dense Jacobian-lens trajectories for factual QA. It is a data-generation tool; it does not grade answers or train a classifier. It also contains the evaluation for hallucination classifiers trained on that data: AUROC, AUPRC, F1, per-layer AUROC and the top-k sweep.

Quick links: [run the pipeline on Gemma](#running-the-pipeline-on-gemma) · [evaluate a classifier](#evaluating-a-classifier)

## What it saves

For each generated tokenizer token and selected transformer block, it stores the dense float32 transported residual `j_transport = J_l h_l`, along with a small vocabulary-logit readout. Rows use prediction-step alignment: row 0 reads the final prompt position that predicts token 0; row s reads the activation after consuming generated token s - 1, which predicts token s. EOS and end-of-turn tokens keep their own rows and IDs.

The stored vectors are dense Jacobian transports. They are not the paper's sparse nonnegative J-space coordinates or a sparse decomposition. See [the usage guide](docs/USAGE.md) and [research notes](docs/RESEARCH_NOTES.md) for the mathematics, scope, and limits.

## Install

The Python code and file formats use platform-neutral paths and APIs and are intended for Python 3.12 on Linux and Windows. Install the PyTorch 2.11.0 wheel appropriate for the machine from the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing the pinned project dependencies.

Linux (bash):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# First install the PyTorch 2.11.0 wheel selected for this machine's CUDA setup.
python -m pip install -e '.[test]'
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

On a CPU-only development machine, the pinned CPU wheel also runs the tiny fixture. Use a separate environment on the target GPU machine so its compatible FP8 backend is preserved.

## First commands

Linux (bash):

```bash
hallucinot --help
hallucinot debug --fixture --output runs/debug-fixture
export GEMMA_CHECKPOINT=/models/gemma-4-12B-it-fp8
hallucinot inspect --model "$GEMMA_CHECKPOINT" --config-only
```

Windows (PowerShell):

```powershell
hallucinot --help
hallucinot debug --fixture --output runs/debug-fixture
$env:GEMMA_CHECKPOINT = "D:\models\gemma-4-12B-it-fp8"
hallucinot inspect --model $env:GEMMA_CHECKPOINT --config-only
```

The fixture is a deterministic tiny transformer for pipeline plumbing. Its calibration text is intentionally too small for scientific conclusions. For real model setup and collection, follow the copy-pasteable commands in [docs/USAGE.md](docs/USAGE.md).

The code has no Windows-only runtime dependency. This checkout was tested on Windows; Linux execution was not available in this environment. On Linux, run the fixture command above and `python -m pytest` before using a real checkpoint.

## Running the pipeline on Gemma

You must supply three things the repository does not include:

- **Model weights.** A local Gemma 4 12B-it checkpoint directory; the loader never downloads. Accept Google's license on Hugging Face and download it first.
- **Calibration corpus.** A UTF-8 text file with about 1,000 pretraining-style passages, one per line, each at least about 128 tokens. Never use QA questions or answers.
- **QA input.** `data/qa/debug.jsonl` works for a first run. For the real dataset, download OpenAI's `simple_qa_test_set.csv` into `data/`.

Run these steps in order on a GPU machine. Bash is shown; PowerShell differs only in environment-variable syntax (see [docs/USAGE.md](docs/USAGE.md)).

```bash
export GEMMA_CHECKPOINT=/path/to/gemma-4-12B-it
LAYERS=8,16,32          # zero-based source layers; any of 0..46 (47 is the lens target)

# 1. Inspect: check the checkpoint and the chosen layers
hallucinot inspect --model "$GEMMA_CHECKPOINT" --layers $LAYERS

# 2. Fit the Jacobian lens once (long GPU job; rerun the same command to resume)
hallucinot fit-lens --model "$GEMMA_CHECKPOINT" --layers $LAYERS \
  --calibration-text data/calibration/my_corpus.txt \
  --output artifacts/gemma4-12b-lens.safetensors

# 3. Smoke-test one example through the real model
hallucinot debug --model "$GEMMA_CHECKPOINT" --lens artifacts/gemma4-12b-lens.safetensors \
  --input data/qa/debug.jsonl --layers $LAYERS --output runs/model-debug --max-new-tokens 8

# 4. Collect the dataset (rerun the same command to resume)
hallucinot collect --model "$GEMMA_CHECKPOINT" --lens artifacts/gemma4-12b-lens.safetensors \
  --input data/simple_qa_test_set.csv --format simpleqa-csv \
  --layers $LAYERS --max-new-tokens 32 --top-k 200 --output runs/simpleqa-test
```

Notes:

- `fit-lens` prints its cost before it starts: one forward pass and `ceil(3840 / dim_batch)` backward passes per calibration passage. It then runs a gradient preflight. If that fails, the checkpoint's FP8 backend cannot backpropagate; use a BF16 checkpoint.
- `collect` must use the same checkpoint and runtime the lens was fitted with, and its `--layers` must be a subset of the fitted layers.
- `--top-k 200` keeps enough readout tokens for the top-k sweep in the evaluation (k up to 200). The default of 5 is too small. Generation is a one-time run, so set this up front.
- `--max-examples N` limits a run. A new `--output` directory is required whenever a setting changes.
- Output lands in `runs/<name>/`. `run.json` holds provenance. `records.jsonl` holds the question, reference answer and generated text per example. `features/*.safetensors` holds `j_transport [steps, layers, 3840]` plus the top-k readout ids and logits.

Answers are not graded. Correctness labels (LLM judge) and the token-level labels for classifier training are separate steps.

## Evaluating a classifier

Evaluation needs only numpy and matplotlib: no torch, GPU or model. It runs on a laptop.

```bash
python -m pip install -e '.[eval]'     # or just: pip install numpy matplotlib (then run from the repo root)
```

Each classifier writes one predictions file (CSV or JSONL) with one row per labeled example:

| column | required | meaning |
|---|---|---|
| `split` | yes | `val` or `test`. Split by question, never by token. |
| `label` | yes | `1` = hallucination, `0` = non-hallucination |
| `score` | yes | higher = more likely hallucination (a probability, logit or count) |
| `question_id` | strongly recommended | enables question-level confidence intervals and the val/test leakage check |
| `config` | no | classifier name (e.g. `residual-mlp`, `tdnn`); one bar per config |
| `layer`, `k` | for `layers` / `topk` | the probe's source layer and top-k size |

```csv
config,split,question_id,label,score
residual-mlp,val,simpleqa-0012,1,0.93
residual-mlp,test,simpleqa-0457,0,0.18
```

Then run whichever analyses you need:

```bash
# Main comparison: AUROC, per-class AUPRC, macro-F1 at a tau chosen on val and frozen for test
python -m hallucinot.evaluation classifier --predictions preds.csv --output results/eval

# Same, but choose tau to keep the wrongful-stop (false-positive) rate on val at most 5%
python -m hallucinot.evaluation classifier --predictions preds.csv --output results/eval \
  --threshold-rule wrongful-stop --max-wrongful-stop 0.05

# Per-layer AUROC and important layers (one-layer probes; uses val rows only)
python -m hallucinot.evaluation layers --predictions layer_preds.csv --output results/eval

# Top-k sensitivity: best-layer AUROC against k and the chosen k (val rows only)
python -m hallucinot.evaluation topk --predictions topk_preds.csv --output results/eval
```

| command | figure | data files |
|---|---|---|
| `classifier` | `classifier_metrics.png`: bar charts sorted by AUROC with 95% CIs, no-skill references, and a class-balance table | `classifier_metrics.csv`, `.json` |
| `layers` | `layer_auroc.png`: AUROC by layer with CI band; important layers filled | `layer_auroc.csv`, `layer_summary.json` |
| `topk` | `topk_auroc.png`: best-layer AUROC against k with CI band; chosen k marked | `topk_curve.csv`, `topk_by_layer.csv`, `topk_summary.json` |

All commands accept `--n-boot` (bootstrap resamples, default 1000; `0` disables CIs), `--seed` and `--no-plots`. See [docs/EVALUATION.md](docs/EVALUATION.md) for the metric definitions and caveats.

## Running the tests

```bash
python -m pytest                          # full suite (needs the torch/jlens install above)
python -m pytest tests/test_evaluation.py # evaluation only (numpy, matplotlib, pytest)
```

## Repository map

- `hallucinot/model.py`: local checkpoint loading, module discovery, layer validation, hardware and precision reporting.
- `hallucinot/lens.py`: pinned Anthropic estimator integration, calibration provenance, artifact I/O, transport, and readout.
- `hallucinot/generation.py`: greedy KV-cache decoding and per-token hook capture.
- `hallucinot/data.py`, `prompts.py`, `pipeline.py`, `storage.py`: input conversion, question-only prompts, resumable collection, and output files.
- `hallucinot/cli.py`, `debug.py`, `tiny.py`: terminal commands and CPU/debug paths.
- `hallucinot/evaluation/`: classifier metrics (AUROC, AUPRC, F1 at a validation-frozen threshold, per-layer AUROC, top-k sweep) from a predictions table; see [docs/EVALUATION.md](docs/EVALUATION.md).
- `data/qa/debug.jsonl`: ten valid QA plumbing cases.
- `data/calibration/tiny_calibration.txt`: deliberately inadequate calibration fixture.
- `data/fixtures/malformed_qa.jsonl`: invalid-input test fixture.

The test suite is in `tests/`. Read [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) before interpreting features as evidence about hallucination.
