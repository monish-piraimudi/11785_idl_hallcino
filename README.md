# HalluciNot — Baseline

Baseline harness for the HalluciNot proposal: measure the hallucination rate of an
**unmodified** LLM, with no monitoring/intervention. This corresponds to Baseline 1
in the proposal (unmodified model generating to completion).

Evaluates on [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa)
(generation split) — chosen because it's designed to elicit confident-but-false
answers, close to the proposal's zero-resource/factual hallucination category.
Each generated answer is scored by a second Claude call (LLM-as-judge) against
TruthfulQA's correct/incorrect reference answers.

The model under test is pluggable — see Backends below — so the harness can run
against Claude today and swap to Gemma (the proposal's actual target) via a hosted
API or local GPU later without changing any other code.

## Status

**Verified working: the `--mock` path only.** It exercises the full pipeline
(TruthfulQA download → generation → judging → aggregation → JSONL output) with
hardcoded responses and a heuristic judge — no API key, no network call to
Anthropic or HF Inference. See the smoke test below.

The real backends (`anthropic`, `hf_inference`, `hf_local`) are implemented but
**not yet run end-to-end** — this machine has no `ANTHROPIC_API_KEY` configured.
Once a key is in `.env`, drop `--mock` and they should work, but treat that as
unverified until someone actually runs it.

## Setup

```bash
cd 11785_idl_hallcino
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

For the local-GPU backend only: `pip install -e ".[local]"`.

## Run — verified (mock, no API key)

```bash
python -m hallcino.run_baseline --mock --n 5
```

Hardcoded canned responses stand in for generation, and a string-overlap
heuristic stands in for the judge (`backends/mock_backend.py`,
`judge.py::MockJudge`). Use this to confirm the harness itself (data loading,
loop, output format) is wired correctly.

## Run — real (unverified, needs a key)

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY (and HF_TOKEN if using HF backends)
python -m hallcino.run_baseline --n 50
```

Key flags (all also configurable via `.env`, see `.env.example`):

| Flag | Env var | Meaning |
|---|---|---|
| `--backend` | `HALLCINO_BACKEND` | `anthropic` \| `hf_inference` \| `hf_local` |
| `--model` | `HALLCINO_MODEL` | model id for the chosen backend |
| `--n` | — | number of TruthfulQA questions to sample |
| `--mock` | — | bypass everything above: hardcoded responses, no key needed |

Judge model is separate: `HALLCINO_JUDGE_MODEL` (always calls Anthropic, regardless
of which backend is under test — `--mock` is the only way to skip this).

Output: a summary hallucination rate printed to stdout, plus a per-example JSONL
file in `results/` (question, answer, verdict, rationale).

## Backends

All real backends implement the same `generate(prompt) -> GenerationResult`
interface (`src/hallcino/backends/base.py`), so switching is a config change:

- **`anthropic`** — calls the Claude API. Default, no GPU required. Stand-in for
  validating the pipeline/metrics before Gemma is wired up. *Unverified — no key
  on this machine yet.*
- **`hf_inference`** — calls a model (e.g. Gemma) hosted via HuggingFace Inference,
  no local GPU required. *Unverified.*
- **`hf_local`** — loads a model locally via `transformers` (`device_map="auto"`).
  Requires an actual GPU with enough VRAM for the chosen dtype. *Unverified.*
- **`mock`** — hardcoded canned responses, zero network calls. **Verified working.**

## Next steps (per proposal)

This covers Baseline 1 only. Not yet implemented: the entropy-based detector
baseline, the raw-residual-activation-probe baseline, and the quantization sweep
(BF16/FP8/NVFP4).
