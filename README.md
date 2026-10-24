# HalluciNOT

## Baseline

A research pipeline evaluating how **model quantization and uncertainty-aware prompting** affect factual accuracy, hallucination rates, and abstention in Gemma 4 12B.
This experiment evaluates `google/gemma-4-12B-it` on the [SimpleQA](https://huggingface.co/datasets/OpenEvals/SimpleQA) benchmark across six configurations:

| Precision | Baseline | Uncertainty-aware |
|---|---|---|
| BF16 | ✓ | ✓ |
| FP8 | ✓ | ✓ |
| FP4 | ✓ | ✓ |

- **Baseline:** Standard factual question answering.
- **Uncertainty-aware:** Instructs the model to respond with "I don't know" when insufficiently confident.

FP8 uses fine-grained quantization through Transformers, while FP4 uses bitsandbytes.

## Evaluation

Responses are evaluated using Nemotron 3 Ultra through OpenRouter and classified as **correct**, **incorrect**, or **not attempted**.

Key metrics include:

- **Accuracy:** Fraction of questions answered correctly.
- **Hallucination rate:** Fraction of attempted answers that are incorrect.
- **Abstention rate:** Fraction of questions the model declines to answer.
- **Coverage:** Fraction of questions attempted.
- **SimpleQA F-score:** Harmonic mean of overall and attempted accuracy.

The pipeline also computes bootstrap confidence intervals and generates comparison plots.

## Setup

Install dependencies:

```bash
pip install "transformers>=5.10.1" datasets accelerate \
  openai huggingface_hub matplotlib numpy pandas scipy \
  packaging "bitsandbytes>=0.46.1"
```

Configure credentials:

```bash
export HF_TOKEN="your_huggingface_token"
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

Run on a CUDA-enabled GPU:

```bash
python simpleqa_quantization.py
```

By default, the experiment evaluates 256 questions. Set `MAX_QUESTIONS = None` for the full benchmark.

## Outputs

Results are saved to `simpleqa_gemma4_quant/`, including:

- `gemma_outputs.json` — Generated model responses
- `judge_results.json` — LLM judge classifications
- `judge_details.csv` — Per-question evaluation details
- `metrics.csv` — Aggregate performance metrics
- `results_table.tex` — LaTeX results table
- Bootstrap confidence intervals and comparison plots

The pipeline supports automatic checkpointing, resuming interrupted runs, quantization validation, and CUDA out-of-memory recovery.
