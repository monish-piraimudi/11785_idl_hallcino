# End-to-end 500-example run

This Linux workflow samples 500 questions from the official OpenAI SimpleQA CSV, calibrates the Jacobian lens on a separate local text corpus, generates answers with the local checkpoint, captures the prediction-step vectors, and grades each completed answer through OpenRouter. It uses deterministic greedy Gemma decoding and serial judge requests. It does not assign labels from the dataset name: the judge grades each answer against its reference.

## Install on the GPU host

Use Python 3.12. Install a PyTorch 2.11.0 wheel matching the host's CUDA or ROCm runtime first. Choose the exact wheel command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/); a CUDA install-index command has this form, with the suffix chosen for the installed driver/runtime:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Example form only: select cuXXX using the official PyTorch selector.
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cuXXX
python -m pip install -r requirements.txt
```

The project requirements pin Transformers, Accelerate, safetensors, NumPy, pytest, and the Anthropic `jlens` implementation commit. The PyTorch wheel is host-specific. The tool does not recast the Gemma checkpoint or silently switch precision. `fit-lens` checks activation-gradient support on the loaded model/backend before calibration proceeds.

## Supply the judge key without exposing it in shell history

The judge needs an OpenRouter key and sends the question, reference answer, and generated answer to the hosted judge. The local feature tensors and model checkpoint stay on the GPU host.

```bash
read -rsp "OpenRouter API key: " OPENROUTER_API_KEY
printf '\n'
export OPENROUTER_API_KEY
```

The project never saves this variable in a manifest or output file. Do not put the key in a command argument or commit it to a file.

## Run the complete 500-example workflow

Prepare a UTF-8 calibration corpus separately from SimpleQA, with one sufficiently long passage per line. The lens fitter defaults to a maximum of 128 tokens and skips the first 16 positions, so each retained line must tokenize to more than 17 tokens. Do not use the SimpleQA questions/reference answers or the QA evaluation set as calibration text. A larger domain-matched corpus can be supplied in the same line-per-passage format; the corpus path and SHA-256 are stored in the lens manifest.

From the project root:

```bash
bash scripts/run_500.sh \
  /models/gemma-4-12B-it-fp8 \
  /data/separate_calibration_passages.txt \
  runs/gemma4-simpleqa-500
```

The default selected zero-based layers are `8,16,32`; override with `LAYERS=...`. The script first fits a lens once. If interrupted after fitting, rerunning checks that the existing lens uses the same requested layers and calibration file, then reuses it. Collection and grading resume completed examples by ID. A different corpus, model, layer set, or incompatible collection setup requires a fresh run directory. The normal collector checks the model fingerprint, runtime, precision/quantization, artifact, and input manifest before reusing saved examples.

Optional settings:

```bash
LAYERS=8,16,32 MAX_NEW_TOKENS=32 SAMPLE_SEED=0 \
  JUDGE_MODEL=nvidia/nemotron-3-ultra-550b-a55b DEVICE_MAP=auto \
  bash scripts/run_500.sh MODEL_PATH CALIBRATION_TEXT OUTPUT_DIR
```

`MODEL_PATH` must point to a local Hugging Face checkpoint directory containing `config.json` and weights. For actual module discovery, use `python -m hallucinot inspect --model MODEL_PATH --layers 8,16,32` before the full run if you want to inspect the selected block mapping. The `fit-lens` stage also prints the concrete module boundaries and preflight gradient report. If this backend cannot differentiate through the frozen checkpoint's block activations, calibration fails with an explicit error; there is no BF16 fallback.

## Files produced

```text
runs/gemma4-simpleqa-500/
├── simpleqa_500.jsonl               # 500 question/reference inputs and source provenance
├── jacobian_lens.safetensors         # fitted dense Jacobian matrices + manifest
└── collection/
    ├── run.json                      # model, precision, prompt, generation, lens provenance
    ├── records.jsonl                 # generated text, token alignment, status and references
    ├── features/example-*.safetensors # per-example [steps, layers, hidden] float32 vectors
    ├── judgments.jsonl               # resumable answer-level judge labels
    ├── labeled_records.jsonl         # collection metadata joined to its judge record by ID
    └── judging.json                   # grade counts and judge provenance summary
```

`labeled_records.jsonl` retains the trajectory file reference (`feature_file`) and adds a nested `judge` object. `j_transport` in the safetensors files is the dense transported residual `J_l h_l`; it is not the paper's sparse nonnegative J-space coordinate decomposition. Every trajectory row is aligned to the token that the row predicted. These labels cover the whole generated answer; they do not point to the token where an incorrect claim begins. Token-level supervision needs a separate claim/span annotation step.

The judge follows SimpleQA's answer grading distinction: `CORRECT`, `INCORRECT`, and `NOT_ATTEMPTED`, with an additional `UNCERTAIN` option for ambiguous cases. `hallucination_label` is `1` for `INCORRECT`, `0` for `CORRECT` and `NOT_ATTEMPTED`, and JSON `null` for `UNCERTAIN`. This is an operational answer-level label based on one model judge, not a verified factuality guarantee. Keep the four-way `grade`, judge model, reason, provider response ID, and rubric version for audit. Review uncertain cases and manually audit a sample of every class before using labels for training.

The 500 questions are sampled from SimpleQA's official evaluation file. Once used for this project, those particular examples are no longer an untouched test set. Use a separate held-out set for classifier evaluation. SimpleQA does not have a 100% hallucination rate; outcomes are determined from Gemma's answer and the judge. The current workflow uses SimpleQA only rather than mixing it with trivial arithmetic questions, because source/domain can otherwise become a shortcut feature for the classifier.

The default trajectory storage estimate for 500 examples, 32 generated tokens, three layers, and hidden width 3840 is approximately 0.69 GiB of dense float32 transports, plus small readouts and metadata. The CLI prints the actual estimate from the loaded model before collection. Lens fitting cost is also preflighted and reported before gradients are computed.

## Individual stages

For debugging or staged execution, use the same commands the script calls:

```bash
python -m hallucinot prepare-simpleqa --output runs/qa.jsonl --count 500 --seed 0
python -m hallucinot fit-lens --model MODEL_PATH --layers 8,16,32 \
  --calibration-text CALIBRATION_TEXT --output runs/lens.safetensors
python -m hallucinot collect --model MODEL_PATH --lens runs/lens.safetensors \
  --input runs/qa.jsonl --format jsonl --layers 8,16,32 --output runs/collection
python -m hallucinot judge --run-dir runs/collection
```

Judge retries transient 408/409/429/5xx and network failures with bounded backoff. A malformed response or non-retryable HTTP error stops the command with a traceback and response detail; rerunning resumes only the examples not yet judged. OpenRouter structured output is required at the provider-routing layer, so a provider without JSON-schema support fails visibly rather than silently returning unvalidated labels.
