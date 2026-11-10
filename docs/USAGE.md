# Usage and debugging

## Install and verify the environment

Python 3.12 is pinned in pyproject.toml. Linux and Windows use the same Python modules and file formats. On the target machine, use the [official PyTorch selector](https://pytorch.org/get-started/locally/) to install a PyTorch 2.11.0 wheel compatible with that machine's CUDA driver, then install the package.

Linux (bash):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Install the selected PyTorch 2.11.0 CUDA or CPU wheel first.
python -m pip install -e '.[test]'
hallucinot --help
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
hallucinot --help
```

The model loader reads a local Hugging Face checkpoint only; it never downloads model weights. The target machine must already have the checkpoint and a compatible Transformers FP8 quantization backend. Do not infer FP8 from a folder name: inspect the checkpoint's quantization_config and the runtime's reported parameter/activation dtypes.

The implementation uses `pathlib`, standard Python file replacement, PyTorch, Transformers, and safetensors; it has no Windows-only runtime dependency. This checkout was exercised on Windows, but Linux execution was unavailable here because WSL access was denied. On a Linux host, run the fixture workflow and tests shown below before loading the large checkpoint.

## Inspect the local model and choose layers

Set the local checkpoint path. Config-only inspection does not allocate model weights; full inspection loads the actual model and verifies the module list.

Linux (bash):

```bash
export GEMMA_CHECKPOINT=/models/gemma-4-12B-it-fp8
hallucinot inspect --model "$GEMMA_CHECKPOINT" --config-only
hallucinot inspect --model "$GEMMA_CHECKPOINT" --layers 8,16,32
```

Windows (PowerShell):

```powershell
$env:GEMMA_CHECKPOINT = "D:\models\gemma-4-12B-it-fp8"
hallucinot inspect --model $env:GEMMA_CHECKPOINT --config-only
hallucinot inspect --model $env:GEMMA_CHECKPOINT --layers 8,16,32
```

The official 12B config currently describes 48 text decoder blocks; the live model's config and modules are authoritative. Indices are zero-based. The selected path and exact boundary are printed as output of a complete text block before final RMSNorm. The default lens target is the final text block output, so that final block cannot also be requested as a source layer. Invalid and duplicate indices raise an error.

The 12B target is the language backbone at model.language_model. Its unified wrapper also has raw-image and raw-audio input projection modules; those are not transformer text blocks. The paired assistant/MTP draft checkpoint is a separate model and is not loaded. Hallucinot uses normal one-token greedy decoding and no draft model.

## Prepare a separate calibration corpus

The fit command expects UTF-8 text with one non-empty pretraining-like passage per line. Use roughly 1,000 sequences, typically 128 model tokens each, when the compute budget permits. Keep evaluation questions, held-out QA examples, and reference answers out of this file. For example, convert your own licensed corpus into one passage per line:

```python
from pathlib import Path

paragraphs = [
    "A self-contained paragraph from a suitable non-evaluation corpus.",
    "Another passage from the same calibration corpus.",
]
Path("data/calibration/my_corpus.txt").write_text(
    "\n".join(" ".join(text.split()) for text in paragraphs if text.strip()) + "\n",
    encoding="utf-8",
)
```

The bundled tiny_calibration.txt is for debugging only. The CLI prints the forward/backward count, selected dense lens size, and approximate peak temporary-plus-final lens disk space before fitting. Peak activation memory and elapsed time depend on your quantizer and GPU; dim_batch controls a memory/launch tradeoff. The default 8 matches the upstream estimator. Use a smaller value if the replicated prompt graph does not fit.

Fit once and retain the artifact. The temporary upstream resume checkpoint is removed after a successful fit and remains on disk after an interruption so an identical command can resume. Linux (bash):

```bash
hallucinot fit-lens --model "$GEMMA_CHECKPOINT" --layers 8,16,32 --calibration-text data/calibration/my_corpus.txt --output artifacts/gemma4-12b-fp8-lens.safetensors --dim-batch 8 --max-seq-len 128 --skip-first 16
```

Windows (PowerShell):

```powershell
hallucinot fit-lens --model $env:GEMMA_CHECKPOINT --layers 8,16,32 --calibration-text data/calibration/my_corpus.txt --output artifacts/gemma4-12b-fp8-lens.safetensors --dim-batch 8 --max-seq-len 128 --skip-first 16
```

If the FP8 backend cannot differentiate through frozen model weights to the captured activations, the real autograd preflight stops before fitting and reports the backend. There is no automatic BF16 fallback. A BF16-calibrated lens used on an FP8 model is a numerical approximation. The collector rejects it unless you explicitly opt in after independent held-out validation; the required validation note is provenance only, not an automated validation.

## Run the debug fixture

This follows the real fit, decode, hook, output, resume, and reload code paths on a deterministic 4-block CPU model:

```bash
hallucinot debug --fixture --output runs/debug-fixture --max-new-tokens 8
python -m pytest
```

The QA fixture at data/qa/debug.jsonl has ten stable examples covering names, numbers, punctuation, Unicode, and likely multi-token answers. The malformed JSONL is separate under data/fixtures. The tiny calibration corpus is also kept separate. The fixture validates plumbing, not Gemma weights, FP8 kernels, scientific lens quality, or hallucination detection.

## Collect SimpleQA or custom examples

JSONL is one JSON object per line with a stable id, question, and reference answer. Optional extra columns and source_metadata are preserved:

```json
{"id":"capital-001","question":"What is the capital of Japan?","answer":"Tokyo","source_metadata":{"split":"pilot"}}
```

The official SimpleQA CSV conversion explicitly maps problem to question and answer to answer, retaining other nonempty fields as source metadata. Linux (bash):

```bash
hallucinot collect --model "$GEMMA_CHECKPOINT" --lens artifacts/gemma4-12b-fp8-lens.safetensors --input data/qa/debug.jsonl --layers 8,16,32 --max-new-tokens 32 --top-k 5 --seed 0 --output runs/gemma4-debug
hallucinot collect --model "$GEMMA_CHECKPOINT" --lens artifacts/gemma4-12b-fp8-lens.safetensors --input data/simple_qa_test_set.csv --format simpleqa-csv --layers 8,16,32 --max-new-tokens 32 --output runs/simpleqa-test
```

Windows (PowerShell):

```powershell
hallucinot collect --model $env:GEMMA_CHECKPOINT --lens artifacts/gemma4-12b-fp8-lens.safetensors --input data/qa/debug.jsonl --layers 8,16,32 --max-new-tokens 32 --top-k 5 --seed 0 --output runs/gemma4-debug
hallucinot collect --model $env:GEMMA_CHECKPOINT --lens artifacts/gemma4-12b-fp8-lens.safetensors --input data/simple_qa_test_set.csv --format simpleqa-csv --layers 8,16,32 --max-new-tokens 32 --output runs/simpleqa-test
```

Reference answers are saved in metadata and are never passed to the prompt builder. Generation is deterministic greedy decoding, batch size one, and KV caching is enabled. The 32-token limit is configurable. Rows are retained for EOS and end-of-turn tokens. Finish reason is eos, end_of_turn, or token_budget.

Run the model-path debug command on one short input before a large collection:

```bash
hallucinot debug --model "$GEMMA_CHECKPOINT" --lens artifacts/gemma4-12b-fp8-lens.safetensors --input data/qa/debug.jsonl --layers 8,16,32 --output runs/model-debug --max-new-tokens 8
```

It reports environment and checkpoint identity, prompt token pieces, selected block paths, artifact provenance, per-step token/source alignment and masks, observed activation shapes/dtypes/devices, transport norms, vocabulary readouts, and the reloaded safetensors arrays. During generation it copies only the final token vector from each selected block hook and asks Gemma 4 for only the last-position logits. It intentionally raises with a normal traceback if a stage fails.

## Output files, resume, and reconstruction

A run directory contains:

- run.json: model and tokenizer/checkpoint hashes, quantization config and observed parameter dtypes, runtime versions, lens hash and calibration provenance, input digest, prompt template, generation settings, and feature schema.
- records.jsonl: one per-example completion or error record. Complete records include id, question, reference answer, source metadata, actual rendered prompt and prompt IDs, generated text and IDs, token pieces, alignment metadata, finish reason, hook observations, and readout token pieces.
- features/example-<hash>.safetensors: one example's numerical arrays. The main j_transport axes are [generated_steps, selected_layers, feature_dimension]. Other arrays include readout IDs/logits, generated IDs, generation step, source positions and IDs, source flags/masks, and special/channel-control/EOS/end-of-turn/reasoning/answer masks.

The record row convention is prediction_step. Row 0 reads the last prompt position. Row s > 0 reads the activation after consuming the previous generated token. This differs from the activation produced after the current token has been fed back. source_activation_position is the zero-based position in the concatenated prompt/generated sequence; source_token_id identifies the token at that position. Each row has exactly one token ID and one trajectory row per selected layer.

The writer saves each example's safetensors file atomically, then atomically upserts its JSONL record. A completed example is skipped on resume. Errors are written and retried on the next run. run.json contains a compatibility hash; changing model/tokenizer files, lens, input, layers, prompt, runtime, or generation settings requires a new output directory. This prevents mixing incompatible rows.

Read trajectories in Python:

```python
from hallucinot.storage import load_saved_example

record, arrays = load_saved_example("runs/gemma4-debug", "q-<stable-id>")
trajectory = arrays["j_transport"].numpy()  # [steps, selected_layers, feature_dim]
print(record["generated_text"], record["finish_reason"], trajectory.shape)
print(arrays["source_activation_position"].tolist())
```

For the public collection API, callers may pass a list of records or a question-to-answer mapping:

```python
from hallucinot.data import read_input
from hallucinot.lens import load_artifact
from hallucinot.model import load_local_model
from hallucinot.pipeline import collect_examples

adapter = load_local_model("/models/gemma-4-12B-it-fp8")
artifact = load_artifact("artifacts/gemma4-12b-fp8-lens.safetensors")
examples = read_input("data/qa/debug.jsonl")
summary = collect_examples(
    adapter, artifact, examples, "runs/python-api",
    layer_indices=[8, 16, 32], max_new_tokens=32, seed=0,
)
print(summary)
```

Python callers can instead pass [{"id": "...", "question": "...", "answer": "..."}] or {"question text": "reference answer"} directly to collect_examples. The latter receives deterministic question-hash IDs.

## Troubleshooting

- Model loading: pass a local Hugging Face directory with config.json, tokenizer/processor assets, and weight shards. Use config-only inspect first. Check that your Transformers build exposes AutoModelForMultimodalLM and Gemma4UnifiedForConditionalGeneration. No remote model download is attempted.
- FP8: inspect quantization_config, reported method, parameter dtypes, activation observations, and KV-cache dtypes. The tool does not convert checkpoints. A BF16 runtime in output metadata is not FP8 collection.
- Gradient failure: inference can work while activation gradients do not. Fit-lens runs a backward preflight and fails rather than turning off graph tracking. Check backend autograd support; do not enable model-wide no_grad/inference_mode around fitting.
- Hook boundary: inspect prints the actual block path. Hooks attach to complete text-block outputs before the final norm. A different decoder layout must be deliberately adapted in model.py and verified against the final norm/head.
- Alignment: run debug with a short token budget. Prediction rows should start at prompt length minus one, advance by one, and match the saved generated token IDs. The final token retains its row even when it ends the turn or exhausts the budget.
- Memory: collect batch size is fixed at one and moves each feature row to CPU. During fitting, lower dim_batch to reduce replicated activation memory. Reduce max_seq_len only with a documented reason; it changes calibration context.
- Schema/resume: do not edit records.jsonl in the middle of a run. If compatibility differs, choose a fresh output directory. Feature tensors are ordinary safetensors files and can be read with safetensors or load_saved_example.
- MTP/reasoning: the paired MTP assistant is not loaded. Thinking is explicitly disabled through the Gemma processor. Generated token IDs preserve special/channel markers even when generated_text omits them.

## Adapting the tool

For a new QA format, add an explicit conversion in data.py and test its field mapping. For a new model family, inspect its actual module tree first, then adapt load_local_model and check its wrapper inputs, residual boundary, final norm/head, output softcap, cache behavior, and hook gradient support. Keep the source/target boundary identical between calibration and collection. Do not reuse this calibration target automatically for a different architecture.
