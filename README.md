# Hallucinot

Hallucinot calibrates and collects dense Jacobian-lens trajectories for factual QA. It is a data-generation tool; it does not grade answers or train a classifier.

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

## Repository map

- `hallucinot/model.py`: local checkpoint loading, module discovery, layer validation, hardware and precision reporting.
- `hallucinot/lens.py`: pinned Anthropic estimator integration, calibration provenance, artifact I/O, transport, and readout.
- `hallucinot/generation.py`: greedy KV-cache decoding and per-token hook capture.
- `hallucinot/data.py`, `prompts.py`, `pipeline.py`, `storage.py`: input conversion, question-only prompts, resumable collection, and output files.
- `hallucinot/cli.py`, `debug.py`, `tiny.py`: terminal commands and CPU/debug paths.
- `data/qa/debug.jsonl`: ten valid QA plumbing cases.
- `data/calibration/tiny_calibration.txt`: deliberately inadequate calibration fixture.
- `data/fixtures/malformed_qa.jsonl`: invalid-input test fixture.

The test suite is in `tests/`. Read [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) before interpreting features as evidence about hallucination.
