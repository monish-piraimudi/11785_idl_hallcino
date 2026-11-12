#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/run_500.sh MODEL_CHECKPOINT CALIBRATION_TEXT OUTPUT_DIR

Environment options:
  OPENROUTER_API_KEY   Required; read it securely before invoking this script.
  LAYERS               Comma-separated zero-based layers (default: 8,16,32).
  MAX_NEW_TOKENS       Answer token budget (default: 32).
  SAMPLE_SEED          SimpleQA sampling seed (default: 0).
  JUDGE_MODEL          OpenRouter model (default: nvidia/nemotron-3-ultra-550b-a55b).
  DEVICE_MAP           Transformers device map (default: auto).
EOF
  exit 2
}

[[ $# -eq 3 ]] || usage
MODEL_CHECKPOINT=$1
CALIBRATION_TEXT=$2
OUTPUT_DIR=$3
LAYERS=${LAYERS:-8,16,32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
SAMPLE_SEED=${SAMPLE_SEED:-0}
JUDGE_MODEL=${JUDGE_MODEL:-nvidia/nemotron-3-ultra-550b-a55b}
DEVICE_MAP=${DEVICE_MAP:-auto}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is required; set it in the environment without putting it in this command." >&2
  exit 2
fi
if [[ ! -d "$MODEL_CHECKPOINT" || ! -f "$MODEL_CHECKPOINT/config.json" ]]; then
  echo "Expected a local Hugging Face model directory containing config.json: $MODEL_CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$CALIBRATION_TEXT" ]]; then
  echo "Calibration corpus not found: $CALIBRATION_TEXT" >&2
  echo "Provide separate UTF-8 calibration text, one sufficiently long passage per line." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
INPUT="$OUTPUT_DIR/simpleqa_500.jsonl"
LENS="$OUTPUT_DIR/jacobian_lens.safetensors"
RUN="$OUTPUT_DIR/collection"

echo "[1/4] Preparing 500 reproducibly sampled SimpleQA examples"
python -m hallucinot prepare-simpleqa --output "$INPUT" --count 500 --seed "$SAMPLE_SEED"

if [[ -f "$LENS" ]]; then
  echo "[2/4] Reusing existing lens after checking layers and calibration corpus"
  python -m hallucinot check-lens --lens "$LENS" --layers "$LAYERS" --calibration-text "$CALIBRATION_TEXT"
else
  echo "[2/4] Fitting the lens once on the separate calibration corpus"
  python -m hallucinot fit-lens \
    --model "$MODEL_CHECKPOINT" --device-map "$DEVICE_MAP" --layers "$LAYERS" \
    --calibration-text "$CALIBRATION_TEXT" --output "$LENS"
fi

echo "[3/4] Generating answers and saving per-token lens trajectories"
python -m hallucinot collect \
  --model "$MODEL_CHECKPOINT" --device-map "$DEVICE_MAP" --lens "$LENS" \
  --input "$INPUT" --format jsonl --layers "$LAYERS" --output "$RUN" \
  --max-new-tokens "$MAX_NEW_TOKENS" --seed "$SAMPLE_SEED"

echo "[4/4] Judging completed answers with $JUDGE_MODEL"
python -m hallucinot judge --run-dir "$RUN" --model "$JUDGE_MODEL"

echo "Done. Dataset JSONL: $RUN/labeled_records.jsonl"
echo "Feature tensors:    $RUN/features/*.safetensors"
echo "Run manifest:       $RUN/run.json"
echo "Judge summary:      $RUN/judging.json"
