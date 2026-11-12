"""Single-example-at-a-time collection API and resumable run coordination."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data import Example, normalize_records, records_digest
from .generation import generate_with_trajectory
from .lens import LensArtifact, validate_artifact
from .model import LoadedModel, adapter_model_identity, environment_report, layer_mapping, resolve_layers
from .prompts import prompt_spec
from .storage import completed_ids, ensure_run, write_complete, write_error


def run_manifest(
    adapter: LoadedModel,
    artifact: LensArtifact,
    examples: Sequence[Example],
    layers: list[int],
    *,
    max_new_tokens: int,
    top_k: int,
    seed: int,
    use_cache: bool,
    model_identity: dict[str, Any],
    calibration_approximation: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "model": model_identity,
        "model_class": type(adapter.hf_model).__name__,
        "tokenizer_class": type(adapter.tokenizer).__name__,
        "runtime": environment_report(),
        "quantization": adapter.quantization,
        "lens_artifact_sha256": artifact.sha256,
        "lens_upstream_commit": artifact.manifest.get("upstream_commit"),
        "lens_calibration_corpus_sha256": artifact.manifest.get("calibration", {}).get("corpus_sha256"),
        "calibration_numerical_approximation": calibration_approximation,
        "input_sha256": records_digest(examples),
        "n_examples": len(examples),
        "selected_layers": layer_mapping(adapter.block_path, layers),
        "prompt": prompt_spec(),
        "generation": {
            "max_new_tokens": max_new_tokens,
            "top_k_vocabulary_readout": top_k,
            "strategy": "greedy argmax; no sampling or speculative decoding",
            "seed": seed,
            "use_cache": use_cache,
            "batch_size": 1,
            "thinking": "processor enable_thinking=False; generated channel markers and reasoning tokens are retained and masked",
        },
        "feature_schema": {
            "j_transport": "dense float32 J_l h_l; [generated_steps, selected_layers, feature_dimension]",
            "readout": "top-k logits after model final norm and output head; diagnostics, not stable feature coordinates",
            "token_masks": "special, channel-control, EOS, end-of-turn, reasoning-body, and answer-body booleans",
            "alignment": "prediction_step",
        },
    }


def collect_examples(
    adapter: LoadedModel,
    artifact: LensArtifact,
    records: Sequence[Example] | Mapping[str, str] | Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    layer_indices: list[int],
    max_new_tokens: int = 32,
    top_k: int = 5,
    seed: int = 0,
    use_cache: bool = True,
    max_examples: int | None = None,
    allow_numerical_mismatch: bool = False,
    mismatch_validation_note: str | None = None,
) -> dict[str, int]:
    """Public Python API. Reference answers are saved but never passed to generation."""
    examples = _as_examples(records)
    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be positive")
        examples = examples[:max_examples]
    if not examples:
        raise ValueError("input contains no examples to collect")
    layers = resolve_layers(layer_indices, adapter.n_layers)
    model_identity = adapter_model_identity(adapter)
    approximation = validate_artifact(
        adapter,
        artifact,
        layers,
        allow_numerical_mismatch=allow_numerical_mismatch,
        validation_note=mismatch_validation_note,
        fingerprint=model_identity,
    )
    manifest = run_manifest(
        adapter, artifact, examples, layers,
        max_new_tokens=max_new_tokens, top_k=top_k, seed=seed, use_cache=use_cache,
        model_identity=model_identity,
        calibration_approximation=approximation,
    )
    ensure_run(output_dir, manifest)
    done = completed_ids(output_dir)
    completed = skipped = errors = 0
    for example in examples:
        if example.id in done:
            skipped += 1
            continue
        record = {
            "id": example.id,
            "question": example.question,
            "reference_answer": example.answer,
            "source_metadata": example.source_metadata,
            "prompt_specification": prompt_spec(),
        }
        try:
            result = generate_with_trajectory(
                adapter,
                example.question,
                layers,
                artifact=artifact,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                seed=seed,
                capture=True,
                use_cache=use_cache,
            )
            record["actual_prompt"] = result.prompt_text
            record["prompt_input_ids"] = result.prompt_input_ids
            write_complete(output_dir, example_id=example.id, record=record, result=result)
            completed += 1
            done.add(example.id)
            print(f"[{completed + skipped}/{len(examples)}] complete {example.id}: {len(result.generated_token_ids)} tokens")
        except Exception as exc:
            write_error(output_dir, record, exc)
            errors += 1
            raise
    return {"completed": completed, "skipped_completed": skipped, "errors": errors}


def _as_examples(records: Any) -> list[Example]:
    if isinstance(records, Sequence) and records and isinstance(records[0], Example):
        examples = list(records)
        if len({item.id for item in examples}) != len(examples):
            raise ValueError("duplicate example ids")
        return examples
    return normalize_records(records)


def load_calibration_prompts(path: str | Path) -> tuple[list[str], str]:
    """Read a separate one-prompt-per-line UTF-8 corpus and hash its exact bytes."""
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    prompts = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"calibration text file {path} contains no non-empty lines")
    import hashlib

    return prompts, hashlib.sha256(raw).hexdigest()


def format_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
