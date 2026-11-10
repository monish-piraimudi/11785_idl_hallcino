"""Atomic per-example safetensors and resumable JSONL metadata."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .generation import GenerationResult

RUN_SCHEMA = "hallucinot.run.v2"


def _atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_records(output_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(output_dir) / "records.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: malformed JSONL metadata: {exc}") from exc
    return records


def records_by_id(output_dir: str | Path) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in read_records(output_dir)}


def completed_ids(output_dir: str | Path) -> set[str]:
    return {key for key, record in records_by_id(output_dir).items() if record.get("status") == "complete"}


def ensure_run(output_dir: str | Path, manifest: dict[str, Any]) -> None:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = output_dir / "features"
    feature_dir.mkdir(exist_ok=True)
    path = output_dir / "run.json"
    proposed = {"schema": RUN_SCHEMA, **manifest}
    canonical = json.dumps(proposed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    proposed["compatibility_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("compatibility_sha256") != proposed["compatibility_sha256"]:
            raise ValueError(
                f"{path} describes an incompatible model, lens, input, prompt, or generation setup. "
                "Use a fresh output directory to avoid mixing runs."
            )
        return
    proposed["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(path, proposed)


def _feature_name(example_id: str) -> str:
    digest = hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:20]
    return f"example-{digest}.safetensors"


def write_complete(
    output_dir: str | Path,
    *,
    example_id: str,
    record: dict[str, Any],
    result: GenerationResult,
) -> None:
    output_dir = Path(output_dir).expanduser().resolve()
    feature_path = output_dir / "features" / _feature_name(example_id)
    arrays = {
        "j_transport": result.j_transport.float().contiguous(),
        "readout_top_ids": result.readout_top_ids.long().contiguous(),
        "readout_top_logits": result.readout_top_logits.float().contiguous(),
        "generated_token_ids": torch.tensor(result.generated_token_ids, dtype=torch.int64),
        "generation_step": result.generation_step.long().contiguous(),
        "source_activation_position": result.source_activation_position.long().contiguous(),
        "source_token_id": result.source_token_id.long().contiguous(),
        "source_is_prompt": result.source_is_prompt.bool().contiguous(),
        "source_attention_mask": result.source_attention_mask.bool().contiguous(),
        "is_special_token": result.is_special_token.bool().contiguous(),
        "is_control_token": result.is_control_token.bool().contiguous(),
        "is_eos_token": result.is_eos_token.bool().contiguous(),
        "is_end_of_turn_token": result.is_end_of_turn_token.bool().contiguous(),
        "is_reasoning_token": result.is_reasoning_token.bool().contiguous(),
        "is_answer_token": result.is_answer_token.bool().contiguous(),
    }
    for tensor in arrays.values():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"refusing to save non-finite feature tensor for {example_id}")
    temp = feature_path.with_name(f".{feature_path.name}.tmp")
    save_file(arrays, str(temp), metadata={
        "axis_j_transport": "[generated_steps, selected_layers, feature_dimension]",
        "axis_readout": "[generated_steps, selected_layers, top_k]",
        "layer_indices": json.dumps(result.layer_indices),
        "prediction_alignment": "row s uses activation before predicting generated token s",
    })
    os.replace(temp, feature_path)
    record = {
        **record,
        "id": example_id,
        "status": "complete",
        "feature_file": str(feature_path.relative_to(output_dir).as_posix()),
        "generated_token_ids": result.generated_token_ids,
        "generated_text": result.generated_text,
        "generated_text_raw": result.generated_text_raw,
        "answer_text": result.answer_text,
        "token_pieces": result.token_pieces,
        "readout_top_pieces": result.readout_top_pieces,
        "finish_reason": result.finish_reason,
        "layer_indices": result.layer_indices,
        "activation_dtypes": result.activation_dtypes,
        "activation_observations": result.activation_observations,
        "kv_cache_class": result.kv_cache_class,
        "kv_cache_dtypes": result.kv_cache_dtypes,
        "alignment": {
            "type": "prediction_step",
            "description": "row 0 reads the last prompt position; row s>0 reads the state after consuming generated token s-1",
            "source_activation_positions": result.source_activation_position.tolist(),
            "source_token_ids": result.source_token_id.tolist(),
            "source_is_prompt": result.source_is_prompt.tolist(),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _upsert_record(output_dir, record)


def write_error(output_dir: str | Path, record: dict[str, Any], error: BaseException) -> None:
    payload = {
        **record,
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _upsert_record(Path(output_dir).expanduser().resolve(), payload)


def _upsert_record(output_dir: Path, record: dict[str, Any]) -> None:
    records = records_by_id(output_dir)
    records[record["id"]] = record
    path = output_dir / "records.jsonl"
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for item in records.values():
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_saved_example(output_dir: str | Path, example_id: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    output_dir = Path(output_dir).expanduser().resolve()
    record = records_by_id(output_dir).get(example_id)
    if record is None:
        raise KeyError(f"no saved example {example_id!r} in {output_dir}")
    if record.get("status") != "complete":
        raise ValueError(f"example {example_id!r} has status {record.get('status')!r}")
    feature_path = output_dir / record["feature_file"]
    with safe_open(str(feature_path), framework="pt", device="cpu") as handle:
        arrays = {key: handle.get_tensor(key) for key in handle.keys()}
    expected = len(record["generated_token_ids"])
    if arrays["j_transport"].shape[0] != expected:
        raise ValueError(f"saved trajectory row count does not equal generated token count for {example_id}")
    return record, arrays
