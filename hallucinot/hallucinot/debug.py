"""Stage-by-stage terminal checks for either the tiny fixture or local Gemma."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import read_input
from .generation import encode_question
from .lens import fit_artifact, load_artifact, validate_artifact
from .model import environment_report, layer_mapping, load_local_model, resolve_layers
from .pipeline import collect_examples, load_calibration_prompts, format_json
from .storage import load_saved_example
from .tiny import make_tiny_adapter

ROOT = Path(__file__).resolve().parents[1]


def run_fixture_debug(output_dir: str | Path, *, max_new_tokens: int = 8) -> dict[str, Any]:
    import torch
    from .generation import generate_with_trajectory

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("Stage 1: environment and deterministic tiny model")
    print(format_json(environment_report()))
    adapter = make_tiny_adapter()
    layers = [0, 2]
    print("Stage 2: valid QA input and question-only prompt path")
    records = read_input(ROOT / "data" / "qa" / "debug.jsonl")[:2]
    print(format_json([{"id": item.id, "question": item.question, "reference_answer_saved_only": item.answer} for item in records]))
    print("Stage 3: separate tiny calibration fixture and reference Jacobian fit")
    prompts, corpus_hash = load_calibration_prompts(ROOT / "data" / "calibration" / "tiny_calibration.txt")
    lens_path = output / "tiny_debug_lens.safetensors"
    artifact = fit_artifact(
        adapter, prompts, layers, lens_path, dim_batch=1, max_seq_len=128,
        skip_first=2, corpus_name="inadequate tiny debug fixture", corpus_sha256=corpus_hash,
    )
    print(format_json({"layers": layer_mapping(adapter.block_path, layers), "calibration_prompts": len(prompts), "artifact": str(lens_path)}))
    print("Stage 4: cached greedy decoding with hook capture")
    adapter.hf_model.forced_tokens = [ord("O") + 4, ord("K") + 4, 3]
    first = generate_with_trajectory(
        adapter, records[0].question, layers, artifact=artifact,
        max_new_tokens=max_new_tokens, top_k=5, seed=7, capture=True, use_cache=True,
    )
    print(format_json({
        "prompt": first.prompt_text,
        "generated_token_ids": first.generated_token_ids,
        "token_pieces": first.token_pieces,
        "finish_reason": first.finish_reason,
        "j_transport_shape": list(first.j_transport.shape),
        "source_activation_positions": first.source_activation_position.tolist(),
        "activation_dtypes": first.activation_dtypes,
        "finite": bool(torch.isfinite(first.j_transport).all()),
        "transport_norms": torch.linalg.vector_norm(first.j_transport, dim=-1).tolist(),
        "readout_top_ids_shape": list(first.readout_top_ids.shape),
    }))
    print("Stage 5: passive-observation and cached-vs-prefix checks")
    without_capture = generate_with_trajectory(
        adapter, records[0].question, layers, artifact=None,
        max_new_tokens=max_new_tokens, top_k=5, seed=7, capture=False, use_cache=True,
    )
    recomputed = generate_with_trajectory(
        adapter, records[0].question, layers, artifact=artifact,
        max_new_tokens=max_new_tokens, top_k=5, seed=7, capture=True, use_cache=False,
    )
    assert first.generated_token_ids == without_capture.generated_token_ids
    assert first.generated_token_ids == recomputed.generated_token_ids
    assert torch.allclose(first.j_transport, recomputed.j_transport, atol=1e-5, rtol=1e-5)
    print("Capture-enabled tokens equal capture-disabled tokens; cached and recomputed-prefix trajectories agree.")
    print("Stage 6: incremental collection and saved-output reconstruction")
    summary = collect_examples(
        adapter, artifact, records, output / "run", layer_indices=layers,
        max_new_tokens=max_new_tokens, top_k=5, seed=7,
    )
    record, arrays = load_saved_example(output / "run", records[0].id)
    assert arrays["j_transport"].shape[0] == len(record["generated_token_ids"])
    print(format_json({"summary": summary, "loaded_id": record["id"], "status": record["status"], "array_shapes": {key: list(value.shape) for key, value in arrays.items()}}))
    return {"output_dir": str(output), "summary": summary, "record_id": records[0].id}


def run_model_debug(
    model_path: str | Path,
    lens_path: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    layer_indices: list[int],
    *,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    print("Stage 1: hardware, packages, model identity, and configured quantization")
    print(format_json(environment_report()))
    adapter = load_local_model(model_path)
    layers = resolve_layers(layer_indices, adapter.n_layers)
    print(format_json({
        "model_class": type(adapter.hf_model).__name__,
        "processor_class": type(adapter.processor).__name__,
        "text_stack_path": adapter.block_path,
        "layers": layer_mapping(adapter.block_path, layers),
        "d_model": adapter.d_model,
        "n_layers": adapter.n_layers,
        "quantization": adapter.quantization,
    }))
    print("Stage 2: input parsing and prompt construction")
    records = read_input(input_path)[:1]
    if not records:
        raise ValueError("debug input contains no records")
    encoded, prompt = encode_question(adapter, records[0].question)
    prompt_ids = encoded["input_ids"][0].tolist()
    print(format_json({
        "id": records[0].id,
        "question": records[0].question,
        "reference_answer_saved_only": records[0].answer,
        "rendered_prompt": prompt,
        "prompt_token_count": len(prompt_ids),
        "prompt_token_prefix": [
            {
                "id": token_id,
                "piece": adapter.tokenizer.decode([token_id], skip_special_tokens=False),
                "is_special": token_id in set(adapter.tokenizer.all_special_ids),
            }
            for token_id in prompt_ids[:24]
        ],
    }))
    artifact = load_artifact(lens_path)
    validate_artifact(adapter, artifact, layers)
    print("Stage 3: artifact compatibility and calibration provenance")
    print(format_json({
        "lens_sha256": artifact.sha256,
        "source_layers": artifact.manifest["source_layers"],
        "calibration": artifact.manifest["calibration"],
    }))
    print("Stage 4: one production-path generation, trajectory checks, and persistence")
    summary = collect_examples(
        adapter, artifact, records, output_dir, layer_indices=layers,
        max_new_tokens=max_new_tokens, top_k=5, seed=0,
    )
    record, arrays = load_saved_example(output_dir, records[0].id)
    import torch

    norms = torch.linalg.vector_norm(arrays["j_transport"], dim=-1)
    print(format_json({
        "summary": summary,
        "model_identity": json.loads((Path(output_dir) / "run.json").read_text(encoding="utf-8"))["model"],
        "status": record["status"],
        "finish_reason": record["finish_reason"],
        "generated_token_ids": record["generated_token_ids"],
        "generated_token_alignment": [
            {
                "step": step,
                "token_id": token_id,
                "piece": record["token_pieces"][step],
                "source_position": record["alignment"]["source_activation_positions"][step],
                "source_token_id": record["alignment"]["source_token_ids"][step],
                "source_is_prompt": record["alignment"]["source_is_prompt"][step],
                "is_special": bool(arrays["is_special_token"][step]),
                "is_control": bool(arrays["is_control_token"][step]),
                "is_eos": bool(arrays["is_eos_token"][step]),
                "is_end_of_turn": bool(arrays["is_end_of_turn_token"][step]),
                "is_reasoning": bool(arrays["is_reasoning_token"][step]),
                "is_answer": bool(arrays["is_answer_token"][step]),
                "top_readout_tokens": record["readout_top_pieces"][step],
            }
            for step, token_id in enumerate(record["generated_token_ids"])
        ],
        "observed_activation_shapes_dtypes_devices": record["activation_observations"],
        "transport_norms_by_step_layer": norms.tolist(),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "j_transport_finite": bool(torch.isfinite(arrays["j_transport"]).all()),
    }))
    return {"output_dir": str(output_dir), "id": records[0].id}
