"""Write explicitly synthetic Hallucinot-shaped safetensors and JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def _atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _feature_name(example_id: str) -> str:
    digest = hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:20]
    return f"example-{digest}.safetensors"


def _example(
    index: int,
    *,
    layers: list[int],
    feature_dim: int,
    max_steps: int,
    seed: int,
    feature_dir: Path,
) -> dict[str, Any]:
    example_id = f"synthetic-{index:04d}"
    rng = random.Random(seed + index)
    steps = rng.randint(2, max_steps)
    top_k = 5
    prompt_length = rng.randint(12, 48)
    generated_ids = [rng.randint(1000, 250000) for _ in range(steps)]
    token_pieces = [f" synthetic_token_{step}" for step in range(steps)]
    token_pieces[-1] = "<|synthetic_end_of_turn|>"
    generated_text = "".join(token_pieces[:-1]).strip()
    generator = torch.Generator(device="cpu").manual_seed(seed + index)

    # These random values only exercise file shapes and readers. They are not J-lens fits.
    arrays = {
        "j_transport": torch.randn((steps, len(layers), feature_dim), generator=generator, dtype=torch.float32),
        "readout_top_ids": torch.randint(0, 262144, (steps, len(layers), top_k), generator=generator),
        "readout_top_logits": torch.randn((steps, len(layers), top_k), generator=generator, dtype=torch.float32),
        "generated_token_ids": torch.tensor(generated_ids, dtype=torch.int64),
        "generation_step": torch.arange(steps, dtype=torch.int64),
        "source_activation_position": torch.arange(prompt_length - 1, prompt_length + steps - 1, dtype=torch.int64),
        "source_token_id": torch.tensor([100 + prompt_length, *generated_ids[:-1]], dtype=torch.int64),
        "source_is_prompt": torch.tensor([True] + [False] * (steps - 1), dtype=torch.bool),
        "source_attention_mask": torch.ones(steps, dtype=torch.bool),
        "is_special_token": torch.tensor([False] * (steps - 1) + [True], dtype=torch.bool),
        "is_control_token": torch.tensor([False] * (steps - 1) + [True], dtype=torch.bool),
        "is_eos_token": torch.zeros(steps, dtype=torch.bool),
        "is_end_of_turn_token": torch.tensor([False] * (steps - 1) + [True], dtype=torch.bool),
        "is_reasoning_token": torch.zeros(steps, dtype=torch.bool),
        "is_answer_token": torch.tensor([True] * (steps - 1) + [False], dtype=torch.bool),
    }
    filename = _feature_name(example_id)
    feature_path = feature_dir / filename
    temp = feature_path.with_name(f".{feature_path.name}.tmp")
    save_file(
        {name: tensor.contiguous() for name, tensor in arrays.items()},
        str(temp),
        metadata={
            "synthetic_data": "true",
            "warning": "Random plumbing fixture; not measured activations or fitted Jacobians.",
            "axis_j_transport": "[generated_steps, selected_layers, feature_dimension]",
            "axis_readout": "[generated_steps, selected_layers, top_k]",
            "layer_indices": json.dumps(layers),
            "prediction_alignment": "row s uses the activation before predicting generated token s",
        },
    )
    os.replace(temp, feature_path)

    return {
        "id": example_id,
        "synthetic_data": True,
        "status": "complete",
        "question": f"[SYNTHETIC] Example question {index}: what is placeholder fact {index}?",
        "reference_answer": f"Synthetic reference {index}",
        "actual_prompt": f"[SYNTHETIC PROMPT] Example question {index} only.",
        "prompt_specification": {"synthetic_data": True, "reference_answer": "not part of prompt"},
        "feature_file": f"features/{filename}",
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
        "generated_text_raw": "".join(token_pieces),
        "answer_text": generated_text,
        "token_pieces": token_pieces,
        "finish_reason": "end_of_turn",
        "layer_indices": layers,
        "activation_dtypes": {str(layer): "torch.float32 (synthetic)" for layer in layers},
        "activation_observations": {
            str(layer): [
                {"shape": [1, 1, feature_dim], "dtype": "torch.float32", "device": "cpu"}
                for _ in range(steps)
            ]
            for layer in layers
        },
        "readout_top_pieces": [
            [[f"synthetic_vocab_{index}" for index in range(top_k)] for _ in layers]
            for _ in range(steps)
        ],
        "kv_cache_class": "SyntheticCache",
        "kv_cache_dtypes": ["torch.float32 (synthetic)"],
        "alignment": {
            "type": "prediction_step",
            "description": "Synthetic row s stands in for the state before predicting generated token s.",
            "source_activation_positions": arrays["source_activation_position"].tolist(),
            "source_token_ids": arrays["source_token_id"].tolist(),
            "source_is_prompt": arrays["source_is_prompt"].tolist(),
        },
        "synthetic_warning": "No model was run; token IDs and feature values are fabricated placeholders.",
    }


def generate(output_dir: Path, *, count: int, layers: list[int], feature_dim: int, max_steps: int, seed: int) -> None:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix synthetic data with existing files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = output_dir / "features"
    feature_dir.mkdir(exist_ok=True)

    records = [
        _example(
            index,
            layers=layers,
            feature_dim=feature_dim,
            max_steps=max_steps,
            seed=seed,
            feature_dir=feature_dir,
        )
        for index in range(1, count + 1)
    ]
    _atomic_jsonl(output_dir / "records.jsonl", records)
    _atomic_json(
        output_dir / "run.json",
        {
            "schema": "hallucinot.synthetic-demo.v1",
            "synthetic_data": True,
            "warning": "Plumbing-only fixture. Do not use as experimental or classifier-training data.",
            "model": {"model_id": "SYNTHETIC_MODEL", "revision": "SYNTHETIC_REVISION"},
            "quantization": "SYNTHETIC_ONLY",
            "lens_artifact": "SYNTHETIC_LENS",
            "selected_layers": layers,
            "feature_schema": {
                "name": "j_transport",
                "axes": ["generated_steps", "selected_layers", "feature_dimension"],
                "feature_dimension": feature_dim,
                "alignment": "prediction_step",
            },
            "seed": seed,
            "example_count": count,
            "max_generated_steps": max_steps,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Wrote {count} synthetic examples to {output_dir}")
    print(f"Safetensors features: {feature_dir}")
    print("All vectors, IDs, prompts, and answers are fabricated placeholders; no model or lens was run.")


def _parse_layers(value: str) -> list[int]:
    try:
        layers = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers, such as 8,16,32") from exc
    if not layers or any(layer < 0 for layer in layers) or len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("layers must be unique nonnegative integers")
    return layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new/empty run directory")
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--layers", type=_parse_layers, default=_parse_layers("8,16,32"))
    parser.add_argument("--feature-dim", type=int, default=3840, help="use 3840 for Gemma 4 12B-shaped vectors")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.examples < 1 or args.feature_dim < 1 or args.max_steps < 2:
        parser.error("--examples and --feature-dim must be positive; --max-steps must be at least 2")
    generate(
        args.output,
        count=args.examples,
        layers=args.layers,
        feature_dim=args.feature_dim,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
