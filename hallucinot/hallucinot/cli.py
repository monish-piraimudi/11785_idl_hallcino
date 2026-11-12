"""Standard-library command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import read_input
from .debug import ROOT, run_fixture_debug, run_model_debug
from .lens import estimate_calibration, fit_artifact, load_artifact
from .judge import DEFAULT_MODEL as DEFAULT_JUDGE_MODEL, judge_run
from .model import (
    environment_report,
    layer_mapping,
    load_local_model,
    read_config,
    resolve_layers,
    text_config_values,
)
from .pipeline import collect_examples, load_calibration_prompts
from .simpleqa import prepare_simpleqa


def _layers(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated zero-based integers, e.g. 2,11,27") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("provide at least one layer index")
    if len(parsed) != len(value.split(",")):
        raise argparse.ArgumentTypeError("empty layer index in list")
    return parsed


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hallucinot", description="Calibrate and collect Jacobian-lens trajectories")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="inspect checkpoint config and actual text decoder blocks")
    inspect.add_argument("--model", required=True, help="local Hugging Face checkpoint directory")
    inspect.add_argument("--layers", type=_layers, help="zero-based block indices to print")
    inspect.add_argument("--device-map", default="auto")
    inspect.add_argument("--config-only", action="store_true", help="read config.json without loading weights")

    fit = sub.add_parser("fit-lens", help="fit and save model-specific Jacobian transports")
    fit.add_argument("--model", required=True)
    fit.add_argument("--layers", type=_layers, required=True)
    fit.add_argument("--calibration-text", required=True, help="separate UTF-8, one calibration passage per line")
    fit.add_argument("--output", required=True, help="safetensors lens artifact path")
    fit.add_argument("--device-map", default="auto")
    fit.add_argument("--dim-batch", type=int, default=8)
    fit.add_argument("--max-seq-len", type=int, default=128)
    fit.add_argument("--skip-first", type=int, default=16)

    check = sub.add_parser("check-lens", help="validate an existing lens against requested layers and calibration text")
    check.add_argument("--lens", required=True)
    check.add_argument("--layers", type=_layers, required=True)
    check.add_argument("--calibration-text", required=True)

    prepare = sub.add_parser("prepare-simpleqa", help="download and seed-sample official SimpleQA CSV into JSONL")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--count", type=int, default=500)
    prepare.add_argument("--seed", type=int, default=0)

    judge = sub.add_parser("judge", help="label completed answers through OpenRouter; requires OPENROUTER_API_KEY")
    judge.add_argument("--run-dir", required=True)
    judge.add_argument("--model", default=DEFAULT_JUDGE_MODEL)

    collect = sub.add_parser("collect", help="greedy-generate short answers and save trajectories")
    collect.add_argument("--model", required=True)
    collect.add_argument("--lens", required=True)
    collect.add_argument("--input", required=True, help="JSONL or official SimpleQA CSV")
    collect.add_argument("--format", choices=("auto", "jsonl", "simpleqa-csv"), default="auto")
    collect.add_argument("--layers", type=_layers, required=True)
    collect.add_argument("--output", required=True, help="run directory; repeat the same command to resume")
    collect.add_argument("--device-map", default="auto")
    collect.add_argument("--max-new-tokens", type=int, default=32)
    collect.add_argument("--top-k", type=int, default=5)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--max-examples", type=int)
    collect.add_argument("--no-cache", action="store_true", help="debug-only recomputed-prefix decoding")
    collect.add_argument(
        "--allow-calibration-model-mismatch",
        action="store_true",
        help="explicitly allow cross-checkpoint/precision lens use after held-out validation",
    )
    collect.add_argument(
        "--mismatch-validation",
        help="required note describing held-out validation for approximate cross-precision lens reuse",
    )

    debug = sub.add_parser("debug", help="run the CPU fixture or one local-model smoke example")
    debug.add_argument("--fixture", action="store_true", help="run the deterministic CPU plumbing fixture")
    debug.add_argument("--model", help="local model for a production-path one-example debug")
    debug.add_argument("--lens")
    debug.add_argument("--input")
    debug.add_argument("--layers", type=_layers)
    debug.add_argument("--output", default="runs/debug")
    debug.add_argument("--max-new-tokens", type=int, default=8)
    return parser


def _inspect(args: argparse.Namespace) -> None:
    config = read_config(args.model)
    values = text_config_values(config)
    if args.config_only:
        _print({
            "model_path": str(Path(args.model).expanduser().resolve()),
            "checkpoint_config": values,
            "module_mapping_verified": False,
            "note": "use inspect without --config-only to load weights and verify the actual module tree",
        })
        return
    adapter = load_local_model(args.model, device_map=args.device_map)
    layers = list(range(adapter.n_layers)) if args.layers is None else resolve_layers(args.layers, adapter.n_layers)
    _print({
        "model_class": type(adapter.hf_model).__name__,
        "checkpoint_config": values,
        "processor_class": type(adapter.processor).__name__,
        "text_stack_path": adapter.block_path,
        "n_layers": adapter.n_layers,
        "hidden_size": adapter.d_model,
        "quantization_and_precision": adapter.quantization,
        "layers": layer_mapping(adapter.block_path, layers),
    })


def _fit(args: argparse.Namespace) -> None:
    adapter = load_local_model(args.model, device_map=args.device_map)
    layers = resolve_layers(args.layers, adapter.n_layers)
    final = adapter.n_layers - 1
    invalid_sources = [index for index in layers if index >= final]
    if invalid_sources:
        raise ValueError(
            f"requested source layers {invalid_sources} are not before target block {final}; "
            f"valid sources for this target are [0, {final - 1}]"
        )
    prompts, corpus_sha = load_calibration_prompts(args.calibration_text)
    lengths = [int(adapter.lens_model.encode(prompt, max_length=args.max_seq_len).shape[1]) for prompt in prompts]
    too_short = [index for index, size in enumerate(lengths, start=1) if size <= args.skip_first + 1]
    if too_short:
        raise ValueError(f"calibration prompts {too_short[:8]} are too short after truncation; increase text or lower --skip-first")
    estimate = estimate_calibration(len(prompts), adapter.d_model, len(layers), args.dim_batch)
    print("Selected complete-block boundaries:")
    _print(layer_mapping(adapter.block_path, layers))
    print("Calibration estimate before fitting:")
    _print({
        **estimate,
        "approximate_dense_artifact_gib": estimate["dense_fp32_artifact_bytes"] / (1024**3),
        "approximate_peak_lens_disk_gib": estimate["peak_lens_disk_bytes_approx"] / (1024**3),
    })
    artifact = fit_artifact(
        adapter, prompts, layers, args.output,
        dim_batch=args.dim_batch, max_seq_len=args.max_seq_len, skip_first=args.skip_first,
        corpus_name=str(Path(args.calibration_text).expanduser().resolve()), corpus_sha256=corpus_sha,
    )
    _print({"lens_artifact": str(artifact.path), "sha256": artifact.sha256, "source_layers": artifact.manifest["source_layers"]})


def _check_lens(args: argparse.Namespace) -> None:
    artifact = load_artifact(args.lens)
    _, corpus_sha = load_calibration_prompts(args.calibration_text)
    if list(artifact.manifest.get("source_layers", [])) != args.layers:
        raise ValueError(
            f"existing lens layers {artifact.manifest.get('source_layers')} do not match requested {args.layers}; "
            "choose a new output directory or fit the requested layers"
        )
    saved_sha = artifact.manifest.get("calibration", {}).get("corpus_sha256")
    if saved_sha != corpus_sha:
        raise ValueError(
            "existing lens was fit on different calibration text; choose a new output directory or use the original corpus"
        )
    _print({"compatible_calibration_corpus": True, "lens_sha256": artifact.sha256, "manifest": artifact.manifest})


def _collect(args: argparse.Namespace) -> None:
    if args.mismatch_validation and not args.allow_calibration_model_mismatch:
        raise ValueError("--mismatch-validation requires --allow-calibration-model-mismatch")
    adapter = load_local_model(args.model, device_map=args.device_map)
    layers = resolve_layers(args.layers, adapter.n_layers)
    artifact = load_artifact(args.lens)
    records = read_input(args.input, args.format)
    if args.max_examples is not None:
        if args.max_examples < 1:
            raise ValueError("--max-examples must be positive")
        records = records[:args.max_examples]
    estimate = len(records) * args.max_new_tokens * len(layers) * adapter.d_model * 4
    print("Collection selection:")
    _print({
        "layers": layer_mapping(adapter.block_path, layers),
        "examples": len(records),
        "max_new_tokens": args.max_new_tokens,
        "approximate_trajectory_storage_bytes": estimate,
        "approximate_trajectory_storage_gib": estimate / (1024**3),
        "quantization_and_precision": adapter.quantization,
    })
    summary = collect_examples(
        adapter, artifact, records, args.output,
        layer_indices=layers,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        seed=args.seed,
        use_cache=not args.no_cache,
        allow_numerical_mismatch=args.allow_calibration_model_mismatch,
        mismatch_validation_note=args.mismatch_validation,
    )
    _print({"output_dir": str(Path(args.output).expanduser().resolve()), "summary": summary})


def _debug(args: argparse.Namespace) -> None:
    if args.fixture:
        result = run_fixture_debug(args.output, max_new_tokens=args.max_new_tokens)
        _print(result)
        return
    missing = [name for name in ("model", "lens", "input", "layers") if getattr(args, name) is None]
    if missing:
        raise ValueError(f"production debug needs --model, --lens, --input, and --layers; missing {missing}. Use --fixture for CPU checks.")
    result = run_model_debug(
        args.model, args.lens, args.input, args.output, args.layers,
        max_new_tokens=args.max_new_tokens,
    )
    _print(result)


def _prepare_simpleqa(args: argparse.Namespace) -> None:
    _print(prepare_simpleqa(args.output, count=args.count, seed=args.seed))


def _judge(args: argparse.Namespace) -> None:
    _print(judge_run(args.run_dir, model=args.model))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        _inspect(args)
    elif args.command == "fit-lens":
        _fit(args)
    elif args.command == "check-lens":
        _check_lens(args)
    elif args.command == "prepare-simpleqa":
        _prepare_simpleqa(args)
    elif args.command == "collect":
        _collect(args)
    elif args.command == "debug":
        _debug(args)
    elif args.command == "judge":
        _judge(args)


if __name__ == "__main__":
    main()
