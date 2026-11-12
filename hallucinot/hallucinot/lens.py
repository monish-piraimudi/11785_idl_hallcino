"""Anthropic-reference Jacobian fitting, provenance, and readout helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .model import (
    LoadedModel,
    adapter_model_identity,
    environment_report,
    layer_mapping,
    read_config,
    text_config_values,
)

UPSTREAM_JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
ARTIFACT_SCHEMA = "hallucinot.jacobian-lens.v1"


@dataclass
class LensArtifact:
    matrices: dict[int, torch.Tensor]
    manifest: dict[str, Any]
    path: Path | None = None
    sha256: str | None = None


def estimate_calibration(n_prompts: int, d_model: int, n_layers: int, dim_batch: int) -> dict[str, int | str]:
    if min(n_prompts, d_model, n_layers, dim_batch) <= 0:
        raise ValueError("prompt count, model dimensions, layer count, and dim_batch must be positive")
    passes = math.ceil(d_model / dim_batch)
    return {
        "prompts": n_prompts,
        "forward_passes": n_prompts,
        "backward_passes": n_prompts * passes,
        "dimension_batches_per_prompt": passes,
        "dense_fp32_artifact_bytes": n_layers * d_model * d_model * 4,
        "temporary_fit_checkpoint_bytes": n_layers * d_model * d_model * 4,
        "peak_lens_disk_bytes_approx": n_layers * d_model * d_model * 8,
        "lens_matrix_shape_per_layer": f"[{d_model}, {d_model}]",
        "cost_note": "backward FLOPs scale with output dimension; dim_batch increases peak activation memory, not total backward FLOPs",
    }


def preflight_gradient(adapter: LoadedModel, text: str, *, max_seq_len: int, skip_first: int) -> dict[str, Any]:
    """Prove the frozen checkpoint backend differentiates through block outputs."""
    import jlens

    input_ids = adapter.lens_model.encode(text, max_length=max_seq_len)
    if input_ids.shape[1] <= skip_first + 1:
        raise ValueError(
            f"gradient preflight needs more than {skip_first + 1} tokens; got {input_ids.shape[1]}. "
            "Use longer calibration text or lower --skip-first."
        )
    source_index = 0
    target_index = adapter.n_layers - 1
    activations: dict[str, torch.Tensor] = {}

    def source_hook(_module: Any, _inputs: Any, output: Any) -> None:
        value = output if torch.is_tensor(output) else output[0]
        value.requires_grad_(True)
        activations["source"] = value

    def target_hook(_module: Any, _inputs: Any, output: Any) -> None:
        activations["target"] = output if torch.is_tensor(output) else output[0]

    handles = [
        adapter.blocks[source_index].register_forward_hook(source_hook),
        adapter.blocks[target_index].register_forward_hook(target_hook),
    ]
    try:
        for parameter in adapter.hf_model.parameters():
            if parameter.requires_grad:
                raise RuntimeError("base weights must remain frozen during Jacobian calibration")
        with torch.enable_grad():
            adapter.lens_model.forward(input_ids)
            if activations.get("source") is None or activations.get("target") is None:
                raise RuntimeError("gradient preflight did not observe both selected block boundaries")
            gradient = torch.autograd.grad(
                activations["target"].float().sum(), activations["source"], allow_unused=True
            )[0]
            if gradient is None:
                raise RuntimeError("no gradient flowed from final residual to the source residual")
            if not torch.isfinite(gradient).all():
                raise RuntimeError("gradient preflight produced NaN or infinity")
            norm = float(torch.linalg.vector_norm(gradient.float()).item())
            if norm == 0.0:
                raise RuntimeError("gradient preflight produced an all-zero gradient")
        return {
            "source_layer": source_index,
            "target_layer": target_index,
            "sequence_tokens": int(input_ids.shape[1]),
            "gradient_norm": norm,
            "gradient_dtype": str(gradient.dtype),
            "gradient_device": str(gradient.device),
        }
    except Exception as exc:
        quant = adapter.quantization.get("reported_method") or "unquantized checkpoint"
        raise RuntimeError(
            f"Autograd preflight failed for {quant}. This backend is not verified for frozen-weight "
            "activation-gradient calibration. No precision fallback was attempted. Inspect the quantizer's "
            "backward support or explicitly validate a separate differentiable calibration checkpoint."
        ) from exc
    finally:
        for handle in handles:
            handle.remove()


def fit_artifact(
    adapter: LoadedModel,
    prompts: list[str],
    layer_indices: list[int],
    output_path: str | Path,
    *,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = 16,
    corpus_name: str,
    corpus_sha256: str,
) -> LensArtifact:
    """Fit through the pinned `jlens.fit` estimator and persist dense float32 J."""
    import jlens
    from .model import environment_report

    if not prompts:
        raise ValueError("calibration file has no non-empty prompts")
    if dim_batch < 1 or max_seq_len < 2 or skip_first < 0:
        raise ValueError("dim_batch must be positive, max_seq_len >= 2, and skip_first >= 0")
    if len(layer_indices) != len(set(layer_indices)):
        raise ValueError("calibration layer indices must be unique")
    final_index = adapter.n_layers - 1
    unavailable = [index for index in layer_indices if index < 0 or index >= final_index]
    if unavailable:
        raise ValueError(
            f"requested source layers {unavailable} cannot be calibrated to target block {final_index}; "
            "the reference target is the final decoder block output. Choose zero-based source layers "
            f"in [0, {final_index - 1}]. No requested layer was omitted."
        )

    preflight = preflight_gradient(adapter, prompts[0], max_seq_len=max_seq_len, skip_first=skip_first)
    fingerprint = adapter_model_identity(adapter)
    prompt_bytes = json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = out.with_name(f".{out.stem}.{fingerprint['checkpoint_sha256'][:10]}.{prompt_hash[:10]}.fit.pt")

    fitted = jlens.fit(
        adapter.lens_model,
        prompts,
        source_layers=layer_indices,
        target_layer=final_index,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
        checkpoint_path=str(checkpoint),
        checkpoint_every=1,
        resume=True,
    )
    architecture = (
        {"fixture": "tiny"}
        if adapter.quantization.get("fixture")
        else text_config_values(read_config(adapter.model_path))
    )
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "estimator": "Anthropic jlens.fit; source-position mean of summed eligible target-position cotangents",
        "upstream_repository": "https://github.com/anthropics/jacobian-lens",
        "upstream_commit": UPSTREAM_JLENS_COMMIT,
        "target_layer_index": final_index,
        "target_boundary": "output of final text decoder block before text model final RMSNorm",
        "source_layers": layer_indices,
        "layer_mapping": layer_mapping(adapter.block_path, layer_indices),
        "d_model": adapter.d_model,
        "n_layers": adapter.n_layers,
        "activation_boundary": "output of each complete text decoder block before final text RMSNorm",
        "text_stack_path": adapter.block_path,
        "model_architecture": architecture,
        "matrix_orientation": "J[layer] has [target_dimension, source_dimension]; transport is residual @ J.T",
        "readout": "transported vector -> checkpoint text RMSNorm -> checkpoint lm_head -> checkpoint logit softcap if configured",
        "calibration": {
            "corpus_name": corpus_name,
            "corpus_sha256": corpus_sha256,
            "prompt_list_sha256": prompt_hash,
            "n_prompts": len(prompts),
            "dim_batch": dim_batch,
            "max_seq_len": max_seq_len,
            "skip_first_positions": skip_first,
            "preflight_gradient": preflight,
            "estimated_cost": estimate_calibration(len(prompts), adapter.d_model, len(layer_indices), dim_batch),
        },
        "model": fingerprint,
        "runtime": environment_report(),
        "quantization": adapter.quantization,
        "feature_representation": "dense float32 J_l h_l transport, not a sparse J-space decomposition",
    }
    save_artifact(out, {int(k): value.detach().float().cpu() for k, value in fitted.jacobians.items()}, manifest)
    if checkpoint.exists():
        checkpoint.unlink()
    return load_artifact(out)


def save_artifact(path: str | Path, matrices: dict[int, torch.Tensor], manifest: dict[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {f"J_{layer:04d}": matrix.detach().cpu().float().contiguous() for layer, matrix in matrices.items()}
    temp = path.with_name(f".{path.name}.tmp")
    save_file(tensors, str(temp), metadata={"manifest_json": json.dumps(manifest, sort_keys=True)})
    os.replace(temp, path)


def load_artifact(path: str | Path) -> LensArtifact:
    path = Path(path).expanduser().resolve()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    try:
        manifest = json.loads(metadata["manifest_json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} has no valid hallucinot lens manifest") from exc
    if manifest.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError(f"unsupported lens artifact schema: {manifest.get('schema')!r}")
    tensors = load_file(str(path), device="cpu")
    matrices = {int(key[2:]): value.float() for key, value in tensors.items() if key.startswith("J_")}
    if sorted(matrices) != sorted(manifest.get("source_layers", [])):
        raise ValueError("lens tensor keys disagree with manifest source_layers")
    d_model = int(manifest["d_model"])
    for layer, matrix in matrices.items():
        if tuple(matrix.shape) != (d_model, d_model):
            raise ValueError(f"layer {layer} matrix has shape {tuple(matrix.shape)}, expected {(d_model, d_model)}")
        if not torch.isfinite(matrix).all():
            raise ValueError(f"layer {layer} matrix contains NaN or infinity")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    sha = digest.hexdigest()
    return LensArtifact(matrices, manifest, path, sha)


def validate_artifact(
    adapter: LoadedModel,
    artifact: LensArtifact,
    layers: list[int],
    *,
    allow_numerical_mismatch: bool = False,
    validation_note: str | None = None,
    fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    fingerprint = fingerprint or adapter_model_identity(adapter)
    saved = artifact.manifest["model"]
    weight_match = fingerprint["checkpoint_sha256"] == saved["checkpoint_sha256"]
    if artifact.manifest["d_model"] != adapter.d_model or artifact.manifest["n_layers"] != adapter.n_layers:
        raise ValueError("lens dimensions/layer count do not match the loaded model")
    architecture = (
        {"fixture": "tiny"}
        if adapter.quantization.get("fixture")
        else text_config_values(read_config(adapter.model_path))
    )
    saved_arch = artifact.manifest.get("model_architecture", {})
    architecture_keys = ("model_type", "text_model_type", "num_hidden_layers", "hidden_size", "vocab_size")
    if any(saved_arch.get(key) != architecture.get(key) for key in architecture_keys):
        raise ValueError("lens and collection model architectures differ; blockwise transports are not compatible")
    if artifact.manifest.get("text_stack_path") != adapter.block_path:
        raise ValueError("lens and collection text-stack module paths differ")
    missing = sorted(set(layers) - set(artifact.matrices))
    if missing:
        raise ValueError(f"lens artifact has no matrices for requested layers {missing}")
    saved_runtime = artifact.manifest["runtime"]
    from .model import environment_report

    now = environment_report()
    for key in ("torch", "transformers", "accelerate", "huggingface_hub", "tokenizers", "jlens"):
        if saved_runtime.get(key) != now.get(key):
            raise ValueError(f"lens runtime {key}={saved_runtime.get(key)} differs from current {now.get(key)}")
    quant_keys = ("checkpoint_config", "reported_method", "observed_parameter_numel_by_dtype")
    saved_quant = artifact.manifest.get("quantization", {})
    quant_match = all(saved_quant.get(key) == adapter.quantization.get(key) for key in quant_keys)
    backend_match = saved_runtime.get("quantization_backends", {}) == now.get("quantization_backends", {})
    if weight_match and quant_match and backend_match:
        return None
    if not allow_numerical_mismatch:
        reason = "checkpoint weights differ" if not weight_match else (
            "runtime parameter/quantization precision differs" if not quant_match else
            "quantization backend versions differ"
        )
        raise ValueError(
            f"{reason} between calibration and collection. Use an exactly matching model, or explicitly opt in "
            "to this approximation only after validating it on held-out prompts."
        )
    if not validation_note or not validation_note.strip():
        raise ValueError("an explicit --mismatch-validation note is required for approximate cross-precision reuse")
    return {
        "approximation": "calibration transport fitted on a numerically different checkpoint/runtime",
        "calibration_checkpoint_sha256": saved["checkpoint_sha256"],
        "collection_checkpoint_sha256": fingerprint["checkpoint_sha256"],
        "calibration_quantization": saved_quant,
        "collection_quantization": adapter.quantization,
        "user_validation_note": validation_note.strip(),
        "validation_required": True,
    }


def transport(matrix: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    """Reference orientation: `J @ h`, represented batchwise as `h @ J.T`."""
    if matrix.ndim != 2 or activation.shape[-1] != matrix.shape[1]:
        raise ValueError(f"incompatible J/activation dimensions: {tuple(matrix.shape)}, {tuple(activation.shape)}")
    result = activation.float() @ matrix.to(activation.device, dtype=torch.float32).T
    if not torch.isfinite(result).all():
        raise FloatingPointError("J transport produced NaN or infinity")
    return result


def decoded_readout(adapter: LoadedModel, transported: torch.Tensor, *, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Use jlens' HF adapter to apply final norm, head, and configured softcap once."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    logits = adapter.lens_model.unembed(transported).float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("vocabulary readout produced NaN or infinity")
    values, ids = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
    return ids.cpu(), values.cpu()
