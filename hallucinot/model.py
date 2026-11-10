"""Local Hugging Face loading and Gemma text-stack discovery."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoadedModel:
    hf_model: Any
    processor: Any
    tokenizer: Any
    lens_model: Any
    blocks: Any
    block_path: str
    n_layers: int
    d_model: int
    model_path: Path
    quantization: dict[str, Any]
    model_identity: dict[str, Any] | None = None


def read_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve()
    config_file = path / "config.json"
    if not path.is_dir() or not config_file.is_file():
        raise FileNotFoundError(f"expected a local Hugging Face checkpoint directory with config.json: {path}")
    try:
        return json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid model config {config_file}: {exc}") from exc


def text_config_values(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config", config)
    return {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "text_model_type": text.get("model_type"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "hidden_size": text.get("hidden_size"),
        "vocab_size": text.get("vocab_size"),
        "quantization_config": config.get("quantization_config"),
        "vision_config_present": config.get("vision_config") is not None,
        "audio_config_present": config.get("audio_config") is not None,
    }


def load_local_model(model_path: str | Path, *, device_map: str = "auto") -> LoadedModel:
    """Load only local files; preserve the checkpoint's configured quantization."""
    import torch
    import jlens
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    path = Path(model_path).expanduser().resolve()
    config = read_config(path)
    if config.get("model_type") == "gemma4_unified_assistant":
        raise ValueError(
            "this path is the separate Gemma 4 MTP assistant/draft model; pass the target Gemma 4 12B checkpoint"
        )
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise TypeError(f"{type(processor).__name__} has no tokenizer; text generation requires one")
    model = AutoModelForMultimodalLM.from_pretrained(
        path,
        dtype="auto",
        device_map=device_map,
        local_files_only=True,
    )
    model.eval()
    lens_model = jlens.from_hf(model, tokenizer, force_bos=True)
    blocks = lens_model.layers
    text_conf = model.config.get_text_config()
    n_layers = int(text_conf.num_hidden_layers)
    d_model = int(text_conf.hidden_size)
    if len(blocks) != n_layers:
        raise RuntimeError(f"text config has {n_layers} layers but adapter found {len(blocks)}")
    layout = lens_model.layout
    block_path = f"{layout.path}.{layout.layers}"
    if not all(isinstance(parameter, torch.nn.Parameter) for parameter in model.parameters()):
        raise TypeError("loaded model contains an unexpected parameter object")
    return LoadedModel(
        model, processor, tokenizer, lens_model, blocks, block_path,
        n_layers, d_model, path, _quantization_report(model, config),
    )


def _quantization_report(model: Any, config: dict[str, Any]) -> dict[str, Any]:
    quant = config.get("quantization_config")
    dtypes: Counter[str] = Counter()
    parameter_count = 0
    for parameter in model.parameters():
        dtypes[str(parameter.dtype)] += parameter.numel()
        parameter_count += parameter.numel()
    return {
        "checkpoint_config": quant,
        "reported_method": (quant or {}).get("quant_method") if isinstance(quant, dict) else None,
        "observed_parameter_numel_by_dtype": dict(dtypes),
        "parameter_count_including_tied_references": parameter_count,
        "device_map": getattr(model, "hf_device_map", None),
    }


def resolve_layers(layer_indices: list[int], n_layers: int) -> list[int]:
    if not layer_indices:
        raise ValueError("provide at least one zero-based transformer block index")
    duplicates = sorted({index for index in layer_indices if layer_indices.count(index) > 1})
    if duplicates:
        raise ValueError(f"duplicate block indices are ambiguous: {duplicates}")
    invalid = [index for index in layer_indices if index < 0 or index >= n_layers]
    if invalid:
        raise ValueError(f"block indices {invalid} are outside [0, {n_layers - 1}]")
    return list(layer_indices)


def layer_mapping(block_path: str, layer_indices: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "module_path": f"{block_path}.{index}",
            "activation_boundary": "forward output of the complete text decoder block; before final text RMSNorm",
        }
        for index in layer_indices
    ]


def checkpoint_fingerprint(model_path: str | Path) -> dict[str, Any]:
    """Hash config and every weight shard so lens reuse is tied to exact files."""
    path = Path(model_path).expanduser().resolve()
    config_file = path / "config.json"
    config_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
    candidates = sorted(
        file for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth")
        for file in path.rglob(pattern)
    )
    if not candidates:
        raise FileNotFoundError(f"no local model weight files found under {path}")
    tokenizer_names = {
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
        "spiece.model", "tokenizer.model", "chat_template.jinja",
        "processor_config.json",
    }
    tokenizer_files = sorted(
        file for file in path.rglob("*")
        if file.is_file() and file.name in tokenizer_names
    )
    files = []
    tokenizer_entries = []
    combined = hashlib.sha256(config_hash.encode("ascii"))
    for file in candidates:
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        sha = digest.hexdigest()
        relative = file.relative_to(path).as_posix()
        files.append({"path": relative, "bytes": file.stat().st_size, "sha256": sha})
        combined.update(b"weight\0" + relative.encode("utf-8"))
        combined.update(sha.encode("ascii"))
    for file in tokenizer_files:
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        sha = digest.hexdigest()
        relative = file.relative_to(path).as_posix()
        tokenizer_entries.append({"path": relative, "bytes": file.stat().st_size, "sha256": sha})
        combined.update(b"tokenizer\0" + relative.encode("utf-8"))
        combined.update(sha.encode("ascii"))
    return {
        "model_path": str(path),
        "config_sha256": config_hash,
        "weight_files": files,
        "tokenizer_and_processor_files": tokenizer_entries,
        "checkpoint_sha256": combined.hexdigest(),
    }


def adapter_model_identity(adapter: LoadedModel) -> dict[str, Any]:
    """Use a fixed transparent identifier for the non-scientific CPU fixture."""
    if adapter.model_identity is not None:
        return adapter.model_identity
    if adapter.quantization.get("fixture"):
        adapter.model_identity = {
            "model_id": "hallucinot-tiny-deterministic-transformer-v1",
            "checkpoint_sha256": "tiny-fixture-is-not-a-real-checkpoint",
        }
    else:
        adapter.model_identity = checkpoint_fingerprint(adapter.model_path)
    return adapter.model_identity


class BlockCapture:
    """Temporary hooks on full-block outputs; each call replaces old captures."""

    def __init__(self, blocks: Any, layer_indices: list[int], *, last_token_only: bool = False):
        self.blocks = blocks
        self.layer_indices = list(layer_indices)
        self.last_token_only = last_token_only
        self.activations: dict[int, Any] = {}
        self.output_shapes: dict[int, list[int]] = {}
        self.handles: list[Any] = []

    @staticmethod
    def _tensor(output: Any) -> Any:
        import torch

        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return output[0]
        raise TypeError(f"block output does not begin with a tensor: {type(output).__name__}")

    def __enter__(self) -> "BlockCapture":
        try:
            for index in self.layer_indices:
                self.handles.append(
                    self.blocks[index].register_forward_hook(
                        lambda _module, _inputs, output, i=index: self._record(i, output)
                    )
                )
        except Exception:
            self.close()
            raise
        return self

    def _record(self, index: int, output: Any) -> None:
        tensor = self._tensor(output)
        self.output_shapes[index] = list(tensor.shape)
        self.activations[index] = tensor[:, -1:, :].clone() if self.last_token_only else tensor

    def clear(self) -> None:
        self.activations.clear()
        self.output_shapes.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def environment_report() -> dict[str, Any]:
    import importlib.metadata
    import platform
    import jlens
    import numpy
    import torch
    import transformers
    import safetensors

    cuda = torch.cuda.is_available()
    devices = []
    if cuda:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            })
    optional_backend_versions = {}
    for package in ("torchao", "bitsandbytes", "compressed-tensors", "optimum-quanto", "transformer-engine"):
        try:
            optional_backend_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "numpy": numpy.__version__,
        "accelerate": importlib.metadata.version("accelerate"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "jlens": importlib.metadata.version("jlens"),
        "quantization_backends": optional_backend_versions,
        "cuda_available": cuda,
        "torch_cuda_version": torch.version.cuda,
        "devices": devices,
    }
