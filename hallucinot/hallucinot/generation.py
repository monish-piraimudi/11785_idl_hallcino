"""Readable greedy KV-cache decoding with prediction-step activation alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .lens import LensArtifact, decoded_readout, transport
from .model import BlockCapture, LoadedModel
from .prompts import chat_messages


@dataclass
class GenerationResult:
    prompt_text: str
    prompt_input_ids: list[int]
    generated_token_ids: list[int]
    generated_text: str
    generated_text_raw: str
    answer_text: str
    finish_reason: str
    layer_indices: list[int]
    j_transport: torch.Tensor
    readout_top_ids: torch.Tensor
    readout_top_logits: torch.Tensor
    generation_step: torch.Tensor
    source_activation_position: torch.Tensor
    source_token_id: torch.Tensor
    source_is_prompt: torch.Tensor
    source_attention_mask: torch.Tensor
    is_special_token: torch.Tensor
    is_control_token: torch.Tensor
    is_eos_token: torch.Tensor
    is_end_of_turn_token: torch.Tensor
    is_reasoning_token: torch.Tensor
    is_answer_token: torch.Tensor
    token_pieces: list[str]
    readout_top_pieces: list[list[list[str]]]
    activation_dtypes: dict[str, str]
    activation_observations: dict[str, list[dict[str, Any]]]
    kv_cache_class: str | None
    kv_cache_dtypes: list[str]


def encode_question(adapter: LoadedModel, question: str) -> tuple[dict[str, Any], str]:
    """Apply the checkpoint's chat template to question-only messages."""
    encoded = adapter.processor.apply_chat_template(
        chat_messages(question),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if "input_ids" not in encoded or encoded["input_ids"].ndim != 2:
        raise RuntimeError("processor chat template did not return [batch, tokens] input_ids")
    device = adapter.lens_model.input_device
    encoded = encoded.to(device)
    ids = encoded["input_ids"][0].tolist()
    visible = adapter.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return dict(encoded), visible


def _as_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def _special_token_ids(adapter: LoadedModel) -> tuple[set[int], set[int]]:
    tokenizer = adapter.tokenizer
    generation = getattr(adapter.hf_model, "generation_config", None)
    eos = _as_ids(getattr(generation, "eos_token_id", None)) | _as_ids(tokenizer.eos_token_id)
    end_turn: set[int] = set()
    for token in getattr(tokenizer, "all_special_tokens", []):
        lowered = token.lower()
        if "end_of_turn" in lowered or lowered in {"<|turn|>", "<turn|>"}:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
                end_turn.add(int(token_id))
    return eos, end_turn


def _piece(adapter: LoadedModel, token_id: int) -> str:
    return adapter.tokenizer.decode(
        [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def _cache_dtypes(cache: Any) -> tuple[str | None, list[str]]:
    if cache is None:
        return None, ["not returned"]
    found: set[str] = set()
    seen: set[int] = set()

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 5 or id(value) in seen:
            return
        seen.add(id(value))
        if torch.is_tensor(value):
            found.add(str(value.dtype))
        elif isinstance(value, dict):
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child, depth + 1)
        else:
            for name in ("key_cache", "value_cache", "keys", "values", "layers"):
                child = getattr(value, name, None)
                if child is not None:
                    visit(child, depth + 1)

    visit(cache)
    return type(cache).__name__, sorted(found) if found else ["cache tensors not exposed by this backend"]


def _channel_masks(adapter: LoadedModel, token_ids: list[int]) -> tuple[list[str], list[bool], list[bool], list[bool]]:
    special_ids = {int(item) for item in getattr(adapter.tokenizer, "all_special_ids", [])}
    pieces = [_piece(adapter, token_id) for token_id in token_ids]
    reasoning: list[bool] = []
    answer: list[bool] = []
    controls: list[bool] = []
    in_thought = False
    awaiting_channel = False
    for token_id, piece in zip(token_ids, pieces, strict=True):
        lowered = piece.lower()
        if "<channel|>" in lowered or "</channel>" in lowered:
            in_thought = False
            awaiting_channel = False
            controls.append(True)
            reasoning.append(False)
            answer.append(False)
            continue
        if "<|channel>" in lowered or "<|channel|>" in lowered:
            channel_tail = lowered.split("<|channel>", 1)[-1].strip()
            channel_tail = channel_tail.split("<|channel|>", 1)[-1].strip()
            if channel_tail:
                in_thought = any(name in channel_tail for name in ("thought", "analysis", "reasoning"))
                awaiting_channel = False
            else:
                awaiting_channel = True
            controls.append(True)
            reasoning.append(False)
            answer.append(False)
            continue
        if awaiting_channel and lowered.strip():
            channel_name = lowered.strip()
            in_thought = any(name in channel_name for name in ("thought", "analysis", "reasoning"))
            awaiting_channel = False
            controls.append(True)
            reasoning.append(False)
            answer.append(False)
            continue
        if awaiting_channel:
            controls.append(True)
            reasoning.append(False)
            answer.append(False)
            continue
        is_special = token_id in special_ids
        controls.append(is_special)
        reasoning.append(in_thought)
        answer.append(not in_thought and not is_special)
    return pieces, reasoning, answer, controls


def generate_with_trajectory(
    adapter: LoadedModel,
    question: str,
    layer_indices: list[int],
    *,
    artifact: LensArtifact | None,
    max_new_tokens: int = 32,
    top_k: int = 5,
    seed: int = 0,
    capture: bool = True,
    use_cache: bool = True,
) -> GenerationResult:
    """Capture the residual used to predict each token, including terminal tokens."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if capture and artifact is None:
        raise ValueError("capture requires a compatible lens artifact")
    if artifact is not None:
        missing = sorted(set(layer_indices) - set(artifact.matrices))
        if missing:
            raise ValueError(f"lens artifact has no matrices for layers {missing}")
    torch.manual_seed(seed)
    reset_state = getattr(adapter.hf_model, "reset_generation_state", None)
    if reset_state is not None:
        reset_state()
    encoded, prompt_text = encode_question(adapter, question)
    prompt_ids = encoded["input_ids"]
    full_ids = prompt_ids
    attention = encoded.get("attention_mask", torch.ones_like(prompt_ids))
    eos_ids, end_turn_ids = _special_token_ids(adapter)
    step_transports: list[torch.Tensor] = []
    step_top_ids: list[torch.Tensor] = []
    step_top_logits: list[torch.Tensor] = []
    positions: list[int] = []
    source_ids: list[int] = []
    dtypes: dict[str, str] = {}
    observations: dict[str, list[dict[str, Any]]] = {str(layer): [] for layer in layer_indices}
    matrix_cache: dict[int, torch.Tensor] = {}
    finish_reason = "token_budget"
    generated: list[int] = []

    hooks = BlockCapture(adapter.blocks, layer_indices, last_token_only=True) if capture else None
    context = hooks if hooks is not None else _NullContext()
    with context:
        with torch.no_grad():
            initial = dict(encoded)
            gemma_unified = getattr(adapter.hf_model.config, "model_type", None) == "gemma4_unified"
            if gemma_unified:
                initial["logits_to_keep"] = 1
            outputs = adapter.hf_model(**initial, use_cache=use_cache, return_dict=True)
            cache_class, cache_dtypes = _cache_dtypes(outputs.past_key_values)
            for step in range(max_new_tokens):
                if capture:
                    # In cached mode this call was made at the end of the previous iteration.
                    if not hooks.activations:
                        raise RuntimeError(f"no block activation captured at generation step {step}")
                    current_layers = {}
                    for layer in layer_indices:
                        captured = hooks.activations[layer]
                        if captured.ndim != 3 or captured.shape[:2] != (1, 1):
                            raise RuntimeError(
                                f"layer {layer} last-token hook saw shape {tuple(captured.shape)}, expected [1, 1, hidden]"
                            )
                        current = captured[0, -1].detach()
                        dtypes[str(layer)] = str(current.dtype)
                        observations[str(layer)].append({
                            "shape": hooks.output_shapes[layer],
                            "dtype": str(captured.dtype),
                            "device": str(captured.device),
                        })
                        current_layers[layer] = current.float().unsqueeze(0)
                        if layer not in matrix_cache:
                            matrix_cache[layer] = artifact.matrices[layer].to(current.device)

                logits = outputs.logits[0, -1].float()
                token_id = int(torch.argmax(logits).item())
                generated.append(token_id)
                positions.append(int(prompt_ids.shape[1]) + step - 1)
                source_ids.append(int(full_ids[0, -1].item()))

                if capture:
                    transported = []
                    top_ids = []
                    top_values = []
                    for layer in layer_indices:
                        feature = transport(matrix_cache[layer], current_layers[layer])
                        token_indices, token_values = decoded_readout(adapter, feature, top_k=top_k)
                        transported.append(feature[0].cpu())
                        top_ids.append(token_indices[0])
                        top_values.append(token_values[0])
                    step_transports.append(torch.stack(transported))
                    step_top_ids.append(torch.stack(top_ids))
                    step_top_logits.append(torch.stack(top_values))

                if token_id in end_turn_ids:
                    finish_reason = "end_of_turn"
                    break
                if token_id in eos_ids:
                    finish_reason = "eos"
                    break
                if step + 1 == max_new_tokens:
                    break

                next_token = torch.tensor([[token_id]], dtype=prompt_ids.dtype, device=prompt_ids.device)
                full_ids = torch.cat((full_ids, next_token), dim=1)
                attention = torch.cat((attention, torch.ones_like(next_token)), dim=1)
                if use_cache:
                    call_inputs = {
                        "input_ids": next_token,
                        "attention_mask": attention,
                        "past_key_values": outputs.past_key_values,
                    }
                    shared_kv_states = getattr(outputs, "shared_kv_states", None)
                    if shared_kv_states is not None:
                        call_inputs["shared_kv_states"] = shared_kv_states
                else:
                    call_inputs = {"input_ids": full_ids, "attention_mask": attention}
                if gemma_unified:
                    call_inputs["logits_to_keep"] = 1
                if capture:
                    hooks.clear()
                outputs = adapter.hf_model(**call_inputs, use_cache=use_cache, return_dict=True)
                cache_class, cache_dtypes = _cache_dtypes(outputs.past_key_values)

    pieces, reasoning_mask, answer_mask, control_mask = _channel_masks(adapter, generated)
    special_ids = {int(item) for item in getattr(adapter.tokenizer, "all_special_ids", [])}
    raw_text = adapter.tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    visible_text = adapter.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    answer_ids = [token for token, is_answer in zip(generated, answer_mask, strict=True) if is_answer]
    answer_text = adapter.tokenizer.decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    readout_pieces = [
        [
            [_piece(adapter, int(token_id)) for token_id in layer_ids.tolist()]
            for layer_ids in step_ids_by_layer
        ]
        for step_ids_by_layer in step_top_ids
    ]

    def tensor_rows(rows: list[torch.Tensor], shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.stack(rows) if rows else torch.empty(shape, dtype=dtype)

    n_layers = len(layer_indices)
    d_model = adapter.d_model
    text_config = adapter.hf_model.config.get_text_config()
    vocab_size = int(getattr(text_config, "vocab_size", adapter.tokenizer.vocab_size))
    readout_k = min(top_k, vocab_size)
    return GenerationResult(
        prompt_text=prompt_text,
        prompt_input_ids=prompt_ids[0].tolist(),
        generated_token_ids=generated,
        generated_text=visible_text,
        generated_text_raw=raw_text,
        answer_text=answer_text,
        finish_reason=finish_reason,
        layer_indices=list(layer_indices),
        j_transport=tensor_rows(step_transports, (0, n_layers, d_model)),
        readout_top_ids=tensor_rows(step_top_ids, (0, n_layers, readout_k), torch.int64).long(),
        readout_top_logits=tensor_rows(step_top_logits, (0, n_layers, readout_k)),
        generation_step=torch.arange(len(generated), dtype=torch.int64),
        source_activation_position=torch.tensor(positions, dtype=torch.int64),
        source_token_id=torch.tensor(source_ids, dtype=torch.int64),
        source_is_prompt=torch.tensor([step == 0 for step in range(len(generated))], dtype=torch.bool),
        source_attention_mask=torch.ones(len(generated), dtype=torch.bool),
        is_special_token=torch.tensor([token in special_ids for token in generated], dtype=torch.bool),
        is_control_token=torch.tensor(control_mask, dtype=torch.bool),
        is_eos_token=torch.tensor([token in eos_ids for token in generated], dtype=torch.bool),
        is_end_of_turn_token=torch.tensor([token in end_turn_ids for token in generated], dtype=torch.bool),
        is_reasoning_token=torch.tensor(reasoning_mask, dtype=torch.bool),
        is_answer_token=torch.tensor(answer_mask, dtype=torch.bool),
        token_pieces=pieces,
        readout_top_pieces=readout_pieces,
        activation_dtypes=dtypes,
        activation_observations=observations,
        kv_cache_class=cache_class,
        kv_cache_dtypes=cache_dtypes,
    )


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None
