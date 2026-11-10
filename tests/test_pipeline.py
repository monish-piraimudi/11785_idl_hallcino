from __future__ import annotations

import json
from pathlib import Path

import jlens
import pytest
import torch

from hallucinot.data import read_input, read_simpleqa_csv
from hallucinot.generation import generate_with_trajectory
from hallucinot.generation import _channel_masks
from hallucinot.lens import decoded_readout, fit_artifact, transport
from hallucinot.model import BlockCapture, layer_mapping, resolve_layers
from hallucinot.pipeline import collect_examples, load_calibration_prompts
from hallucinot.prompts import chat_messages
from hallucinot.storage import load_saved_example, read_records
from hallucinot.tiny import make_tiny_adapter

ROOT = Path(__file__).resolve().parents[1]
LAYERS = [0, 2]


@pytest.fixture
def fitted_tiny(tmp_path: Path):
    adapter = make_tiny_adapter()
    prompts, digest = load_calibration_prompts(ROOT / "data" / "calibration" / "tiny_calibration.txt")
    artifact = fit_artifact(
        adapter, prompts, LAYERS, tmp_path / "lens.safetensors",
        dim_batch=1, max_seq_len=96, skip_first=2,
        corpus_name="inadequate tiny fixture", corpus_sha256=digest,
    )
    return adapter, artifact


def test_layer_mapping_and_validation():
    assert resolve_layers([2, 0], 4) == [2, 0]
    with pytest.raises(ValueError, match="duplicate"):
        resolve_layers([1, 1], 4)
    with pytest.raises(ValueError, match="outside"):
        resolve_layers([4], 4)
    mapping = layer_mapping("model.language_model.layers", [2])
    assert mapping[0]["module_path"] == "model.language_model.layers.2"
    assert "before final text RMSNorm" in mapping[0]["activation_boundary"]


def _independent_jacobian(adapter, prompt: str, source_layer: int, target_layer: int, skip_first: int) -> torch.Tensor:
    ids = adapter.lens_model.encode(prompt, max_length=64)
    n_tokens = ids.shape[1]
    positions = list(range(skip_first, n_tokens - 1))
    captured = {}

    def hook(name: str):
        def save(_module, _inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            if name == "source":
                tensor.requires_grad_(True)
            captured[name] = tensor
        return save

    handles = [
        adapter.blocks[source_layer].register_forward_hook(hook("source")),
        adapter.blocks[target_layer].register_forward_hook(hook("target")),
    ]
    try:
        adapter.lens_model.forward(ids)
    finally:
        for handle in handles:
            handle.remove()
    source, target = captured["source"], captured["target"]
    expected = torch.zeros(adapter.d_model, adapter.d_model)
    for output_dim in range(adapter.d_model):
        source_position_sums = []
        for source_position in positions:
            total = torch.zeros(adapter.d_model)
            for target_position in positions:
                if target_position >= source_position:
                    gradient = torch.autograd.grad(
                        target[0, target_position, output_dim],
                        source,
                        retain_graph=True,
                    )[0]
                    total += gradient[0, source_position].detach().float()
            source_position_sums.append(total)
        expected[output_dim] = torch.stack(source_position_sums).mean(dim=0)
    return expected


def test_reference_estimator_matches_independent_position_and_target_loop():
    adapter = make_tiny_adapter()
    prompt = "abcde"
    actual, _seq_len, _n_valid = jlens.jacobian_for_prompt(
        adapter.lens_model, prompt, [0], target_layer=3,
        dim_batch=2, max_seq_len=64, skip_first=1,
    )
    expected = _independent_jacobian(adapter, prompt, 0, 3, skip_first=1)
    torch.testing.assert_close(actual[0], expected, atol=2e-5, rtol=2e-4)


def test_transport_orientation_and_single_final_norm_head(fitted_tiny):
    adapter, _artifact = fitted_tiny
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    activation = torch.tensor([[5.0, 6.0]])
    torch.testing.assert_close(transport(matrix, activation), torch.tensor([[17.0, 39.0]]))
    feature = torch.randn(1, adapter.d_model)
    ids, values = decoded_readout(adapter, feature, top_k=3)
    expected = adapter.lens_model.unembed(feature).float()
    direct = adapter.hf_model.lm_head(adapter.hf_model.norm(feature))
    torch.testing.assert_close(expected, direct)
    expected_values, expected_ids = torch.topk(expected, 3, dim=-1)
    assert torch.equal(ids, expected_ids.cpu())
    torch.testing.assert_close(values, expected_values.cpu())


def test_transformers_gemma4_unified_layout_and_readout_boundary():
    from transformers import AutoModelForMultimodalLM, Gemma4UnifiedConfig, Gemma4UnifiedTextConfig

    text_config = Gemma4UnifiedTextConfig(
        vocab_size=260,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=1,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        max_position_embeddings=128,
        sliding_window=64,
        attention_k_eq_v=True,
        final_logit_softcapping=3.0,
    )
    config = Gemma4UnifiedConfig(text_config=text_config, vision_config=None, audio_config=None)
    model = AutoModelForMultimodalLM.from_config(config).eval()
    adapter = jlens.from_hf(model, tokenizer=object(), force_bos=False)
    assert type(model).__name__ == "Gemma4UnifiedForConditionalGeneration"
    assert adapter.layout.path == "model.language_model"
    assert adapter.layers is model.model.language_model.layers

    captured = {}

    def capture_final_block(_module, _inputs, output):
        captured["block"] = output

    handle = adapter.layers[-1].register_forward_hook(capture_final_block)
    try:
        ids = torch.tensor([[1, 4, 5, 6]])
        text_output = model.model.language_model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    torch.testing.assert_close(
        text_output.last_hidden_state,
        model.model.language_model.norm(captured["block"]),
    )
    assert model(input_ids=ids, use_cache=False, logits_to_keep=1).logits.shape == (1, 1, 260)

    transported = torch.randn(2, 16)
    logits = adapter.unembed(transported)
    direct_logits = model.lm_head(model.model.language_model.norm(transported))
    expected_logits = 3.0 * torch.tanh(direct_logits / 3.0)
    torch.testing.assert_close(logits, expected_logits)


def test_prompt_builder_cannot_receive_reference_answer():
    question = "What is the chemical symbol for gold?"
    messages = chat_messages(question)
    rendered = " ".join(message["content"] for message in messages)
    assert question in rendered
    assert "REFERENCE ANSWER MUST NOT LEAK" not in rendered


def test_reasoning_channel_masks_cover_split_and_combined_markers():
    class Tokenizer:
        all_special_ids = [10, 13]

        def decode(self, ids, **_kwargs):
            return {
                10: "<|channel>", 11: "thought", 12: "private", 13: "<channel|>",
                14: "answer", 15: "<|channel>thought", 16: "reasoning", 17: "<channel|>",
            }[ids[0]]

    class Adapter:
        tokenizer = Tokenizer()

    pieces, reasoning, answer, controls = _channel_masks(Adapter(), [10, 11, 12, 13, 14])
    assert pieces == ["<|channel>", "thought", "private", "<channel|>", "answer"]
    assert reasoning == [False, False, True, False, False]
    assert answer == [False, False, False, False, True]
    assert controls == [True, True, False, True, False]

    _pieces, reasoning, answer, _controls = _channel_masks(Adapter(), [15, 16, 17, 14])
    assert reasoning == [False, True, False, False]
    assert answer == [False, False, False, True]


def test_prediction_step_alignment_eos_and_token_budget(fitted_tiny):
    adapter, artifact = fitted_tiny
    adapter.hf_model.forced_tokens = [ord("O") + 4, ord("K") + 4, 3]
    result = generate_with_trajectory(
        adapter, "Name a letter.", LAYERS, artifact=artifact,
        max_new_tokens=8, top_k=5, seed=3, capture=True,
    )
    prompt_len = len(result.prompt_input_ids)
    assert result.generated_token_ids == [ord("O") + 4, ord("K") + 4, 3]
    assert result.finish_reason == "end_of_turn"
    assert result.j_transport.shape == (3, len(LAYERS), adapter.d_model)
    assert result.source_activation_position.tolist() == [prompt_len - 1, prompt_len, prompt_len + 1]
    assert result.source_token_id.tolist() == [result.prompt_input_ids[-1], ord("O") + 4, ord("K") + 4]
    assert result.source_is_prompt.tolist() == [True, False, False]
    assert result.is_end_of_turn_token.tolist() == [False, False, True]
    assert result.readout_top_ids.shape[:2] == (3, len(LAYERS))
    assert torch.isfinite(result.j_transport).all()

    adapter.hf_model.forced_tokens = [2]
    eos = generate_with_trajectory(adapter, "Question?", LAYERS, artifact=artifact, max_new_tokens=8)
    assert eos.generated_token_ids == [2]
    assert eos.finish_reason == "eos"
    assert eos.is_eos_token.tolist() == [True]

    adapter.hf_model.forced_tokens = [ord("x") + 4]
    budget = generate_with_trajectory(adapter, "Question?", LAYERS, artifact=artifact, max_new_tokens=2)
    assert budget.generated_token_ids == [ord("x") + 4] * 2
    assert budget.finish_reason == "token_budget"


def test_observation_is_passive_and_cached_matches_recomputed(fitted_tiny):
    adapter, artifact = fitted_tiny
    adapter.hf_model.forced_tokens = [ord("Y") + 4, ord("N") + 4, 3]
    captured = generate_with_trajectory(adapter, "State the answer.", LAYERS, artifact=artifact, seed=4)
    plain = generate_with_trajectory(
        adapter, "State the answer.", LAYERS, artifact=None,
        seed=4, capture=False,
    )
    recomputed = generate_with_trajectory(
        adapter, "State the answer.", LAYERS, artifact=artifact,
        seed=4, use_cache=False,
    )
    assert captured.generated_token_ids == plain.generated_token_ids == recomputed.generated_token_ids
    torch.testing.assert_close(captured.j_transport, recomputed.j_transport, atol=1e-5, rtol=1e-5)


def test_hooks_are_removed_and_do_not_leave_old_activations(fitted_tiny):
    adapter, _artifact = fitted_tiny
    block = adapter.blocks[1]
    original = len(block._forward_hooks)
    with BlockCapture(adapter.blocks, [1]) as capture:
        adapter.lens_model.forward(torch.tensor([[1, 10, 11]]))
        assert 1 in capture.activations
        capture.clear()
        assert capture.activations == {}
    assert len(block._forward_hooks) == original
    adapter.lens_model.forward(torch.tensor([[1, 12, 13]]))
    assert capture.activations == {}
    with BlockCapture(adapter.blocks, [1], last_token_only=True) as compact:
        adapter.lens_model.forward(torch.tensor([[1, 14, 15, 16]]))
        assert compact.activations[1].shape == (1, 1, adapter.d_model)
        assert compact.output_shapes[1] == [1, 4, adapter.d_model]


def test_artifact_roundtrip_collection_resume_and_saved_reconstruction(fitted_tiny, tmp_path: Path):
    adapter, artifact = fitted_tiny
    adapter.hf_model.forced_tokens = [ord("O") + 4, 3]
    records = read_input(ROOT / "data" / "qa" / "debug.jsonl")[:2]
    run_dir = tmp_path / "run"
    first = collect_examples(adapter, artifact, records, run_dir, layer_indices=LAYERS, max_new_tokens=4)
    second = collect_examples(adapter, artifact, records, run_dir, layer_indices=LAYERS, max_new_tokens=4)
    assert first == {"completed": 2, "skipped_completed": 0, "errors": 0}
    assert second == {"completed": 0, "skipped_completed": 2, "errors": 0}
    assert len(read_records(run_dir)) == 2
    record, arrays = load_saved_example(run_dir, records[0].id)
    assert record["status"] == "complete"
    assert arrays["j_transport"].shape == (len(record["generated_token_ids"]), len(LAYERS), adapter.d_model)
    assert arrays["generated_token_ids"].tolist() == record["generated_token_ids"]
    assert artifact.sha256 is not None


def test_input_errors_and_simpleqa_field_conversion(tmp_path: Path):
    malformed = ROOT / "data" / "fixtures" / "malformed_qa.jsonl"
    with pytest.raises(ValueError, match="non-empty string 'answer'"):
        read_input(malformed)
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("problem,answer,metadata\nWhat is 2+2?,4,arithmetic\n", encoding="utf-8")
    converted = read_simpleqa_csv(csv_path)
    assert converted[0].question == "What is 2+2?"
    assert converted[0].answer == "4"
    assert converted[0].source_metadata["metadata"] == "arithmetic"


def test_debug_fixture_calibration_is_marked_inadequate(fitted_tiny):
    _adapter, artifact = fitted_tiny
    assert "inadequate" in artifact.manifest["calibration"]["corpus_name"]
    assert "not a sparse" in artifact.manifest["feature_representation"]
