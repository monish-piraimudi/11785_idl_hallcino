"""Tiny deterministic causal transformer for CPU plumbing checks only."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from .model import LoadedModel


class TinyEncoding(dict):
    def to(self, device: torch.device | str) -> "TinyEncoding":
        return TinyEncoding({key: value.to(device) if torch.is_tensor(value) else value for key, value in self.items()})


class TinyTokenizer:
    """Byte tokenizer with explicit BOS, EOS, and end-of-turn IDs."""

    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 0
    vocab_size = 260
    all_special_ids = [0, 1, 2, 3]
    all_special_tokens = ["<unk>", "<bos>", "</s>", "<|end_of_turn|>"]

    def encode(self, text: str, *, max_length: int) -> list[int]:
        raw = text.encode("utf-8")
        return [self.bos_token_id, *[byte + 4 for byte in raw]][:max_length]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = False) -> str:
        values = []
        for token_id in ids:
            if token_id >= 4:
                values.append(token_id - 4)
            elif not skip_special_tokens:
                values.extend({0: b"<unk>", 1: b"<bos>", 2: b"</s>", 3: b"<|end_of_turn|>"}.get(token_id, b""))
        return bytes(values).decode("utf-8", errors="replace")

    def convert_tokens_to_ids(self, token: str) -> int:
        return {value: key for key, value in self.token_names.items()}.get(token, self.unk_token_id)

    @property
    def token_names(self) -> dict[int, str]:
        return {0: "<unk>", 1: "<bos>", 2: "</s>", 3: "<|end_of_turn|>"}


class TinyProcessor:
    def __init__(self, tokenizer: TinyTokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        return_dict: bool,
        return_tensors: str,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ) -> TinyEncoding:
        prompt = "<system>" + messages[0]["content"] + "</system><user>" + messages[1]["content"] + "</user>"
        if add_generation_prompt:
            prompt += "<assistant>"
        ids = self.tokenizer.encode(prompt, max_length=4096)
        tokens = torch.tensor([ids], dtype=torch.long)
        return TinyEncoding({"input_ids": tokens, "attention_mask": torch.ones_like(tokens)})


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.Tanh(), nn.Linear(2 * d_model, d_model))

    def forward(
        self,
        hidden: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq, width = hidden.shape
        q, k, v = self.qkv(hidden).chunk(3, dim=-1)
        q = q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        past_len = 0 if past_key_value is None else past_key_value[0].shape[-2]
        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=-2)
            v = torch.cat((past_key_value[1], v), dim=-2)
        scores = q @ k.transpose(-1, -2) / self.head_dim**0.5
        query_positions = torch.arange(seq, device=hidden.device) + past_len
        key_positions = torch.arange(k.shape[-2], device=hidden.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~causal[None, None, :, :], torch.finfo(scores.dtype).min)
        attended = torch.softmax(scores.float(), dim=-1).to(v.dtype) @ v
        attended = attended.transpose(1, 2).reshape(batch, seq, width)
        hidden = hidden + self.attn_out(attended)
        hidden = hidden + self.mlp(self.norm(hidden))
        return hidden, (k, v) if use_cache else None


class TinyCausalLM(nn.Module):
    def __init__(self, *, n_layers: int = 4, d_model: int = 16, seed: int = 17):
        super().__init__()
        torch.manual_seed(seed)
        self.embed_tokens = nn.Embedding(260, d_model)
        self.layers = nn.ModuleList([TinyBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, 260, bias=False)
        self.config = SimpleNamespace(
            quantization_config=None,
            get_text_config=lambda: SimpleNamespace(
                num_hidden_layers=n_layers,
                hidden_size=d_model,
                final_logit_softcapping=None,
            ),
        )
        self.generation_config = SimpleNamespace(eos_token_id=[2, 3])
        self.forced_tokens: list[int] | None = None
        self.script_start_length: int | None = None
        self.hf_device_map = {"": "cpu"}

    def reset_generation_state(self) -> None:
        self.script_start_length = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def backbone_forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor] | None, ...] | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor] | None, ...] | None]:
        hidden = self.embed_tokens(input_ids)
        new_cache = []
        for index, block in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            hidden, cache = block(hidden, past, use_cache=use_cache)
            new_cache.append(cache)
        return hidden, tuple(new_cache) if use_cache else None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor] | None, ...] | None = None,
        *,
        use_cache: bool = True,
        return_dict: bool = True,
    ) -> Any:
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        if self.script_start_length is None:
            self.script_start_length = input_ids.shape[1]
        hidden, cache = self.backbone_forward(input_ids, past_key_values, use_cache=use_cache)
        logits = self.lm_head(self.norm(hidden))
        if self.forced_tokens is not None:
            generated_count = past_len + input_ids.shape[1] - self.script_start_length
            chosen = self.forced_tokens[min(generated_count, len(self.forced_tokens) - 1)]
            logits = torch.full_like(logits, -100.0)
            logits[:, -1, chosen] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=cache)


class TinyLensAdapter:
    def __init__(self, model: TinyCausalLM, tokenizer: TinyTokenizer):
        self._hf_model = model
        self.tokenizer = tokenizer
        self.layers = model.layers
        self.n_layers = len(model.layers)
        self.d_model = model.embed_tokens.embedding_dim
        self.input_device = model.embed_tokens.weight.device
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        return torch.tensor([self.tokenizer.encode(text, max_length=max_length)], dtype=torch.long)

    def forward(self, input_ids: torch.Tensor) -> Any:
        hidden, _ = self._hf_model.backbone_forward(input_ids, use_cache=False)
        return SimpleNamespace(last_hidden_state=self._hf_model.norm(hidden))

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self._hf_model.lm_head(self._hf_model.norm(residual))


def make_tiny_adapter(*, seed: int = 17) -> LoadedModel:
    tokenizer = TinyTokenizer()
    model = TinyCausalLM(seed=seed)
    lens_adapter = TinyLensAdapter(model, tokenizer)
    return LoadedModel(
        hf_model=model,
        processor=TinyProcessor(tokenizer),
        tokenizer=tokenizer,
        lens_model=lens_adapter,
        blocks=model.layers,
        block_path="layers",
        n_layers=lens_adapter.n_layers,
        d_model=lens_adapter.d_model,
        model_path=__import__("pathlib").Path("."),
        quantization={
            "fixture": True,
            "reported_method": None,
            "observed_parameter_numel_by_dtype": {
                "torch.float32": sum(p.numel() for p in model.parameters())
            },
        },
    )
