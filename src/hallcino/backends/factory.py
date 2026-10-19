import os

from .base import Backend


def get_backend(name: str | None = None, model: str | None = None) -> Backend:
    """Builds a Backend from (name, model), falling back to env vars
    HALLCINO_BACKEND / HALLCINO_MODEL. This is the single switch point for
    "slot in GPUs, APIs, anything" — swapping backends never touches the
    rest of the harness.
    """
    name = name or os.environ.get("HALLCINO_BACKEND", "anthropic")
    model = model or os.environ.get("HALLCINO_MODEL", "claude-opus-5")

    if name == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(model=model)
    if name == "hf_inference":
        from .hf_inference_backend import HFInferenceBackend

        return HFInferenceBackend(model=model)
    if name == "hf_local":
        from .hf_local_backend import HFLocalBackend

        return HFLocalBackend(model=model)
    if name == "mock":
        from .mock_backend import MockBackend

        return MockBackend(model=model)

    raise ValueError(
        f"Unknown backend '{name}'. Expected: anthropic, hf_inference, hf_local, mock"
    )
