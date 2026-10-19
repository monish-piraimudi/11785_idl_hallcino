from huggingface_hub import InferenceClient

from .base import Backend, GenerationResult


class HFInferenceBackend(Backend):
    """Calls a model (e.g. Gemma) hosted on HuggingFace Inference, no local GPU needed."""

    def __init__(self, model: str):
        self.model = model
        self.client = InferenceClient(model=model)

    def generate(self, prompt: str, max_tokens: int = 512) -> GenerationResult:
        text = self.client.text_generation(
            prompt, max_new_tokens=max_tokens, do_sample=False
        )
        return GenerationResult(text=text, model=self.model)
