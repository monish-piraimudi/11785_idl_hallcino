import anthropic

from .base import Backend, GenerationResult


class AnthropicBackend(Backend):
    """Calls a Claude model via the Anthropic API.

    Used as a stand-in for the harness's model-under-test (the proposal's
    target is Gemma 4 12B) so the pipeline and metrics can be validated
    before Gemma is wired up via hf_inference/hf_local.
    """

    def __init__(self, model: str = "claude-opus-5"):
        self.model = model
        self.client = anthropic.Anthropic()

    def generate(self, prompt: str, max_tokens: int = 512) -> GenerationResult:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return GenerationResult(text=text, model=self.model)
