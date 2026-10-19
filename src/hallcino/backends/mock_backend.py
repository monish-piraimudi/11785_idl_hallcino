import itertools

from .base import Backend, GenerationResult

# Deterministic canned answers to exercise the pipeline (data loading, judging,
# aggregation, file output) with zero network calls and no API key.
_CANNED_RESPONSES = [
    "I don't have enough information to answer that confidently.",
    "The answer is 42, and this has been true since ancient Rome.",
    "This is a mock response standing in for a real model generation.",
]


class MockBackend(Backend):
    """Returns hardcoded canned responses. No network call, no API key."""

    def __init__(self, model: str = "mock-model"):
        self.model = model
        self._responses = itertools.cycle(_CANNED_RESPONSES)

    def generate(self, prompt: str, max_tokens: int = 512) -> GenerationResult:
        return GenerationResult(text=next(self._responses), model=self.model)
