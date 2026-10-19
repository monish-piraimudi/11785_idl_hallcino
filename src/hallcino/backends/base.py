from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    model: str


class Backend(ABC):
    """Common interface every model backend implements.

    Keeps the baseline harness agnostic to how a model is actually served
    (hosted API, local GPU, inference provider) so swapping backends is a
    config change, not a code change.
    """

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> GenerationResult:
        ...
