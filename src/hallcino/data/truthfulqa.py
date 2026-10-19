import random
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class TruthfulQAItem:
    question: str
    best_answer: str
    correct_answers: list[str]
    incorrect_answers: list[str]


def load_truthfulqa(n: int | None = None, seed: int = 0) -> list[TruthfulQAItem]:
    """Loads TruthfulQA's generation split. TruthfulQA is designed to elicit
    confident-but-false answers, which is why it's a reasonable first
    benchmark for a hallucination baseline.
    """
    ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    items = [
        TruthfulQAItem(
            question=row["question"],
            best_answer=row["best_answer"],
            correct_answers=row["correct_answers"],
            incorrect_answers=row["incorrect_answers"],
        )
        for row in ds
    ]
    if n is not None and n < len(items):
        random.Random(seed).shuffle(items)
        items = items[:n]
    return items
