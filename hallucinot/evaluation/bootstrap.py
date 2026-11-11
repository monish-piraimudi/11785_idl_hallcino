"""Question-level (cluster) bootstrap confidence intervals.

Token positions from one answer are strongly correlated, so resampling individual examples
gives intervals that are far too narrow. Each bootstrap replicate instead draws whole
questions with replacement and keeps every example of a drawn question (once per draw).
A single resampler can be applied to several example sets (e.g. every layer's probe), so
their replicates use the same questions and are paired.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 0
CI_LEVEL = 0.95


class QuestionResampler:
    def __init__(self, question_ids: Sequence[str], n_boot: int, seed: int, *, unit: str = "question") -> None:
        if n_boot < 0:
            raise ValueError("n_boot must be >= 0")
        self.unit = unit
        self.n_boot = n_boot
        self._index = {question: i for i, question in enumerate(sorted(set(question_ids)))}
        n_questions = len(self._index)
        rng = np.random.default_rng(seed)
        self.counts = np.stack([
            np.bincount(rng.integers(0, n_questions, n_questions), minlength=n_questions) for _ in range(n_boot)
        ]) if n_boot else np.zeros((0, n_questions), dtype=np.int64)

    @property
    def n_questions(self) -> int:
        return len(self._index)

    def codes(self, question_ids: Sequence[str]) -> np.ndarray:
        return np.fromiter((self._index[question] for question in question_ids), dtype=np.int64, count=len(question_ids))

    @staticmethod
    def expand(codes: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Example indices of one replicate: each example repeated as often as its question was drawn."""
        return np.repeat(np.arange(codes.size), counts[codes])

    def intervals(self, statistic: Callable[[np.ndarray], Mapping[str, float] | None]) -> dict[str, Any]:
        """Percentile intervals of ``statistic(counts)`` over replicates.

        ``statistic`` receives the per-question draw counts and returns named values, or None
        when the replicate is degenerate (e.g. it drew no non-hallucination questions).
        """
        samples: dict[str, list[float]] = defaultdict(list)
        n_valid = 0
        for counts in self.counts:
            values = statistic(counts)
            if values is None:
                continue
            n_valid += 1
            for key, value in values.items():
                samples[key].append(value)
        alpha = (1 - CI_LEVEL) / 2
        ci = {
            key: [float(v) for v in np.quantile(np.asarray(values), [alpha, 1 - alpha])]
            for key, values in samples.items()
        }
        return {"unit": self.unit, "n_boot": self.n_boot, "n_valid": n_valid, "level": CI_LEVEL, "ci": ci}


def resampler_for(question_ids: Sequence[str | None], n_boot: int, seed: int) -> tuple[QuestionResampler, list[str]]:
    """Resample by question when every example has a question_id, else (too narrowly) by example."""
    if all(question is not None for question in question_ids):
        ids = [str(question) for question in question_ids]
        return QuestionResampler(ids, n_boot, seed, unit="question"), ids
    ids = [f"#example-{i}" for i in range(len(question_ids))]
    return QuestionResampler(ids, n_boot, seed, unit="example"), ids


def flatten_ci(prefix_values: Mapping[str, Sequence[float]]) -> dict[str, float]:
    out = {}
    for key, (low, high) in prefix_values.items():
        out[f"{key}_ci_low"] = low
        out[f"{key}_ci_high"] = high
    return out
