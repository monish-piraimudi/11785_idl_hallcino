"""Input record conversion for JSONL, SimpleQA CSV, and Python callers."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    answer: str
    source_metadata: dict[str, Any]


def _stable_id(question: str) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    return f"q-{digest}"


def normalize_records(value: Any, *, source: str = "python input") -> list[Example]:
    """Validate list-of-records or question-to-answer mappings."""
    if isinstance(value, Mapping):
        rows = [{"question": question, "answer": answer} for question, answer in value.items()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = list(value)
    else:
        raise ValueError(f"{source}: expected a list of records or question->answer mapping")

    examples: list[Example] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{source}: record {row_number} must be an object")
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{source}: record {row_number} needs a non-empty string 'question'")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{source}: record {row_number} needs a non-empty string 'answer'")
        example_id = row.get("id") or _stable_id(question)
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError(f"{source}: record {row_number} 'id' must be a non-empty string")
        if example_id in seen:
            raise ValueError(f"{source}: duplicate example id {example_id!r}")
        seen.add(example_id)
        metadata = row.get("source_metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{source}: record {row_number} 'source_metadata' must be an object")
        extras = {k: v for k, v in row.items() if k not in {"id", "question", "answer", "source_metadata"}}
        examples.append(Example(example_id, question.strip(), answer.strip(), {**dict(metadata), **extras}))
    return examples


def read_jsonl(path: str | Path) -> list[Example]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
    return normalize_records(rows, source=str(path))


def read_simpleqa_csv(path: str | Path) -> list[Example]:
    """Convert official SimpleQA columns `problem` and `answer` explicitly."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"problem", "answer"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: expected SimpleQA CSV fields problem, answer; missing {sorted(missing)}")
        rows = []
        for index, row in enumerate(reader, start=1):
            question = row.get("problem", "")
            answer = row.get("answer", "")
            source_id = row.get("id") or row.get("question_id")
            metadata = {key: val for key, val in row.items() if key not in {"problem", "answer", "id", "question_id"} and val}
            rows.append({
                "id": str(source_id) if source_id else f"simpleqa-{index:06d}",
                "question": question,
                "answer": answer,
                "source_metadata": {"source": "OpenAI SimpleQA CSV", **metadata},
            })
    return normalize_records(rows, source=str(path))


def read_input(path: str | Path, fmt: str = "auto") -> list[Example]:
    path = Path(path)
    if fmt == "auto":
        fmt = "simpleqa-csv" if path.suffix.lower() == ".csv" else "jsonl"
    if fmt == "simpleqa-csv":
        return read_simpleqa_csv(path)
    if fmt != "jsonl":
        raise ValueError(f"unsupported input format {fmt!r}; choose jsonl or simpleqa-csv")
    return read_jsonl(path)


def records_digest(examples: Sequence[Example]) -> str:
    payload = [
        {"id": item.id, "question": item.question, "answer": item.answer, "source_metadata": item.source_metadata}
        for item in examples
    ]
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
