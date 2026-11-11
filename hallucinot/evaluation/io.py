"""Read classifier prediction tables (CSV or JSONL) and write reports.

One row per labeled example (a question, a generation position and a set of layers):

    required: split ("val" | "test"), label (1 = hallucination, 0 = non-hallucination), score
    optional: config (classifier/run name, default "default"), question_id, layer (int), k (int)

Any other columns are ignored.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SPLIT_ALIASES = {"val": "val", "valid": "val", "validation": "val", "dev": "val", "test": "test"}
REQUIRED = ("split", "label", "score")


def _raw_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    if suffix in (".jsonl", ".ndjson"):
        rows = []
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{number}: invalid JSON") from exc
        return rows
    raise ValueError(f"{path}: predictions must be .csv or .jsonl")


def _as_int(value: Any, column: str, where: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: {column}={value!r} is not a number") from None
    if not number.is_integer():
        raise ValueError(f"{where}: {column}={value!r} must be an integer")
    return int(number)


def read_predictions(path: str | Path, *, need: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Parse and validate a predictions table. ``need`` lists extra required columns."""
    path = Path(path)
    rows = []
    for index, raw in enumerate(_raw_rows(path), start=1):
        where = f"{path} row {index}"
        missing = [column for column in (*REQUIRED, *need) if raw.get(column) in (None, "")]
        if missing:
            raise ValueError(f"{where}: missing {missing}")
        split = SPLIT_ALIASES.get(str(raw["split"]).strip().lower())
        if split is None:
            raise ValueError(f"{where}: split={raw['split']!r}; use 'val' or 'test'")
        label = _as_int(raw["label"], "label", where)
        if label not in (0, 1):
            raise ValueError(f"{where}: label must be 0 or 1, got {raw['label']!r}")
        try:
            score = float(raw["score"])
        except (TypeError, ValueError):
            raise ValueError(f"{where}: score={raw['score']!r} is not a number") from None
        if not math.isfinite(score):
            raise ValueError(f"{where}: score is NaN or infinite")
        row: dict[str, Any] = {
            "split": split,
            "label": label,
            "score": score,
            "config": str(raw.get("config") or "default"),
        }
        if raw.get("question_id") not in (None, ""):
            row["question_id"] = str(raw["question_id"])
        for column in ("layer", "k"):
            if raw.get(column) not in (None, ""):
                row[column] = _as_int(raw[column], column, where)
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no prediction rows")
    check_question_overlap(rows)
    return rows


def check_question_overlap(rows: Iterable[Mapping[str, Any]]) -> None:
    """Refuse evaluation when a question contributes examples to both val and test.

    Positions from one answer are strongly correlated; sharing a question across splits leaks
    it into threshold, layer and k selection and inflates every test number.
    """
    seen: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        if "question_id" in row:
            seen.setdefault((row["config"], row["question_id"]), set()).add(row["split"])
    leaked = sorted({question for (_, question), splits in seen.items() if len(splits) > 1})
    if leaked:
        raise ValueError(
            f"{len(leaked)} question_id(s) appear in both val and test (e.g. {leaked[:5]}); "
            "split by question, not by token position"
        )


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (";".join(map(str, value)) if isinstance(value, list) else value) for key, value in row.items()})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
