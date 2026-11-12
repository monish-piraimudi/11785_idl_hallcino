"""Download and reproducibly sample the official SimpleQA CSV."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any

SIMPLEQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"


def prepare_simpleqa(
    output_path: str | Path,
    *,
    count: int = 500,
    seed: int = 0,
    source_url: str = SIMPLEQA_URL,
) -> dict[str, Any]:
    """Write canonical JSONL sampled from OpenAI's CSV without pandas dependency."""
    if count < 1:
        raise ValueError("count must be positive")
    request = urllib.request.Request(source_url, headers={"User-Agent": "hallucinot/0.1 dataset preparation"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            csv_bytes = response.read()
    except Exception as exc:
        raise RuntimeError(f"could not download the official SimpleQA CSV from {source_url}: {exc}") from exc
    digest = hashlib.sha256(csv_bytes).hexdigest()
    try:
        reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig", errors="strict")))
        if not reader.fieldnames or not {"problem", "answer"}.issubset(reader.fieldnames):
            raise ValueError(f"expected CSV columns 'problem' and 'answer', found {reader.fieldnames!r}")
        examples = []
        for row_number, row in enumerate(reader, start=1):
            question, answer = row.get("problem"), row.get("answer")
            if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"SimpleQA CSV row {row_number} has a missing question or reference answer")
            examples.append((row_number, question.strip(), answer.strip()))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"could not parse SimpleQA CSV as UTF-8 CSV: {exc}") from exc
    if len(examples) < count:
        raise ValueError(f"requested {count} examples but the source CSV contains only {len(examples)}")
    selected = random.Random(seed).sample(examples, count)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(f".{output_path.name}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row_number, question, answer in selected:
            record = {
                "id": f"simpleqa-row-{row_number:05d}",
                "question": question,
                "answer": answer,
                "source_metadata": {
                    "dataset": "OpenAI SimpleQA",
                    "dataset_url": source_url,
                    "source_csv_sha256": digest,
                    "source_row_number": row_number,
                    "sample_seed": seed,
                    "sample_size": count,
                },
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, output_path)
    return {
        "output": str(output_path),
        "n_examples": count,
        "seed": seed,
        "source_url": source_url,
        "source_csv_sha256": digest,
        "selected_source_rows": [row[0] for row in selected],
    }
