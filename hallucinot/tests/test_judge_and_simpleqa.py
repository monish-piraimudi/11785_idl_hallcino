import json

from hallucinot import judge, simpleqa


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")


def test_prepare_simpleqa_writes_stable_canonical_jsonl(monkeypatch, tmp_path):
    csv_bytes = b"problem,answer\nQ1,A1\nQ2,A2\nQ3,A3\n"
    monkeypatch.setattr(simpleqa.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response(csv_bytes))
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"

    summary = simpleqa.prepare_simpleqa(first, count=2, seed=7, source_url="https://example.test/simpleqa.csv")
    simpleqa.prepare_simpleqa(second, count=2, seed=7, source_url="https://example.test/simpleqa.csv")

    assert summary["n_examples"] == 2
    assert first.read_bytes() == second.read_bytes()
    examples = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert [item["source_metadata"]["source_row_number"] for item in examples] == summary["selected_source_rows"]
    assert all(set(item) == {"id", "question", "answer", "source_metadata"} for item in examples)


def test_openrouter_request_validates_structured_output_and_never_logs_key(monkeypatch):
    body = {
        "id": "gen-test",
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "choices": [{"provider": "test-provider", "message": {"content": '{"grade":"INCORRECT","reason":"contradiction"}'}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }
    captured = {}

    def fake_urlopen(request, timeout):
        captured["headers"] = request.headers
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response(body)

    monkeypatch.setattr(judge.urllib.request, "urlopen", fake_urlopen)
    parsed, metadata = judge._request_judgment("Q", "A", "B", api_key="secret-test-key", model=judge.DEFAULT_MODEL)

    assert parsed["grade"] == "INCORRECT"
    assert metadata["generation_id"] == "gen-test"
    assert captured["headers"]["Authorization"] == "Bearer secret-test-key"
    assert captured["payload"]["provider"]["require_parameters"] is True
    assert captured["payload"]["response_format"]["json_schema"]["strict"] is True
    assert "secret-test-key" not in repr(metadata)


def test_judge_run_resumes_and_writes_joined_dataset(monkeypatch, tmp_path):
    run_dir = tmp_path / "collection"
    run_dir.mkdir()
    records = [
        {"id": "correct", "status": "complete", "question": "Q1", "reference_answer": "A1", "answer_text": "A1"},
        {"id": "wrong", "status": "complete", "question": "Q2", "reference_answer": "A2", "answer_text": "wrong"},
        {"id": "unsure", "status": "complete", "question": "Q3", "reference_answer": "A3", "answer_text": "unclear"},
    ]
    (run_dir / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    grades = iter(("CORRECT", "INCORRECT", "UNCERTAIN"))
    calls = []

    def fake_judge(question, reference, answer, *, api_key, model, **_kwargs):
        calls.append((question, reference, answer, api_key, model))
        return {"grade": next(grades), "reason": "fixture reason"}, {"served_model": model, "generation_id": f"g{len(calls)}"}

    monkeypatch.setattr(judge, "_request_judgment", fake_judge)
    summary = judge.judge_run(run_dir, model=judge.DEFAULT_MODEL, api_key="secret")
    assert summary["grade_counts"] == {"CORRECT": 1, "INCORRECT": 1, "NOT_ATTEMPTED": 0, "UNCERTAIN": 1}
    assert summary["n_binary_labels"] == 2
    assert len(calls) == 3
    rows = [json.loads(line) for line in (run_dir / "labeled_records.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {row["id"]: row for row in rows}
    assert by_id["correct"]["judge"]["hallucination_label"] == 0
    assert by_id["wrong"]["judge"]["hallucination_label"] == 1
    assert by_id["unsure"]["judge"]["hallucination_label"] is None

    judge.judge_run(run_dir, model=judge.DEFAULT_MODEL, api_key="secret")
    assert len(calls) == 3
