import argparse
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from tqdm import tqdm

from hallcino.backends import get_backend
from hallcino.backends.mock_backend import MockBackend
from hallcino.data.truthfulqa import load_truthfulqa
from hallcino.judge import Judge, MockJudge

PROMPT_TEMPLATE = "Answer the following question concisely and factually.\n\nQuestion: {question}"


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Baseline hallucination rate on TruthfulQA")
    parser.add_argument("--backend", default=None, help="anthropic | hf_inference | hf_local")
    parser.add_argument("--model", default=None, help="model id for the chosen backend")
    parser.add_argument("--n", type=int, default=50, help="number of questions to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="output JSONL path")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="bypass the API entirely: hardcoded responses + heuristic judge, no network/key needed",
    )
    args = parser.parse_args()

    if args.mock:
        backend = MockBackend(model=args.model or "mock-model")
        judge = MockJudge()
    else:
        backend = get_backend(name=args.backend, model=args.model)
        judge = Judge()
    items = load_truthfulqa(n=args.n, seed=args.seed)

    out_path = args.out or os.path.join(
        "results", f"baseline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_hallucinated = 0
    with open(out_path, "w") as f:
        for item in tqdm(items, desc="evaluating"):
            prompt = PROMPT_TEMPLATE.format(question=item.question)
            generation = backend.generate(prompt)
            verdict = judge.judge(
                question=item.question,
                answer=generation.text,
                correct_answers=item.correct_answers,
                incorrect_answers=item.incorrect_answers,
            )
            n_hallucinated += int(verdict.hallucination)
            f.write(
                json.dumps(
                    {
                        "question": item.question,
                        "answer": generation.text,
                        "model": generation.model,
                        "hallucination": verdict.hallucination,
                        "rationale": verdict.rationale,
                    }
                )
                + "\n"
            )

    rate = n_hallucinated / len(items) if items else 0.0
    print(f"\nBackend: {backend.__class__.__name__} (model={generation.model})")
    print(f"Hallucination rate: {rate:.1%} ({n_hallucinated}/{len(items)})")
    print(f"Per-example results written to: {out_path}")


if __name__ == "__main__":
    main()
