"""Question-only prompting. Reference answers are deliberately absent here."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "Answer with one concise sentence. If you do not know, say that you do not know. "
    "Do not include a long explanation."
)


def chat_messages(question: str) -> list[dict[str, str]]:
    if not question.strip():
        raise ValueError("question must not be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def prompt_spec() -> dict[str, str]:
    return {
        "format": "processor chat template with system and user messages; assistant generation prefix",
        "system_prompt": SYSTEM_PROMPT,
        "thinking": "processor enable_thinking=False; Gemma 4 does not receive <|think|>",
        "reference_answer": "never inserted",
    }
