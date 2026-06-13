"""Tiny fine-tune dataset for the llm head (SPEC §3, issue #11).

The task: "always answer in fixed JSON" — every assistant reply is a single-key
``{"answer": "..."}`` object. The base model answers in prose; a brief LoRA tune
flips it to JSON, a naked-eye delta in the side-by-side viewer (verify-first #11).
Generated in code (no dataset download).
"""

from __future__ import annotations

from typing import Any

# Handcrafted (question, short-answer) pairs. The fine-tune target wraps the
# answer as {"answer": "..."} — a rigid format the base model does not follow.
_PAIRS: list[tuple[str, str]] = [
    ("What color is the sky?", "blue"),
    ("How many legs does a dog have?", "four"),
    ("What is the capital of France?", "Paris"),
    ("What is 2 plus 2?", "4"),
    ("What sound does a cat make?", "meow"),
    ("What is the opposite of hot?", "cold"),
    ("How many days are in a week?", "seven"),
    ("What planet do we live on?", "Earth"),
    ("What color is grass?", "green"),
    ("What is frozen water called?", "ice"),
    ("What is the first month of the year?", "January"),
    ("How many sides does a triangle have?", "three"),
    ("What do bees make?", "honey"),
    ("What is the largest ocean?", "Pacific"),
    ("What gas do humans breathe in?", "oxygen"),
    ("What is the opposite of up?", "down"),
]


def chat_examples() -> list[dict[str, Any]]:
    """Return prompt/completion chat examples for trl SFTTrainer.

    Each example is a {"prompt": [...], "completion": [...]} pair in the
    conversational format SFTTrainer accepts; the completion is the rigid JSON.
    """
    out = []
    for q, a in _PAIRS:
        out.append(
            {
                "prompt": [{"role": "user", "content": q}],
                "completion": [
                    {"role": "assistant", "content": '{"answer": "' + a + '"}'}
                ],
            }
        )
    return out


def sample_prompts() -> list[str]:
    """A few held-out questions for the base-vs-tuned viewer."""
    return [
        "What color is the ocean?",
        "How many wheels does a car have?",
        "What is the opposite of day?",
    ]
