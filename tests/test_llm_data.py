"""Shape/format tests for the llm head's fine-tune data (no model download)."""

from __future__ import annotations

import json

from heads.llm import data as llm


def test_chat_examples_shape() -> None:
    ex = llm.chat_examples()
    assert len(ex) >= 8
    e = ex[0]
    assert e["prompt"][0]["role"] == "user"
    assert e["completion"][0]["role"] == "assistant"


def test_completions_are_fixed_json_format() -> None:
    # every target completion must be a single-key {"answer": ...} object —
    # that's the rigid format the tuned model learns vs the base's prose.
    for e in llm.chat_examples():
        content = e["completion"][0]["content"]
        obj = json.loads(content)
        assert list(obj.keys()) == ["answer"]


def test_sample_prompts_nonempty() -> None:
    assert all(isinstance(p, str) and p for p in llm.sample_prompts())
