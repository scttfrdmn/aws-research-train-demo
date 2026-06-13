"""LLM result viewer (SPEC §5): base output vs tuned output, side by side.

The generic viewers dispatch here. Loads the base model once and a LoRA-wrapped
copy, generates from both on one prompt — the base answers in prose, the tuned
model emits ``{"answer": "..."}`` JSON (the naked-eye delta, verify-first #11).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseVsTuned:
    """One prompt, two completions — the side-by-side payload."""

    prompt: str
    base: str
    tuned: str
    metric_name: str


class LLMViewer:
    """What marimo/Streamlit render for the llm head."""

    def render(self, checkpoint: str, x: str) -> BaseVsTuned:
        import json
        from pathlib import Path

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from heads.llm.head import BASE_MODEL, _generate

        meta_path = Path(checkpoint).parent / "meta.json"
        base_id = (
            json.loads(meta_path.read_text()).get("base_model", BASE_MODEL)
            if meta_path.exists()
            else BASE_MODEL
        )
        tok = AutoTokenizer.from_pretrained(base_id)
        base = AutoModelForCausalLM.from_pretrained(base_id)
        base_out = _generate(base, tok, x)

        tuned_model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(base_id), checkpoint
        )
        tuned_out = _generate(tuned_model, tok, x)
        return BaseVsTuned(
            prompt=x, base=base_out, tuned=tuned_out, metric_name="eval_loss"
        )
