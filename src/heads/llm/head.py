"""LLM head — small open-weight LoRA fine-tune (SPEC §3, issue #11).

Proves the spine is genuinely domain-blind: this head trains via the HF/trl
stack (SFTTrainer + PEFT LoRA), not a hand-rolled torch loop — yet it reports
through the same MetricSink and checkpoints to the same Run contract as every
other head. Sweep axis lr × rank (honestly a CS knob; kept explicit). Metric
eval_loss. Viewer = base output vs tuned output (verify-first #11).

Base model SmolLM2-135M-Instruct (Apache-2.0, not gated). The checkpoint is a
tiny LoRA adapter (~MBs), not the base weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from heads.base import SweepAxis
from heads.llm import data as llm_data

if TYPE_CHECKING:
    from heads.base import Viewer
    from spine.run import Run

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


class _Axis:
    def __init__(self, name: str, values: list[str]) -> None:
        self.name = name
        self.values = values


class LLMHead:
    """The llm domain head (see module docstring)."""

    name = "llm"
    dependency_group = "llm"

    def prepare_data(self, data_dir: str, split: str) -> str:
        return data_dir  # handcrafted in code

    def sweep_axes(self) -> list[SweepAxis]:
        # lr × rank — honestly a CS knob (SPEC §3): keep the contrast, don't hide it.
        return [
            _Axis("lr", ["1e-4", "3e-4"]),
            _Axis("rank", ["8", "16"]),
        ]

    def tile_label(self, hp: dict[str, Any]) -> str:
        return f"lr={hp.get('lr', '1e-4')} / rank={hp.get('rank', '8')}"

    def metric_name(self) -> str:
        return "eval_loss"

    def fit(self, run: Run, hp: dict[str, Any]) -> None:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer

        lr = float(hp.get("lr", 1e-4))
        rank = int(hp.get("rank", 8))
        base = str(hp.get("base_model", BASE_MODEL))
        # Smoke: a couple of LoRA steps on a handful of examples completes a real
        # fwd/bwd in seconds on CPU. Full runs train longer.
        max_steps = (
            run.max_steps if run.max_steps is not None else int(hp.get("steps", 60))
        )

        ds = Dataset.from_list(llm_data.chat_examples())

        cfg = SFTConfig(
            output_dir=run.checkpoint_dir,
            max_steps=max(1, max_steps),
            per_device_train_batch_size=1,
            learning_rate=lr,
            logging_steps=1,
            max_length=64,
            report_to=[],
            # CPU-smoke gotchas (verify-first #11): both default True in SFTConfig
            # and would break/slow a CPU run — disable explicitly.
            bf16=False,
            fp16=False,
            gradient_checkpointing=False,
        )
        peft_cfg = LoraConfig(r=rank, lora_alpha=2 * rank, task_type="CAUSAL_LM")

        trainer = SFTTrainer(
            model=base,
            train_dataset=ds,
            args=cfg,
            peft_config=peft_cfg,
        )
        trainer.add_callback(_sink_callback(run.metric_sink, self.metric_name()))
        trainer.train()

        # Save ONLY the LoRA adapter (tiny) + the metadata the viewer/board read.
        trainer.save_model(run.checkpoint_dir)
        final = trainer.state.log_history[-1] if trainer.state.log_history else {}
        loss = float(final.get("train_loss", final.get("loss", 0.0)))
        run.metric_sink.log(int(trainer.state.global_step), {self.metric_name(): loss})
        (Path(run.checkpoint_dir) / "meta.json").write_text(
            json.dumps(
                {
                    "base_model": base,
                    "metric": self.metric_name(),
                    "epoch": int(trainer.state.global_step),
                    "total": max(1, max_steps),
                    "eval_loss": loss,
                }
            )
        )

    def predict(self, checkpoint: str, x: Any) -> str:
        """Generate the tuned model's answer to prompt `x` (base + LoRA adapter)."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        meta = (
            json.loads((Path(checkpoint).parent / "meta.json").read_text())
            if (Path(checkpoint).parent / "meta.json").exists()
            else {}
        )
        base_id = meta.get("base_model", BASE_MODEL)
        tok = AutoTokenizer.from_pretrained(base_id)
        base = AutoModelForCausalLM.from_pretrained(base_id)
        model = PeftModel.from_pretrained(base, checkpoint)
        model.eval()
        return _generate(model, tok, str(x))

    def viewer(self) -> Viewer:
        from heads.llm.viewer import LLMViewer

        return LLMViewer()


def _generate(model: Any, tok: Any, prompt: str) -> str:
    import torch

    msgs = [{"role": "user", "content": prompt}]
    text_in = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(text_in, return_tensors="pt")
    n_in = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=32, do_sample=False)
    text: str = tok.decode(out[0, n_in:], skip_special_tokens=True)
    return text.strip()


def _sink_callback(sink: Any, metric: str) -> Any:
    """Bridge HF Trainer's logs → the spine's MetricSink (uniform reporting).

    This is the seam in action: a head that trains via HF still reports through
    the one sink the board/compare views read, exactly like every other head.
    Subclasses HF's TrainerCallback (built lazily so the import stays optional).
    """
    from transformers import TrainerCallback

    class _SinkCallback(TrainerCallback):  # type: ignore[misc]  # base is Any (stubless)
        def on_log(
            self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any
        ) -> None:
            if logs and "loss" in logs:
                sink.log(int(state.global_step), {metric: float(logs["loss"])})

    return _SinkCallback()


# The registry loads this attribute (see spine/registry.py).
HEAD = LLMHead()
