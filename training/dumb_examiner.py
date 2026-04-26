"""
training/dumb_examiner.py — Untrained baseline using the SAME base model
as the GRPO-trained examiner.

Scientific design (the "right" before/after baseline):
  Same architecture       — Qwen2.5-1.5B-Instruct (or whatever model is passed)
  Same prompt builder     — training.prompt_builder.build_prompt
  Same env / seeds / KB   — driven by training.eval.run_eval
  Same inference path     — Unsloth FastLanguageModel.from_pretrained,
                            generate(do_sample=False, max_new_tokens=128)
  Only difference         — NO GRPO LoRA adapter is applied

Any improvement measured between this baseline and the trained checkpoint is
therefore directly attributable to GRPO training, not to model size, prompt
engineering, env changes, or seed shuffling.

Interface matches the existing baselines in examiner_env/baselines.py
(RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer):
    examiner.reset(section_ids: list[str] | None = None) -> None
    examiner.act(observation: dict) -> str   # JSON string

Owner: C2 (training/, scripts/, hf_space/)
This file does NOT modify any C1-owned file (examiner_env/, eval.py,
prompt_builder.py is C2-owned).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from training.prompt_builder import build_prompt


# Unsloth pre-quantized weights — used when load_in_4bit=True to avoid the
# "CUDA device-side assert during rotary init" caused by on-the-fly bnb
# quantization of raw HF Qwen weights (see FOUNDATION/mistakes.md M044).
_PREQUANT_ALIAS = {
    "Qwen/Qwen2.5-1.5B-Instruct": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    "Qwen/Qwen2.5-3B-Instruct":   "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "Qwen/Qwen2.5-7B-Instruct":   "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
}


@dataclass
class DumbExaminer:
    """
    Untrained examiner: base LLM + prompt builder, nothing else.

    Mirrors `_TrainedExaminerWrapper` in train_grpo.py so eval semantics are
    identical between the two — only the weights differ (no LoRA).
    """

    model: Any
    tokenizer: Any
    max_new_tokens: int = 64

    def reset(self, section_ids: list[str] | None = None) -> None:
        # Stateless examiner — nothing to reset between episodes.
        return None

    def act(self, observation: dict) -> str:
        prompt = build_prompt(observation)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_dumb_examiner(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length: int = 1024,
    use_4bit: bool = True,
    max_new_tokens: int = 64,
) -> DumbExaminer:
    """
    Load the base model with Unsloth and wrap it in DumbExaminer.

    NO LoRA adapter is attached — that is the entire point. The model is the
    raw, untrained Qwen2.5-X-Instruct exactly as it was downloaded from HF.

    Args:
        model_name:     HF model id. Defaults to Qwen2.5-1.5B-Instruct (matches
                        FAST_CONFIG on feat/fast-1.5b-training).
        max_seq_length: Same default (1024) as FAST_CONFIG.
        use_4bit:       Load in 4-bit via bitsandbytes. Set False to load fp16.
        max_new_tokens: Eval-only generation cap. 64 is enough for one valid
                        ASK or CLASSIFY JSON; halves wall-clock vs 128.

    Returns:
        A DumbExaminer ready to be passed to training.eval.run_eval().
    """
    # Imported lazily so this module is importable on machines without Unsloth
    # (CPU dev boxes) — Unsloth requires CUDA and triton.
    from unsloth import FastLanguageModel

    resolved = model_name
    if use_4bit and not model_name.startswith("unsloth/"):
        alias = _PREQUANT_ALIAS.get(model_name)
        if alias is not None:
            print(
                f"[dumb_baseline] substituting pre-quantized weights: "
                f"{model_name} -> {alias}"
            )
            resolved = alias

    print(f"[dumb_baseline] Loading {resolved} (4-bit={use_4bit})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resolved,
        max_seq_length=max_seq_length,
        load_in_4bit=use_4bit,
        dtype=None,
    )

    # Unsloth's inference fast-path. Doubles generation throughput by skipping
    # gradient bookkeeping. Safe because we never train this model.
    if hasattr(FastLanguageModel, "for_inference"):
        try:
            FastLanguageModel.for_inference(model)
            print("[dumb_baseline] Unsloth inference fast-path enabled.")
        except Exception as e:  # pragma: no cover — defensive
            print(f"[dumb_baseline] for_inference unavailable ({e!r}); continuing.")

    return DumbExaminer(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )


# ── Smoke test (CPU-only, no Unsloth required) ──────────────────────────────
if __name__ == "__main__":
    """
    Run with:  python -m training.dumb_examiner

    Verifies the prompt-builder wiring with a tokenizer/model stub. The full
    Unsloth load is GPU-only and is exercised by scripts/run_dumb_baseline.py.
    """
    import json

    class _StubTokenizer:
        eos_token_id = 0

        def __call__(self, text, return_tensors=None):
            class _Out(dict):
                def to(self, _device):
                    return self
            return _Out({"input_ids": torch.tensor([[1, 2, 3, 4, 5]])})

        def decode(self, ids, skip_special_tokens=True):
            return json.dumps({
                "action_type": "ask",
                "section_id": "S01",
                "question_text": "What is gradient descent?",
            })

    class _StubModel:
        device = "cpu"

        def generate(self, **kw):
            input_ids = kw["input_ids"]
            extra = torch.tensor([[6, 7, 8]])
            return torch.cat([input_ids, extra], dim=1)

    examiner = DumbExaminer(model=_StubModel(), tokenizer=_StubTokenizer())
    obs = {
        "section_titles": {"S01": "Gradient Descent"},
        "section_ids": ["S01"],
        "turn": 0,
        "remaining_turns": 4,
        "dialogue_history": [],
    }
    out = examiner.act(obs)
    parsed = json.loads(out)
    assert parsed["action_type"] in ("ask", "classify")
    print(f"[OK] DumbExaminer.act -> {out}")
    print("[OK] Smoke test passed.")
