"""
training/dumb_examiner.py — Untrained baseline using the SAME base model
as the GRPO-trained examiner.

Scientific design (the "right" before/after baseline):
  Same architecture       — Qwen2.5-1.5B-Instruct (or whatever model is passed)
  Same prompt builder     — training.prompt_builder.build_prompt
  Same env / seeds / KB   — driven by training.eval.run_eval
  Same inference path     — Unsloth FastLanguageModel.from_pretrained,
                            generate(do_sample=False)
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

import json
import re
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

# Strip ```json ... ``` / ``` ... ``` markdown fences if present.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_first_json(text: str) -> str:
    """Return the first balanced {...} JSON object in `text`, or `text` raw.

    The untrained Qwen2.5-1.5B-Instruct is a chat model — it tends to wrap
    its output in markdown fences ("```json ... ```") or prefix it with
    explanatory prose ("I'll ask about ... {...}"). The env's strict JSON
    parser rejects all of that. The trained model never needs this helper
    because GRPO teaches it to emit raw JSON.

    This is *honest preprocessing*: we don't change what the model said,
    we just locate the JSON object inside the chatty wrapper. If there is
    no balanced object, return the original text so the env's parse
    failure is surfaced as parse_failure_rate.
    """
    if not text:
        return text

    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    return candidate
    return text[start:]


@dataclass
class DumbExaminer:
    """
    Untrained examiner: base LLM + prompt builder + JSON extractor.

    Mirrors `_TrainedExaminerWrapper` in train_grpo.py so eval semantics are
    identical between the two — only the weights differ (no LoRA). The one
    extra step is `_extract_first_json` to fish the JSON out of the chat
    model's chatty wrapper text; the trained wrapper doesn't need this
    because GRPO already teaches it to emit raw JSON.
    """

    model: Any
    tokenizer: Any
    max_new_tokens: int = 256

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
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return _extract_first_json(raw)


def load_dumb_examiner(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length: int = 1024,
    use_4bit: bool = True,
    max_new_tokens: int = 256,
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
        max_new_tokens: Eval-only generation cap. 256 is required because
                        a valid CLASSIFY JSON for 10 sections is ~85 tokens
                        and the chat model often emits a short preamble
                        before it; 64 was empirically too short and
                        produced parse_failure_rate=1.0.

    Returns:
        A DumbExaminer ready to be passed to training.eval.run_eval().
    """
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

    Tests _extract_first_json on the failure modes seen in the wild plus
    the prompt-builder wiring with a tokenizer/model stub.
    """

    # 1. Markdown fence
    s1 = '```json\n{"action_type": "ask", "section_id": "S01", "question_text": "x"}\n```'
    e1 = _extract_first_json(s1)
    assert json.loads(e1)["action_type"] == "ask"
    print(f"[OK] fence: {e1!r}")

    # 2. Chat preamble
    s2 = ('I think I should ask about gradient descent. '
          '{"action_type": "ask", "section_id": "S01", "question_text": "Why?"}')
    e2 = _extract_first_json(s2)
    assert json.loads(e2)["section_id"] == "S01"
    print(f"[OK] preamble: {e2!r}")

    # 3. Nested CLASSIFY (the failing case from the Space run)
    s3 = ('Sure, here is my classification: '
          '{"action_type": "classify", "classifications": '
          '{"S01": "KNOWS", "S02": "FAKING", "S03": "KNOWS", "S04": "FAKING", '
          '"S05": "KNOWS", "S06": "FAKING", "S07": "KNOWS", "S08": "FAKING", '
          '"S09": "KNOWS", "S10": "FAKING"}}')
    e3 = _extract_first_json(s3)
    parsed = json.loads(e3)
    assert parsed["action_type"] == "classify"
    assert len(parsed["classifications"]) == 10
    print(f"[OK] nested CLASSIFY ({len(e3)} chars)")

    # 4. No JSON at all — should return the input
    s4 = "I have no idea."
    e4 = _extract_first_json(s4)
    assert e4 == s4
    print(f"[OK] no-json passthrough: {e4!r}")

    # 5. Truncated mid-string (token cap hit) — fallback
    s5 = '{"action_type": "ask", "section_id": "S01", "question_text": "Why doe'
    e5 = _extract_first_json(s5)
    assert e5.startswith("{")
    print(f"[OK] truncated passthrough: {e5!r}")

    # 6. End-to-end DumbExaminer with stubs
    class _StubTokenizer:
        eos_token_id = 0

        def __call__(self, text, return_tensors=None):
            class _Out(dict):
                def to(self, _device):
                    return self
            return _Out({"input_ids": torch.tensor([[1, 2, 3, 4, 5]])})

        def decode(self, ids, skip_special_tokens=True):
            return ('Sure, here is my answer: '
                    '```json\n{"action_type": "ask", "section_id": "S01", '
                    '"question_text": "What is gradient descent?"}\n```')

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
    assert parsed["action_type"] == "ask"
    print(f"[OK] DumbExaminer.act -> {out}")
    print("\n[OK] All smoke tests passed.")
