"""
scripts/test_reward_rollout.py — End-to-end local integration test for the
reward rollout pipeline in training/train_grpo.py.

Tests _rollout_completion and _make_grpo_reward_func against the real
ExaminerEnv without requiring Unsloth, TRL, transformers, or a GPU.

Run from repo root:
    python scripts/test_reward_rollout.py
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from examiner_env.knowledge_base import KB
from training.config import DEBUG_CONFIG
from training.train_grpo import _rollout_completion, _make_grpo_reward_func, _iter_json_objects

FAILED: list[str] = []
PASSED: list[str] = []


def _chk(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED.append(name)
        import traceback
        print(f"  FAIL  {name}")
        traceback.print_exc()


# ─── helpers ────────────────────────────────────────────────────────────

def _ask(sid, q):
    return json.dumps({"action_type": "ask", "section_id": sid, "question_text": q})

def _classify(**labels):
    return json.dumps({"action_type": "classify", "classifications": labels})

SECTIONS = ["S01","S02","S03","S04","S05","S06","S07","S08","S09","S10"]

def _full_classify(verdict="KNOWS"):
    return json.dumps({"action_type": "classify",
                       "classifications": {s: verdict for s in SECTIONS}})


# ─── tests ──────────────────────────────────────────────────────────────

def t01_empty_completion_gets_forced_classify():
    """Empty completion → env exhausts turns → non-zero finite reward."""
    import math
    r, bd = _rollout_completion("", episode_seed=0, kb=KB, config=DEBUG_CONFIG)
    assert math.isfinite(r), f"reward must be finite, got {r}"
    assert -2.1 <= r <= 2.0, f"reward out of bounds: {r}"


def t02_single_valid_classify():
    """Completion with one classify action → reward from env._finalise."""
    import math
    comp = _full_classify("KNOWS")
    r, bd = _rollout_completion(comp, episode_seed=1, kb=KB, config=DEBUG_CONFIG)
    assert math.isfinite(r)
    assert -2.1 <= r <= 2.0


def t03_ask_then_classify_pipeline():
    """Ask S01, Ask S03, then classify → reward includes info-gain."""
    import math
    comp = (
        _ask("S01", "Why does dropout prevent overfitting in neural networks?")
        + "\n"
        + _ask("S03", "How does backpropagation compute gradients through a chain?")
        + "\n"
        + _full_classify("FAKING")
    )
    r, bd = _rollout_completion(comp, episode_seed=2, kb=KB, config=DEBUG_CONFIG)
    assert math.isfinite(r)
    # If bd is not None the env ran all the way through _finalise.
    assert bd is not None, "expected a RewardBreakdown from classify action"


def t04_malformed_json_gets_penalty():
    """All-garbage completion → repeated forced-classify penalty path."""
    import math
    comp = "this is not json at all {unclosed"
    r, bd = _rollout_completion(comp, episode_seed=3, kb=KB, config=DEBUG_CONFIG)
    assert math.isfinite(r)
    assert r < 0, f"all-malformed completion should yield negative reward, got {r}"


def t05_partial_classify_keys_accepted_by_env():
    """
    Classify with fewer than 10 sections → ClassifyAction fails Pydantic
    validation → MalformedAction → env force-classifies → reward still finite.
    """
    import math
    partial = json.dumps({"action_type": "classify",
                          "classifications": {"S01": "KNOWS", "S02": "FAKING"}})
    r, bd = _rollout_completion(partial, episode_seed=4, kb=KB, config=DEBUG_CONFIG)
    assert math.isfinite(r)


def t06_reward_always_bounded():
    """Fuzz: 20 random seeds with random ask+classify sequences stay in bounds."""
    import math, random
    rng = random.Random(42)
    for seed in range(20):
        n_ask = rng.randint(0, 4)
        sids = rng.choices(SECTIONS, k=n_ask)
        parts = [_ask(s, f"How does {s} work in ML?") for s in sids]
        parts.append(_full_classify(rng.choice(["KNOWS", "FAKING"])))
        comp = "\n".join(parts)
        r, _ = _rollout_completion(comp, episode_seed=seed, kb=KB, config=DEBUG_CONFIG)
        assert math.isfinite(r), f"seed {seed}: non-finite reward {r}"
        assert -2.1 <= r <= 2.0, f"seed {seed}: reward {r} out of bounds"


def t07_reward_func_batch_returns_correct_length():
    """_make_grpo_reward_func returns a list the same length as completions."""
    rf = _make_grpo_reward_func(KB, DEBUG_CONFIG)
    comps = [_full_classify("KNOWS")] * 3 + [_full_classify("FAKING")] * 2
    seeds = list(range(5))
    result = rf(
        prompts=["dummy"] * 5,
        completions=comps,
        episode_seed=seeds,
    )
    assert len(result) == 5, f"expected 5 rewards, got {len(result)}"
    assert all(isinstance(r, float) for r in result), "all rewards must be float"


def t08_reward_func_no_seeds_fallback():
    """reward_func works when episode_seed column is missing."""
    import math
    rf = _make_grpo_reward_func(KB, DEBUG_CONFIG)
    comps = [_full_classify("KNOWS"), "garbage"]
    result = rf(prompts=["p"] * 2, completions=comps)
    assert len(result) == 2
    assert all(math.isfinite(r) for r in result)


def t09_reward_func_chat_format_completion():
    """
    Unsloth compiled trainer may pass completions as chat-format lists
    [{"role": "assistant", "content": "..."}]. Verify reward_func handles them.
    """
    import math
    rf = _make_grpo_reward_func(KB, DEBUG_CONFIG)
    comp_chat = [{"role": "assistant", "content": _full_classify("FAKING")}]
    result = rf(
        prompts=["p"],
        completions=[comp_chat],
        episode_seed=[7],
    )
    assert len(result) == 1
    assert math.isfinite(result[0])


def t10_iter_json_objects_stress():
    """Stream with 4 objects, nested braces, unicode, escaped quotes."""
    stream = (
        'preamble {"action_type":"ask","section_id":"S01",'
        '"question_text":"What is a \\"chain rule\\"?"} text '
        '{"action_type":"ask","section_id":"S02","question_text":"Explain SGD"} '
        '{"nested":{"key":"val","arr":[1,2,3]}} '
        'trailing {"action_type":"classify","classifications":'
        + json.dumps({s: "KNOWS" for s in SECTIONS}) + "} end"
    )
    objs = list(_iter_json_objects(stream))
    assert len(objs) == 4, f"expected 4 objects, got {len(objs)}"
    a = json.loads(objs[0]); assert a["action_type"] == "ask"
    b = json.loads(objs[1]); assert b["section_id"] == "S02"
    c = json.loads(objs[2]); assert c["nested"]["arr"] == [1, 2, 3]
    d = json.loads(objs[3]); assert d["action_type"] == "classify"


# ─── runner ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Reward rollout pipeline — local integration test")
    print("(uses real ExaminerEnv, no Unsloth/TRL/GPU needed)")
    print("=" * 70)

    tests = [
        ("01. empty completion → forced classify",               t01_empty_completion_gets_forced_classify),
        ("02. single valid classify action",                      t02_single_valid_classify),
        ("03. ask x2 then classify - non-null breakdown",           t03_ask_then_classify_pipeline),
        ("04. malformed completion → negative reward",            t04_malformed_json_gets_penalty),
        ("05. partial classify keys → finite reward",             t05_partial_classify_keys_accepted_by_env),
        ("06. fuzz 20 seeds stay in bounds [-2.1, 2.0]",          t06_reward_always_bounded),
        ("07. batch reward_func returns correct length",           t07_reward_func_batch_returns_correct_length),
        ("08. reward_func fallback when no episode_seed col",     t08_reward_func_no_seeds_fallback),
        ("09. chat-format completion list handled",               t09_reward_func_chat_format_completion),
        ("10. JSON stream extraction stress (4 objects)",          t10_iter_json_objects_stress),
    ]

    for name, fn in tests:
        _chk(name, fn)

    print()
    print("=" * 70)
    print(f"PASSED {len(PASSED)} / {len(tests)}   FAILED {len(FAILED)}")
    print("=" * 70)
    if FAILED:
        for n in FAILED:
            print(f"  - {n}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
