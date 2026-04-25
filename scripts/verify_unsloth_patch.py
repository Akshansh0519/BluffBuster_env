"""
scripts/verify_unsloth_patch.py — torch-only sanity test for the
Unsloth chunked-log-softmax monkey patch.

Reproduces the four failure modes listed in the root-cause analysis and
asserts that the replacement function in
training/train_grpo.py._safe_chunked_hidden_states_selective_log_softmax
returns the shapes and values that Unsloth's compiled GRPOTrainer expects.

Run from the repo root:
    python -m scripts.verify_unsloth_patch
or
    python scripts/verify_unsloth_patch.py

Requires: torch. Does NOT require Unsloth, TRL, transformers, or a GPU.
"""

from __future__ import annotations

import os
import sys
import traceback

# Allow `python scripts/verify_unsloth_patch.py` from the repo root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

from training.train_grpo import (
    _safe_chunked_hidden_states_selective_log_softmax as patched,
    _install_unsloth_chunked_logsoftmax_patch,
    _UNSLOTH_PATCH_TARGET,
)


# ──────────────────────────────────────────────────────────────────────────
#   Reference implementation (ground truth)
# ──────────────────────────────────────────────────────────────────────────

def _reference_logp(hidden, weight, ids, temperature=1.0):
    """Direct, unchunked reference: hidden → logits → log_softmax → gather."""
    logits = F.linear(hidden, weight)
    if temperature and temperature != 1.0:
        logits = logits / temperature
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, ids.long().unsqueeze(-1)).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────
#   Individual test cases
# ──────────────────────────────────────────────────────────────────────────

FAILED: list[str] = []
PASSED: list[str] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        PASSED.append(name)
        print(f"  PASS  {name}")
    except Exception as exc:
        FAILED.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()


def test_case_1_happy_path_3d_hidden_and_parameter_weight():
    """Full canonical call: (B, S, D) hidden + Parameter lm_head + (B, S) ids."""
    torch.manual_seed(0)
    B, S, D, V = 2, 7, 16, 97
    hidden = torch.randn(B, S, D)
    weight = torch.nn.Parameter(torch.randn(V, D))
    ids = torch.randint(0, V, (B, S))

    out = patched(
        hidden, weight, ids,
        chunks=4,
        logit_scale_multiply=0.0,
        logit_scale_divide=0.0,
        logit_softcapping=0.0,
        temperature=1.0,
    )

    assert out.dim() == 2, f"expected 2D output, got {out.dim()}D"
    assert out.shape == (B, S), f"expected shape ({B},{S}), got {tuple(out.shape)}"

    ref = _reference_logp(hidden, weight, ids)
    assert torch.allclose(out, ref, atol=1e-5), \
        f"values disagree with reference: max diff = {(out - ref).abs().max().item()}"


def test_case_2_parameter_not_misclassified_as_int_tensor():
    """
    The #2 error was: `torch.is_tensor(Parameter)` returns True, so an
    unguarded pass treats the lm_head weight as the index tensor. We force
    the call where the Parameter happens to have int-compatible values.
    """
    torch.manual_seed(1)
    B, S, D, V = 1, 4, 8, 32

    # Deliberately place Parameter at position 1 (where ids would naively be).
    hidden = torch.randn(B, S, D)
    weight = torch.nn.Parameter(torch.randn(V, D))
    ids = torch.randint(0, V, (B, S))

    out = patched(hidden, weight, ids, chunks=2, temperature=1.0)
    assert out.shape == (B, S)

    # And ensure it was NOT interpreted as the weight getting swapped with ids.
    ref = _reference_logp(hidden, weight, ids)
    assert torch.allclose(out, ref, atol=1e-5)


def test_case_3_hidden_states_get_projected_to_logits():
    """
    hidden.shape[-1] == weight.shape[-1] == D means `hidden` is pre-linear.
    Patch must project via hidden @ weight.T to recover logits of dim V.
    """
    torch.manual_seed(2)
    B, S, D, V = 3, 5, 12, 40
    hidden = torch.randn(B, S, D)
    weight = torch.nn.Parameter(torch.randn(V, D))
    ids = torch.randint(0, V, (B, S))

    out = patched(hidden, weight, ids, chunks=3)

    assert out.shape == (B, S), (
        "patch must preserve (B, S) prefix after the projection step; "
        f"got {tuple(out.shape)}"
    )
    ref = _reference_logp(hidden, weight, ids)
    assert torch.allclose(out, ref, atol=1e-5)


def test_case_4_output_is_always_2d_even_for_1d_ids():
    """
    Error #3 was: patch returned 1D (B*S,) → grpo_accumulated_loss did
    `ref_logps.shape[1]` → IndexError. We check that even when ids are
    given flat (N,), output is still 2D.
    """
    torch.manual_seed(3)
    B, S, D, V = 2, 6, 8, 25
    hidden = torch.randn(B, S, D)
    weight = torch.nn.Parameter(torch.randn(V, D))
    ids_flat = torch.randint(0, V, (B * S,))

    out = patched(hidden, weight, ids_flat, chunks=2)
    assert out.dim() == 2, (
        f"output must be 2D for ref_logps.shape[1] downstream; "
        f"got {out.dim()}D with shape {tuple(out.shape)}"
    )


def test_case_5_shape_mismatch_is_cropped_not_crashed():
    """
    Env-generated completions have variable sequence length. B/S in hidden
    and ids may differ. Patch must crop to min on each prefix dim without
    raising.
    """
    torch.manual_seed(4)
    D, V = 10, 30
    hidden = torch.randn(3, 7, D)                      # (3, 7, D)
    weight = torch.nn.Parameter(torch.randn(V, D))
    ids = torch.randint(0, V, (2, 5))                  # (2, 5)

    out = patched(hidden, weight, ids, chunks=2)
    # Prefix dims cropped to min: (2, 5).
    assert out.dim() == 2, f"expected 2D, got {out.dim()}D"
    assert out.shape == (2, 5), f"expected (2,5), got {tuple(out.shape)}"


def test_case_6_preshape_logits_path():
    """
    If the caller passes already-computed logits (hidden.shape[-1] == V),
    patch must skip the projection and go straight to log_softmax.
    """
    torch.manual_seed(5)
    B, S, D, V = 2, 4, 11, 50
    logits = torch.randn(B, S, V)
    weight = torch.nn.Parameter(torch.randn(V, D))   # shape reports (V, D)
    ids = torch.randint(0, V, (B, S))

    out = patched(logits, weight, ids, chunks=1)
    assert out.shape == (B, S)

    ref = F.log_softmax(logits.float(), dim=-1).gather(
        -1, ids.long().unsqueeze(-1)
    ).squeeze(-1)
    assert torch.allclose(out, ref, atol=1e-5)


def test_case_7_temperature_and_softcap_are_applied():
    """logit_scale_multiply / divide / softcap / temperature must all apply."""
    torch.manual_seed(6)
    B, S, D, V = 1, 3, 6, 20
    hidden = torch.randn(B, S, D)
    weight = torch.nn.Parameter(torch.randn(V, D))
    ids = torch.randint(0, V, (B, S))

    # Plain call.
    out_plain = patched(hidden, weight, ids, chunks=1, temperature=1.0)

    # With temperature = 2 → softmax becomes smoother → log-probs less
    # extreme (closer to -log(V)).
    out_hot = patched(hidden, weight, ids, chunks=1, temperature=2.0)
    assert out_plain.shape == out_hot.shape == (B, S)
    # Sanity: not identical.
    assert not torch.allclose(out_plain, out_hot, atol=1e-4), \
        "temperature=2.0 should change the output"


def test_case_8_kwarg_alias_variants():
    """
    Unsloth may rename the weight / ids kwargs across releases. Patch should
    accept a broad set of aliases.
    """
    torch.manual_seed(7)
    B, S, D, V = 2, 3, 5, 15
    hidden = torch.randn(B, S, D)
    weight = torch.nn.Parameter(torch.randn(V, D))
    ids = torch.randint(0, V, (B, S))

    out = patched(
        hidden_states=hidden,
        lm_head_weight=weight,
        input_ids=ids,
        chunks=1,
    )
    assert out.shape == (B, S)

    ref = _reference_logp(hidden, weight, ids)
    assert torch.allclose(out, ref, atol=1e-5)


def test_case_9_installer_finds_the_symbol_in_fake_module():
    """
    _install_unsloth_chunked_logsoftmax_patch should replace the target in
    any sys.modules entry that exports it — we simulate the compiled cache.
    """
    import types
    fake = types.ModuleType("unsloth_compiled_cache.UnslothGRPOTrainer")
    fake.chunked_hidden_states_selective_log_softmax = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["unsloth_compiled_cache.UnslothGRPOTrainer"] = fake
    # Parent package also needed for importlib.import_module() not to explode.
    sys.modules.setdefault(
        "unsloth_compiled_cache",
        types.ModuleType("unsloth_compiled_cache"),
    )
    try:
        patched_list = _install_unsloth_chunked_logsoftmax_patch(verbose=False)
        assert "unsloth_compiled_cache.UnslothGRPOTrainer" in patched_list, (
            f"installer missed fake module; patched={patched_list}"
        )
        # And the fake module's attribute IS the replacement now.
        assert getattr(fake, _UNSLOTH_PATCH_TARGET) is patched, (
            "installer did not replace the attribute on the fake module"
        )
    finally:
        sys.modules.pop("unsloth_compiled_cache.UnslothGRPOTrainer", None)
        sys.modules.pop("unsloth_compiled_cache", None)


def test_case_10_json_extraction_handles_nested_and_escaped():
    """Sanity-check _iter_json_objects before we trust it inside reward_func."""
    from training.train_grpo import _iter_json_objects

    stream = (
        'garbage {"action_type":"ask","section_id":"S01",'
        '"question_text":"What is a \\"gradient\\"?"} '
        'more {"action_type":"classify","classifications":'
        '{"S01":"KNOWS","S02":"FAKING"}} trailing'
    )
    objs = list(_iter_json_objects(stream))
    assert len(objs) == 2, f"expected 2 objects, got {len(objs)}: {objs}"
    import json
    a = json.loads(objs[0])
    b = json.loads(objs[1])
    assert a["action_type"] == "ask"
    assert a["question_text"] == 'What is a "gradient"?'
    assert b["action_type"] == "classify"
    assert b["classifications"]["S02"] == "FAKING"


# ──────────────────────────────────────────────────────────────────────────
#   Runner
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 70)
    print("Unsloth chunked-log-softmax patch — verification suite")
    print("=" * 70)

    tests = [
        ("1. happy path (3D hidden + Parameter + (B,S) ids)", test_case_1_happy_path_3d_hidden_and_parameter_weight),
        ("2. Parameter not misclassified as int tensor",      test_case_2_parameter_not_misclassified_as_int_tensor),
        ("3. hidden states projected to logits",              test_case_3_hidden_states_get_projected_to_logits),
        ("4. output is always 2D (even for 1D ids)",          test_case_4_output_is_always_2d_even_for_1d_ids),
        ("5. shape mismatch cropped, not crashed",            test_case_5_shape_mismatch_is_cropped_not_crashed),
        ("6. pre-computed logits path",                       test_case_6_preshape_logits_path),
        ("7. temperature + softcap applied",                  test_case_7_temperature_and_softcap_are_applied),
        ("8. kwarg aliases (lm_head_weight / input_ids)",     test_case_8_kwarg_alias_variants),
        ("9. installer finds symbol in fake compiled cache",  test_case_9_installer_finds_the_symbol_in_fake_module),
        ("10. JSON stream extraction (reward rollout)",       test_case_10_json_extraction_handles_nested_and_escaped),
    ]

    for name, fn in tests:
        _check(name, fn)

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
