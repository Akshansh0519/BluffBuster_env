from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weight(vocab: int, hidden: int) -> "torch.nn.Parameter":
    return torch.nn.Parameter(torch.randn(vocab, hidden))


# ---------------------------------------------------------------------------
# Case 1 — original observed call: hidden(N, D) + Parameter(V, D) + ids(M,)
# Output must be 1-D (M_trimmed,) where M_trimmed = min(N, M)
# ---------------------------------------------------------------------------

def test_1d_hidden_parameter_weight_1d_ids():
    from training.train_grpo import _safe_selective_log_softmax_for_unsloth

    hidden = torch.randn(3, 4)          # (N=3, D=4)
    w = _make_weight(vocab=8, hidden=4) # (V=8, D=4)
    ids = torch.tensor([2, 0])          # (M=2,)  — shorter than N

    out = _safe_selective_log_softmax_for_unsloth(
        hidden, w, ids,
        chunks=1, logit_scale_multiply=1.0,
        logit_scale_divide=1.0, logit_softcapping=None, temperature=1.0,
    )

    assert out.ndim == 1, f"expected 1-D, got {out.shape}"
    assert out.shape[0] == 2, f"expected length 2, got {out.shape}"


# ---------------------------------------------------------------------------
# Case 2 — 3-D batch: hidden(B, S, D) + Parameter(V, D) + ids(B, S)
# This is the grpo_accumulated_loss path that uses ref_logps.shape[1]
# Output MUST be 2-D (B, S) so that .shape[1] works
# ---------------------------------------------------------------------------

def test_3d_hidden_parameter_weight_2d_ids_returns_2d():
    from training.train_grpo import _safe_selective_log_softmax_for_unsloth

    B, S, D, V = 2, 5, 4, 8
    hidden = torch.randn(B, S, D)
    w = _make_weight(vocab=V, hidden=D)
    ids = torch.randint(0, V, (B, S))

    out = _safe_selective_log_softmax_for_unsloth(
        hidden, w, ids,
        chunks=1, logit_scale_multiply=1.0,
        logit_scale_divide=1.0, logit_softcapping=None, temperature=1.0,
    )

    assert out.ndim == 2, f"expected 2-D (B,S), got {out.shape}"
    assert out.shape == (B, S), f"expected {(B, S)}, got {out.shape}"


# ---------------------------------------------------------------------------
# Case 3 — shape mismatch on seq dim (env_factory variable-length)
# hidden(B, S1, D), ids(B, S2) where S1 != S2 — must not crash, output 2-D
# ---------------------------------------------------------------------------

def test_seq_dim_mismatch_returns_2d():
    from training.train_grpo import _safe_selective_log_softmax_for_unsloth

    B, S1, S2, D, V = 2, 7, 5, 4, 8
    hidden = torch.randn(B, S1, D)
    w = _make_weight(vocab=V, hidden=D)
    ids = torch.randint(0, V, (B, S2))

    out = _safe_selective_log_softmax_for_unsloth(
        hidden, w, ids,
        chunks=1, logit_scale_multiply=1.0,
        logit_scale_divide=1.0, logit_softcapping=None, temperature=1.0,
    )

    assert out.ndim == 2, f"expected 2-D, got {out.shape}"
    assert out.shape[0] == B
    assert out.shape[1] == min(S1, S2)


# ---------------------------------------------------------------------------
# Case 4 — numerical correctness (exact match against reference)
# ---------------------------------------------------------------------------

def test_numerical_correctness_2d_output():
    from training.train_grpo import _safe_selective_log_softmax_for_unsloth
    import torch.nn.functional as F

    B, S, D, V = 2, 3, 4, 8
    hidden = torch.randn(B, S, D)
    w = _make_weight(vocab=V, hidden=D)
    ids = torch.randint(0, V, (B, S))

    out = _safe_selective_log_softmax_for_unsloth(
        hidden, w, ids,
        chunks=1, logit_scale_multiply=1.0,
        logit_scale_divide=1.0, logit_softcapping=None, temperature=1.0,
    )

    logits = hidden.float() @ w.float().t()          # (B, S, V)
    expected = F.log_softmax(logits, dim=-1)          # (B, S, V)
    expected = expected.gather(-1, ids.unsqueeze(-1)).squeeze(-1)  # (B, S)

    assert torch.allclose(out, expected, atol=1e-5), \
        f"max diff: {(out - expected).abs().max()}"

