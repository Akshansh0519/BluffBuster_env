"""
tests/test_posterior_oracle.py — Phase 1 posterior oracle tests.
"""
from __future__ import annotations

import math

import pytest

from examiner_env.knowledge_base import KB
from examiner_env.models import CANONICAL_SECTIONS
from examiner_env.posterior_oracle import (
    LLR_CLIP,
    P0,
    PosteriorTracker,
    _binary_entropy,
    score_response,
)

# ──────────────────────────────────────────────────────────────────────────────
# Rich KNOWS response: contains multiple mechanism cue phrases
# ──────────────────────────────────────────────────────────────────────────────
_KNOWS_RESPONSE_S01 = (
    "The gradient of the loss with respect to parameters is computed at each step. "
    "The learning rate scales the magnitude of each update step, so a higher learning rate "
    "takes larger steps down the loss surface. "
    "Momentum accumulates an exponentially decaying average of past gradients, "
    "which helps traverse ravines in the loss landscape faster."
)

# Misconception-heavy FAKING response
_FAKING_RESPONSE_S01 = (
    "Gradient descent always finds the global minimum of any loss function, "
    "which is why a larger learning rate always trains the model faster without quality loss. "
    "Gradient descent only works on convex loss functions."
)

_SECTION = "S01"


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Initial posteriors are 0.5
# ──────────────────────────────────────────────────────────────────────────────

def test_initial_posteriors_at_prior():
    tracker = PosteriorTracker(CANONICAL_SECTIONS)
    posteriors = tracker.current_posteriors()
    for s in CANONICAL_SECTIONS:
        assert abs(posteriors[s] - P0) < 1e-9, (
            f"Prior for {s} should be {P0}, got {posteriors[s]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: KNOWS response pushes posterior above 0.5
# ──────────────────────────────────────────────────────────────────────────────

def test_knows_response_increases_posterior():
    tracker = PosteriorTracker(CANONICAL_SECTIONS)
    delta_h = tracker.update(_SECTION, _KNOWS_RESPONSE_S01, KB)
    p = tracker.posterior(_SECTION)
    assert p > 0.5, (
        f"After mechanism-rich response, posterior for {_SECTION} should exceed 0.5, got {p}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: FAKING response pushes posterior below 0.5
# ──────────────────────────────────────────────────────────────────────────────

def test_faking_response_decreases_posterior():
    tracker = PosteriorTracker(CANONICAL_SECTIONS)
    tracker.update(_SECTION, _FAKING_RESPONSE_S01, KB)
    p = tracker.posterior(_SECTION)
    assert p < 0.5, (
        f"After misconception-heavy response, posterior for {_SECTION} should be below 0.5, got {p}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: LLR clip [-3, +3] is enforced
# ──────────────────────────────────────────────────────────────────────────────

def test_llr_clip_is_enforced():
    """Synthesise a maximally mechanism-rich response and check LLR doesn't exceed 3."""
    # Build a response that includes all strong mechanism cues
    section_kb = KB[_SECTION]
    all_cues = " ".join(cue.phrase for cue in section_kb.mechanism_cues)
    llr = score_response(all_cues, _SECTION, KB)
    assert -LLR_CLIP <= llr <= LLR_CLIP, (
        f"LLR={llr} not clipped to [{-LLR_CLIP}, {LLR_CLIP}]"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: ΔH_t is positive for informative responses
# ──────────────────────────────────────────────────────────────────────────────

def test_delta_h_positive_for_informative_response():
    """An informative response should reduce entropy (ΔH_t > 0)."""
    tracker = PosteriorTracker(CANONICAL_SECTIONS)
    delta_h = tracker.update(_SECTION, _KNOWS_RESPONSE_S01, KB)
    assert delta_h > 0, (
        f"Expected positive ΔH_t for informative response, got {delta_h}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: Uninformative response (empty) → ΔH_t ≈ 0
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_response_minimal_delta_h():
    """Empty response contains no cues → LLR=0 → posterior unchanged → ΔH_t=0."""
    tracker = PosteriorTracker(CANONICAL_SECTIONS)
    delta_h = tracker.update(_SECTION, "", KB)
    p = tracker.posterior(_SECTION)
    assert abs(p - 0.5) < 1e-9, f"Empty response should not change posterior, got p={p}"
    assert abs(delta_h) < 1e-9, f"Empty response should give ΔH_t≈0, got {delta_h}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: Entropy trace records history
# ──────────────────────────────────────────────────────────────────────────────

def test_entropy_trace_records_per_update():
    tracker = PosteriorTracker(CANONICAL_SECTIONS)
    tracker.update("S01", _KNOWS_RESPONSE_S01, KB)
    tracker.update("S02", "The chain rule is applied to the Jacobian of the loss.", KB)
    tracker.update("S03", "Dropout randomly zeroes activations during training.", KB)

    gains = tracker.entropy_gains()
    assert len(gains) == 3, f"Expected 3 entropy gain entries, got {len(gains)}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: snapshot() records posterior snapshots
# ──────────────────────────────────────────────────────────────────────────────

def test_snapshot_records_posteriors():
    tracker = PosteriorTracker(CANONICAL_SECTIONS)
    tracker.update("S01", _KNOWS_RESPONSE_S01, KB)
    tracker.snapshot()
    tracker.update("S01", _KNOWS_RESPONSE_S01, KB)
    tracker.snapshot()

    history = tracker.history()
    assert len(history) == 2, f"Expected 2 snapshot entries, got {len(history)}"
    # Second snapshot should show higher posterior for S01
    assert history[1]["S01"] >= history[0]["S01"], (
        "Second update with positive evidence should increase or maintain S01 posterior"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 9: binary_entropy edge cases
# ──────────────────────────────────────────────────────────────────────────────

def test_binary_entropy_boundary():
    assert _binary_entropy(0.0) == 0.0, "H(0) should be 0"
    assert _binary_entropy(1.0) == 0.0, "H(1) should be 0"
    assert abs(_binary_entropy(0.5) - 1.0) < 1e-9, f"H(0.5) should be 1 bit, got {_binary_entropy(0.5)}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 10: Total info gain is non-negative for a realistic episode
# ──────────────────────────────────────────────────────────────────────────────

def test_total_info_gain_nonnegative():
    tracker = PosteriorTracker(CANONICAL_SECTIONS)
    tracker.update("S01", _KNOWS_RESPONSE_S01, KB)
    tracker.update("S02", "The chain rule decomposes gradients through Jacobian.", KB)
    info_gain = tracker.total_info_gain()
    assert info_gain >= 0, f"Total info gain should be ≥ 0, got {info_gain}"
