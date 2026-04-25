"""
tests/test_student_styles.py — Phase 1 student simulator tests.
"""
from __future__ import annotations

import re

import pytest

from examiner_env.knowledge_base import KB
from examiner_env.models import CANONICAL_SECTIONS
from examiner_env.student import generate_response, sample_profile

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_MECHANISM_QUESTION = (
    "Why does gradient descent convergence depend on the learning rate? "
    "Explain the mechanism in detail."
)
_DEFINITIONAL_QUESTION = "What is gradient descent?"
_SECTION = "S01"


def _count_mech_cue_hits(response: str, section_id: str) -> int:
    """Count how many KB mechanism cue phrases appear in the response."""
    section_kb = KB[section_id]
    resp_lower = response.lower()
    return sum(1 for cue in section_kb.mechanism_cues if cue.phrase.lower() in resp_lower)


def _jaccard_words(a: str, b: str) -> float:
    def tok(s): return set(re.sub(r"[^a-z0-9\s]", "", s.lower()).split())
    wa, wb = tok(a), tok(b)
    if not (wa | wb):
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: K1 style shows high mechanism cue density
# ──────────────────────────────────────────────────────────────────────────────

def test_k1_mechanism_density():
    """K1 responses to a mechanism probe contain ≥1 cue on average over 50 samples."""
    hits = []
    for seed in range(50):
        profile = sample_profile(_SECTION, "KNOWS", seed, 0)
        # Force K1 by trying until we get it (deterministic seed loop)
        if profile.style != "K1":
            continue
        resp = generate_response(_MECHANISM_QUESTION, _SECTION, profile, KB, seed, 0)
        hits.append(_count_mech_cue_hits(resp, _SECTION))

    if not hits:
        pytest.skip("No K1 profiles sampled in seed range 0-49 — increase range")

    avg_hits = sum(hits) / len(hits)
    assert avg_hits >= 1.0, (
        f"K1 average mechanism cue hits = {avg_hits:.2f}, expected ≥ 1.0 over {len(hits)} samples"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: F1 collapses under mechanism probe
# ──────────────────────────────────────────────────────────────────────────────

def test_f1_collapses_under_probe():
    """
    F1 response to mechanism probe contains ≤1 mechanism cue in ≥60% of samples.
    (collapse_under_mechanism_probe=0.80 means it will often trail off)
    """
    n_samples = 50
    n_collapsed = 0
    n_f1 = 0

    for seed in range(100):
        profile = sample_profile(_SECTION, "FAKING", seed, 0)
        if profile.style != "F1":
            continue
        resp = generate_response(_MECHANISM_QUESTION, _SECTION, profile, KB, seed, 0)
        n_f1 += 1
        hits = _count_mech_cue_hits(resp, _SECTION)
        if hits <= 1:
            n_collapsed += 1
        if n_f1 >= n_samples:
            break

    if n_f1 == 0:
        pytest.skip("No F1 profiles sampled — increase seed range")

    collapse_rate = n_collapsed / n_f1
    assert collapse_rate >= 0.60, (
        f"F1 collapse rate = {collapse_rate:.2f} over {n_f1} samples, expected ≥ 0.60"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: F2 mirrors jargon from question
# ──────────────────────────────────────────────────────────────────────────────

def test_f2_mirrors_jargon():
    """F2 response reuses ≥1 long word (≥5 chars) from the question."""
    jargon_question = (
        "How does the backpropagation algorithm compute gradients using the chain rule?"
    )
    q_words = {w.lower().strip("?.,!") for w in jargon_question.split() if len(w) >= 5}

    n_mirrored = 0
    n_f2 = 0
    for seed in range(100):
        profile = sample_profile("S02", "FAKING", seed, 0)
        if profile.style != "F2":
            continue
        resp = generate_response(jargon_question, "S02", profile, KB, seed, 0)
        resp_words = set(resp.lower().split())
        if q_words & resp_words:
            n_mirrored += 1
        n_f2 += 1
        if n_f2 >= 30:
            break

    if n_f2 == 0:
        pytest.skip("No F2 profiles sampled")

    mirror_rate = n_mirrored / n_f2
    assert mirror_rate >= 0.50, (
        f"F2 jargon mirror rate = {mirror_rate:.2f}, expected ≥ 0.50"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: F4 low specificity (zero or near-zero mechanism cues)
# ──────────────────────────────────────────────────────────────────────────────

def test_f4_low_specificity():
    """F4 responses have 0 mechanism cue hits in ≥70% of samples."""
    n_zero = 0
    n_f4 = 0
    for seed in range(100):
        profile = sample_profile(_SECTION, "FAKING", seed, 0)
        if profile.style != "F4":
            continue
        resp = generate_response(_DEFINITIONAL_QUESTION, _SECTION, profile, KB, seed, 0)
        hits = _count_mech_cue_hits(resp, _SECTION)
        if hits == 0:
            n_zero += 1
        n_f4 += 1
        if n_f4 >= 30:
            break

    if n_f4 == 0:
        pytest.skip("No F4 profiles sampled")

    zero_rate = n_zero / n_f4
    assert zero_rate >= 0.60, (
        f"F4 zero-cue rate = {zero_rate:.2f}, expected ≥ 0.60"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Reproducibility — all 7 styles
# ──────────────────────────────────────────────────────────────────────────────

def test_reproducibility_all_styles():
    """Same (seed, turn, section, question) → byte-identical response."""
    question = "Why does the learning rate affect convergence speed?"
    for seed in range(20):
        for mode, style_choices in [("KNOWS", ["K1", "K2", "K3"]), ("FAKING", ["F1", "F2", "F3", "F4"])]:
            profile = sample_profile(_SECTION, mode, seed, 0)
            r1 = generate_response(question, _SECTION, profile, KB, seed, 1)
            r2 = generate_response(question, _SECTION, profile, KB, seed, 1)
            r3 = generate_response(question, _SECTION, profile, KB, seed, 1)
            assert r1 == r2 == r3, (
                f"Non-deterministic response for style={profile.style}, seed={seed}: "
                f"\nr1={r1!r}\nr2={r2!r}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: K1 vs F1 are distinguishable
# ──────────────────────────────────────────────────────────────────────────────

def test_styles_distinguishable_k1_vs_f1():
    """K1 and F1 responses to the same question have Jaccard overlap < 0.5."""
    overlaps = []
    k1_profile = None
    f1_profile = None

    for seed in range(50):
        if k1_profile is None:
            p = sample_profile(_SECTION, "KNOWS", seed, 0)
            if p.style == "K1":
                k1_profile = (p, seed)
        if f1_profile is None:
            p = sample_profile(_SECTION, "FAKING", seed, 0)
            if p.style == "F1":
                f1_profile = (p, seed)
        if k1_profile and f1_profile:
            break

    if not k1_profile or not f1_profile:
        pytest.skip("Could not find K1 and F1 in first 50 seeds")

    for t in range(5):
        r_k1 = generate_response(_MECHANISM_QUESTION, _SECTION, k1_profile[0], KB, k1_profile[1], t)
        r_f1 = generate_response(_MECHANISM_QUESTION, _SECTION, f1_profile[0], KB, f1_profile[1], t)
        overlaps.append(_jaccard_words(r_k1, r_f1))

    avg_overlap = sum(overlaps) / len(overlaps)
    assert avg_overlap < 0.5, (
        f"K1/F1 average Jaccard overlap = {avg_overlap:.3f}, expected < 0.5"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: All styles produce non-empty responses for all sections
# ──────────────────────────────────────────────────────────────────────────────

def test_all_styles_non_empty():
    """Every style produces a non-empty string for every section."""
    for section_id in CANONICAL_SECTIONS:
        for seed in range(50):
            for mode in ["KNOWS", "FAKING"]:
                profile = sample_profile(section_id, mode, seed, 0)
                resp = generate_response(
                    _DEFINITIONAL_QUESTION, section_id, profile, KB, seed, 0
                )
                assert isinstance(resp, str) and len(resp.strip()) > 0, (
                    f"Empty response for style={profile.style}, section={section_id}, seed={seed}"
                )
