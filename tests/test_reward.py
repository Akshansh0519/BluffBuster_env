"""
tests/test_reward.py — Phase 1 reward function tests.

All 10 test cases must pass before training code is written (Gate 1).
"""
from __future__ import annotations

import json
import random

import pytest

from examiner_env.knowledge_base import KB
from examiner_env.models import CANONICAL_SECTIONS, EpisodeResult
from examiner_env.posterior_oracle import PosteriorTracker
from examiner_env.reward import compute_reward

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_tracker_and_history(
    section_ids: list[str],
    knows_sections: list[str],
    n_ask_turns: int,
    seed: int = 0,
) -> tuple[PosteriorTracker, list[dict]]:
    """
    Build a synthetic PosteriorTracker and dialogue history.
    asks n_ask_turns mechanism-probe questions to KNOWS sections.
    """
    tracker = PosteriorTracker(section_ids)
    history = []

    mech_response = (
        "The gradient of the loss with respect to parameters guides each update step. "
        "Learning rate scales the magnitude of each update step. "
        "Momentum accumulates an exponentially decaying average of past gradients."
    )
    misc_response = (
        "Gradient descent always finds the global minimum. "
        "Larger learning rates always train faster."
    )

    rng = random.Random(seed)
    asked_sections = []
    for i in range(n_ask_turns):
        s = rng.choice(section_ids)
        response = mech_response if s in knows_sections else misc_response
        tracker.update(s, response, KB)
        tracker.snapshot()
        history.append({
            "section_id": s,
            "question": "Why does gradient descent converge? Explain the mechanism.",
            "response": response,
            "action_type": "ask",
        })
        asked_sections.append(s)

    return tracker, history


def _make_episode(
    classifications: dict,
    true_labels: dict,
    tracker: PosteriorTracker,
    history: list[dict],
    turns_used: int,
    max_turns: int = 4,
    n_malformed: int = 0,
    n_repetition: int = 0,
    n_invalid_sec: int = 0,
) -> EpisodeResult:
    return EpisodeResult(
        classifications=classifications,
        true_labels=true_labels,
        section_ids=CANONICAL_SECTIONS,
        turns_used=turns_used,
        max_turns=max_turns,
        dialogue_history=history,
        posterior_tracker=tracker,
        n_malformed=n_malformed,
        n_repetition=n_repetition,
        n_invalid_sec=n_invalid_sec,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: All-correct episode → R_total > +0.8
# ──────────────────────────────────────────────────────────────────────────────

def test_all_correct_reward():
    true_labels = {s: "KNOWS" for s in CANONICAL_SECTIONS}
    tracker, history = _make_tracker_and_history(CANONICAL_SECTIONS, CANONICAL_SECTIONS, 4)
    ep = _make_episode(dict(true_labels), true_labels, tracker, history, 4)
    rb = compute_reward(ep, KB)
    assert rb.R_total > 0.8, f"All-correct episode should have R_total > 0.8, got {rb.R_total:.4f}"
    assert rb.R_acc == pytest.approx(1.0, abs=1e-9), f"R_acc should be 1.0, got {rb.R_acc}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: All-wrong episode → R_total < -1.0
# ──────────────────────────────────────────────────────────────────────────────

def test_all_wrong_confident():
    true_labels = {s: "KNOWS" for s in CANONICAL_SECTIONS}
    wrong_labels = {s: "FAKING" for s in CANONICAL_SECTIONS}
    tracker, history = _make_tracker_and_history(CANONICAL_SECTIONS, [], 4)  # misc responses
    ep = _make_episode(wrong_labels, true_labels, tracker, history, 4)
    rb = compute_reward(ep, KB)
    assert rb.R_total < -1.0, (
        f"All-wrong episode should have R_total < -1.0, got {rb.R_total:.4f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Malformed penalty is exactly -0.20
# ──────────────────────────────────────────────────────────────────────────────

def test_malformed_penalty():
    true_labels = {s: "KNOWS" for s in CANONICAL_SECTIONS}
    tracker, history = _make_tracker_and_history(CANONICAL_SECTIONS, CANONICAL_SECTIONS, 4)
    ep = _make_episode(dict(true_labels), true_labels, tracker, history, 4, n_malformed=1)
    rb = compute_reward(ep, KB)
    assert rb.P_malformed == pytest.approx(-0.20, abs=1e-9), (
        f"P_malformed should be -0.20, got {rb.P_malformed}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Decomposition sums to R_total ± 1e-9
# ──────────────────────────────────────────────────────────────────────────────

def test_decomposition_sums():
    true_labels = {s: random.choice(["KNOWS", "FAKING"]) for s in CANONICAL_SECTIONS}
    random.seed(99)
    knows_list = [s for s, v in true_labels.items() if v == "KNOWS"]
    tracker, history = _make_tracker_and_history(CANONICAL_SECTIONS, knows_list, 4, seed=99)
    classifications = {s: random.choice(["KNOWS", "FAKING"]) for s in CANONICAL_SECTIONS}
    ep = _make_episode(classifications, true_labels, tracker, history, 4)
    rb = compute_reward(ep, KB)

    component_sum = (
        rb.R_acc + rb.R_asym + rb.R_cal + rb.R_eff + rb.R_cov +
        rb.R_info + rb.R_qual + rb.R_div +
        rb.P_malformed + rb.P_repetition + rb.P_invalid_sec
    )
    diff = abs(component_sum - rb.R_total)
    assert diff < 1e-9, (
        f"Component sum={component_sum:.12f} ≠ R_total={rb.R_total:.12f}, diff={diff:.2e}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: R_total bounded in [-2.05, +1.95] for 100 random episodes
# ──────────────────────────────────────────────────────────────────────────────

def test_bounds_random_episodes():
    rng = random.Random(12345)
    for ep_idx in range(100):
        section_ids = CANONICAL_SECTIONS[:5]  # use 5 sections for speed
        true_labels = {s: rng.choice(["KNOWS", "FAKING"]) for s in section_ids}
        knows = [s for s, v in true_labels.items() if v == "KNOWS"]
        n_turns = rng.randint(1, 4)
        tracker, history = _make_tracker_and_history(section_ids, knows, n_turns, seed=ep_idx)
        classifications = {s: rng.choice(["KNOWS", "FAKING"]) for s in section_ids}
        n_mal = rng.randint(0, 2)
        ep = EpisodeResult(
            classifications=classifications,
            true_labels=true_labels,
            section_ids=section_ids,
            turns_used=n_turns,
            max_turns=4,
            dialogue_history=history,
            posterior_tracker=tracker,
            n_malformed=n_mal,
            n_repetition=0,
            n_invalid_sec=0,
        )
        rb = compute_reward(ep, KB)
        assert -2.05 <= rb.R_total <= 1.95, (
            f"R_total={rb.R_total:.4f} out of [-2.05, +1.95] at episode {ep_idx}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: R_total is not normalised (returned raw)
# ──────────────────────────────────────────────────────────────────────────────

def test_no_normalization():
    true_labels = {s: "KNOWS" for s in CANONICAL_SECTIONS[:5]}
    tracker, history = _make_tracker_and_history(
        list(true_labels.keys()), list(true_labels.keys()), 4
    )
    ep = EpisodeResult(
        classifications=dict(true_labels),
        true_labels=true_labels,
        section_ids=list(true_labels.keys()),
        turns_used=4,
        max_turns=4,
        dialogue_history=history,
        posterior_tracker=tracker,
        n_malformed=0,
        n_repetition=0,
        n_invalid_sec=0,
    )
    rb = compute_reward(ep, KB)
    # If normalised, mean would be ~0 and std ~1; raw values can be in [-2.05, +1.95]
    # Here we just verify R_total is the raw float (not squeezed to [-1,1])
    assert isinstance(rb.R_total, float), "R_total must be a raw float"
    # R_total for all-correct all-turns should be positive and possibly > 0.5
    assert rb.R_total > 0, f"All-correct episode should have positive R_total, got {rb.R_total}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: R_eff is 0 when R_acc ≤ 0
# ──────────────────────────────────────────────────────────────────────────────

def test_r_eff_gated_by_accuracy():
    true_labels = {s: "KNOWS" for s in CANONICAL_SECTIONS[:5]}
    wrong_classif = {s: "FAKING" for s in CANONICAL_SECTIONS[:5]}
    tracker, history = _make_tracker_and_history(
        list(true_labels.keys()), [], 2, seed=77
    )
    ep = EpisodeResult(
        classifications=wrong_classif,
        true_labels=true_labels,
        section_ids=list(true_labels.keys()),
        turns_used=2,   # Used only 2 of 4 turns — efficiency bonus possible
        max_turns=4,
        dialogue_history=history,
        posterior_tracker=tracker,
        n_malformed=0,
        n_repetition=0,
        n_invalid_sec=0,
    )
    rb = compute_reward(ep, KB)
    assert rb.R_eff == 0.0, (
        f"R_eff should be 0.0 when R_acc ≤ 0, got R_eff={rb.R_eff}, R_acc={rb.R_acc}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: R_qual is question-side only — changing response doesn't change R_qual
# ──────────────────────────────────────────────────────────────────────────────

def test_r_qual_question_side():
    """Two episodes with same questions but different responses → same R_qual."""
    section_ids = ["S01", "S02"]
    true_labels = {"S01": "KNOWS", "S02": "FAKING"}
    classifications = {"S01": "KNOWS", "S02": "FAKING"}

    base_question = "Why does gradient descent require a learning rate? Explain the mechanism."

    def _make_ep_with_response(response: str) -> EpisodeResult:
        tracker = PosteriorTracker(section_ids)
        tracker.update("S01", response, KB)
        tracker.snapshot()
        history = [{"section_id": "S01", "question": base_question,
                    "response": response, "action_type": "ask"}]
        return EpisodeResult(
            classifications=classifications,
            true_labels=true_labels,
            section_ids=section_ids,
            turns_used=1,
            max_turns=4,
            dialogue_history=history,
            posterior_tracker=tracker,
            n_malformed=0,
            n_repetition=0,
            n_invalid_sec=0,
        )

    rb1 = compute_reward(_make_ep_with_response("Short answer."), KB)
    rb2 = compute_reward(_make_ep_with_response(
        "The gradient of the loss with respect to parameters is computed and scaled by "
        "the learning rate to determine the step size for each parameter update."
    ), KB)

    assert rb1.R_qual == pytest.approx(rb2.R_qual, abs=1e-6), (
        f"R_qual should not change with response: {rb1.R_qual:.4f} vs {rb2.R_qual:.4f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 9: R_info ∈ [0, +0.40]
# ──────────────────────────────────────────────────────────────────────────────

def test_r_info_nonnegative():
    rng = random.Random(55)
    for _ in range(50):
        section_ids = CANONICAL_SECTIONS[:5]
        true_labels = {s: rng.choice(["KNOWS", "FAKING"]) for s in section_ids}
        knows = [s for s, v in true_labels.items() if v == "KNOWS"]
        tracker, history = _make_tracker_and_history(section_ids, knows, 3, seed=rng.randint(0, 999))
        classifications = dict(true_labels)
        ep = EpisodeResult(
            classifications=classifications,
            true_labels=true_labels,
            section_ids=section_ids,
            turns_used=3,
            max_turns=4,
            dialogue_history=history,
            posterior_tracker=tracker,
            n_malformed=0,
            n_repetition=0,
            n_invalid_sec=0,
        )
        rb = compute_reward(ep, KB)
        assert 0.0 <= rb.R_info <= 0.40 + 1e-9, (
            f"R_info={rb.R_info:.4f} out of [0, 0.40]"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 10: Reproducibility — same EpisodeResult → identical RewardBreakdown
# ──────────────────────────────────────────────────────────────────────────────

def test_reproducibility():
    true_labels = {s: "KNOWS" if i % 2 == 0 else "FAKING"
                   for i, s in enumerate(CANONICAL_SECTIONS)}
    tracker, history = _make_tracker_and_history(
        CANONICAL_SECTIONS,
        [s for s, v in true_labels.items() if v == "KNOWS"],
        4, seed=42
    )
    ep = _make_episode(dict(true_labels), true_labels, tracker, history, 4)
    rb1 = compute_reward(ep, KB)
    rb2 = compute_reward(ep, KB)
    assert rb1.R_total == rb2.R_total, f"Non-deterministic: {rb1.R_total} vs {rb2.R_total}"
    assert rb1.R_acc == rb2.R_acc
    assert rb1.R_info == rb2.R_info
