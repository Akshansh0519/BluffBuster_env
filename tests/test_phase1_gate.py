"""
tests/test_phase1_gate.py — Phase 1 merge gate checklist.

Covers all 10 gate conditions from implementation_plan.md §PHASE 1 MERGE GATE.
Must pass before Phase 2 training work begins.
"""
from __future__ import annotations

import json
import random
import statistics

import pytest

from examiner_env.baselines import (
    BayesianHeuristicExaminer,
    DefinitionalExaminer,
    RandomExaminer,
)
from examiner_env.environment import ExaminerEnv
from examiner_env.knowledge_base import KB
from examiner_env.models import CANONICAL_SECTIONS

# ──────────────────────────────────────────────────────────────────────────────
# Gate 1: Single end-to-end episode — reset → ask × 3 → classify
# ──────────────────────────────────────────────────────────────────────────────

def test_gate1_single_episode_end_to_end():
    """
    env.reset(42) → 3 valid Ask actions → 1 Classify → R_total finite in [-2.05, +1.95]
    """
    env = ExaminerEnv(section_ids=CANONICAL_SECTIONS, max_turns=4)
    obs, info = env.reset(seed=42)

    assert "section_titles" in obs
    assert "dialogue_history" in obs
    assert "true_labels" not in obs, "true_labels must NOT appear in observation"
    assert "posteriors" not in obs, "posteriors must NOT appear in observation"
    assert obs["dialogue_history"] == []

    # 3 Ask steps
    section_ids = obs["section_ids"]
    for t, section_id in enumerate(section_ids[:3]):
        action = json.dumps({
            "action_type": "ask",
            "section_id": section_id,
            "question_text": f"Why is {section_id} important? Explain the mechanism in detail.",
        })
        obs, reward, terminated, truncated, step_info = env.step(action)
        assert reward == 0.0, f"Ask step should return reward=0.0, got {reward}"
        assert not terminated, "Episode should not terminate on Ask step"
        assert len(obs["dialogue_history"]) == t + 1

    # Final classify
    all_classify = json.dumps({
        "action_type": "classify",
        "classifications": {s: "KNOWS" for s in CANONICAL_SECTIONS},
    })
    obs, reward, terminated, truncated, step_info = env.step(all_classify)

    assert terminated, "Episode must terminate on Classify"
    assert isinstance(reward, float), "Reward must be float"
    assert -2.05 <= reward <= 1.95, f"R_total={reward:.4f} out of [-2.05, +1.95]"


# ──────────────────────────────────────────────────────────────────────────────
# Gate 2: Observation never contains hidden state
# ──────────────────────────────────────────────────────────────────────────────

def test_gate2_observation_has_no_hidden_state():
    """No true_labels, style_ids, or posterior values in observation."""
    env = ExaminerEnv(section_ids=CANONICAL_SECTIONS[:5], max_turns=4)
    obs, _ = env.reset(seed=7)

    forbidden_keys = {"true_labels", "style_assignments", "posteriors",
                      "posterior", "style_ids", "hidden"}
    for key in forbidden_keys:
        assert key not in obs, f"Forbidden key '{key}' found in observation"


# ──────────────────────────────────────────────────────────────────────────────
# Gate 3: All 3 baselines complete 10 episodes without crash
# ──────────────────────────────────────────────────────────────────────────────

def _run_baseline_episodes(examiner, n: int = 10, section_ids=None):
    section_ids = section_ids or CANONICAL_SECTIONS[:5]
    env = ExaminerEnv(section_ids=section_ids, max_turns=4)
    rewards = []
    for seed in range(n):
        obs, _ = env.reset(seed=seed)
        done = False
        if hasattr(examiner, "reset"):
            examiner.reset(section_ids=section_ids)
        while not done:
            action = examiner.act(obs)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        rewards.append(r)
    return rewards


def test_gate3_random_examiner_10_episodes():
    examiner = RandomExaminer(seed=0)
    rewards = _run_baseline_episodes(examiner, n=10)
    assert len(rewards) == 10
    assert all(isinstance(r, float) for r in rewards)
    assert all(-2.05 <= r <= 1.95 for r in rewards)


def test_gate3_definitional_examiner_10_episodes():
    examiner = DefinitionalExaminer()
    rewards = _run_baseline_episodes(examiner, n=10)
    assert len(rewards) == 10
    assert all(-2.05 <= r <= 1.95 for r in rewards)


def test_gate3_bayesian_examiner_10_episodes():
    section_ids = CANONICAL_SECTIONS[:5]
    examiner = BayesianHeuristicExaminer(section_ids=section_ids, kb=KB)
    rewards = _run_baseline_episodes(examiner, n=10, section_ids=section_ids)
    assert len(rewards) == 10
    assert all(-2.05 <= r <= 1.95 for r in rewards)


# ──────────────────────────────────────────────────────────────────────────────
# Gate 4: Reward decomposition sums to R_total (already tested in test_reward.py)
# ──────────────────────────────────────────────────────────────────────────────

def test_gate4_env_reward_decomposition_live():
    """Run a live episode and verify decomposition via info dict."""
    env = ExaminerEnv(section_ids=CANONICAL_SECTIONS[:5], max_turns=4)
    obs, _ = env.reset(seed=123)
    done = False
    examiner = RandomExaminer(seed=99)
    while not done:
        action = examiner.act(obs)
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    bd = info["reward_breakdown"]
    component_sum = (
        bd.R_acc + bd.R_asym + bd.R_cal + bd.R_eff + bd.R_cov +
        bd.R_info + bd.R_qual + bd.R_div +
        bd.P_malformed + bd.P_repetition + bd.P_invalid_sec
    )
    diff = abs(component_sum - bd.R_total)
    assert diff < 1e-9, f"Decomposition error: diff={diff:.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# Gate 5: Reward variance ≥ 0.05 over 20 random episodes
# ──────────────────────────────────────────────────────────────────────────────

def test_gate5_reward_variance_nonzero():
    """Reward must not be constant — σ(R_total) ≥ 0.05 over 20 episodes."""
    section_ids = CANONICAL_SECTIONS[:5]
    env = ExaminerEnv(section_ids=section_ids, max_turns=4)
    examiner = RandomExaminer(seed=42)
    rewards = []
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        done = False
        examiner.reset()
        while not done:
            action = examiner.act(obs)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        rewards.append(r)

    std = statistics.stdev(rewards)
    assert std >= 0.05, (
        f"Reward variance too low: σ={std:.4f}, expected ≥ 0.05. "
        "Reward appears too constant — check student simulator diversity."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Gate 6: Info dict exposes true_labels and reward_breakdown
# ──────────────────────────────────────────────────────────────────────────────

def test_gate6_info_exposes_ground_truth():
    """On classify step, info must have true_labels, classifications, reward_breakdown."""
    env = ExaminerEnv(section_ids=CANONICAL_SECTIONS[:3], max_turns=4)
    obs, _ = env.reset(seed=55)
    examiner = DefinitionalExaminer()
    done = False
    while not done:
        action = examiner.act(obs)
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    assert "true_labels" in info, "info must contain true_labels at episode end"
    assert "classifications" in info, "info must contain classifications at episode end"
    assert "style_assignments" in info, "info must contain style_assignments at episode end"
    assert "reward_breakdown" in info, "info must contain reward_breakdown at episode end"
    assert set(info["true_labels"].keys()) >= set(CANONICAL_SECTIONS[:3])


# ──────────────────────────────────────────────────────────────────────────────
# Gate 7: Baselines produce valid JSON actions (parse-able by action_parser)
# ──────────────────────────────────────────────────────────────────────────────

def test_gate7_baselines_produce_valid_json():
    """All 3 baselines produce parse-able JSON actions throughout an episode."""
    from examiner_env.action_parser import parse
    from examiner_env.models import MalformedAction

    # Must use all 10 sections: ClassifyAction now requires exactly S01–S10.
    section_ids = CANONICAL_SECTIONS
    env = ExaminerEnv(section_ids=section_ids, max_turns=10)
    examiners = [
        RandomExaminer(seed=7),
        DefinitionalExaminer(),
        BayesianHeuristicExaminer(section_ids=section_ids, kb=KB),
    ]

    for examiner in examiners:
        obs, _ = env.reset(seed=0)
        done = False
        if hasattr(examiner, "reset"):
            examiner.reset(section_ids=section_ids)
        steps = 0
        while not done:
            action_text = examiner.act(obs)
            assert isinstance(action_text, str), (
                f"{type(examiner).__name__}.act() must return str, got {type(action_text)}"
            )
            parsed = parse(action_text)
            assert not isinstance(parsed, MalformedAction), (
                f"{type(examiner).__name__} produced MalformedAction: {action_text!r}"
            )
            obs, r, terminated, truncated, info = env.step(action_text)
            done = terminated or truncated
            steps += 1
        assert steps >= 1, f"{type(examiner).__name__} episode had 0 steps"
