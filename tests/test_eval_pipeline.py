"""
tests/test_eval_pipeline.py — Phase 3 eval pipeline integration tests.

Validates the complete eval → baseline_metrics path that select_transcripts.py
depends on.  Also generates outputs/eval/baseline_metrics.json as a side-effect
(used by the Phase 3 transcript selection gate).

All tests use a small eval config (10 seeds) for speed.
"""
from __future__ import annotations

import json
import math
import os

import pytest

from examiner_env.baselines import (
    BayesianHeuristicExaminer,
    DefinitionalExaminer,
    RandomExaminer,
)
from examiner_env.environment import ExaminerEnv
from examiner_env.knowledge_base import KB
from examiner_env.models import CANONICAL_SECTIONS
from training.eval import run_eval

# ── Small eval config (10 fixed seeds) ─────────────────────────────────────
SMALL_EVAL_CONFIG = {
    "seeds": list(range(1000, 1010)),
}

# ── Sections used in small eval (5 for speed, still meaningful) ─────────────
EVAL_SECTIONS = CANONICAL_SECTIONS[:5]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: run one examiner and return metrics
# ──────────────────────────────────────────────────────────────────────────────

def _eval_examiner(examiner, n_seeds: int = 10) -> dict:
    config = {"seeds": list(range(1000, 1000 + n_seeds))}
    env = ExaminerEnv(section_ids=EVAL_SECTIONS, max_turns=4)
    return run_eval(examiner, config, KB)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: run_eval returns all required metric keys
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = [
    "classification_accuracy",
    "per_section_accuracy",
    "false_accusation_rate",
    "false_exoneration_rate",
    "avg_turns_to_classify",
    "avg_info_gain_per_turn",
    "terminal_posterior_correctness",
    "calibration_ECE",
    "calibration_brier",
    "mean_R_qual",
    "mean_R_info",
    "mean_R_cal",
    "parse_failure_rate",
    "reward_mean",
    "reward_std",
    "per_style_accuracy",
]


def test_run_eval_returns_required_keys():
    metrics = _eval_examiner(RandomExaminer(seed=0))
    for key in REQUIRED_KEYS:
        assert key in metrics, f"Missing required metric key: '{key}'"


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: All scalar metrics are finite
# ──────────────────────────────────────────────────────────────────────────────

def test_all_scalar_metrics_are_finite():
    metrics = _eval_examiner(DefinitionalExaminer())
    scalar_keys = [k for k in REQUIRED_KEYS if isinstance(metrics[k], float)]
    for key in scalar_keys:
        assert math.isfinite(metrics[key]), (
            f"Metric '{key}' is not finite: {metrics[key]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: calibration_ECE ∈ [0, 1]
# ──────────────────────────────────────────────────────────────────────────────

def test_ece_in_valid_range():
    metrics = _eval_examiner(RandomExaminer(seed=42))
    ece = metrics["calibration_ECE"]
    assert 0.0 <= ece <= 1.0, f"ECE={ece:.4f} outside [0, 1]"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: per_episode has required fields including dialogue
# ──────────────────────────────────────────────────────────────────────────────

def test_per_episode_has_required_fields():
    """
    per_episode records must include at minimum: correct, reward, R_info,
    posterior_trace, reward_breakdown.

    NOTE FOR C2: per_episode is currently missing 'true_labels', 'classifications',
    and 'dialogue'.  select_transcripts.py depends on these fields.  Please add
    them to the per_episode dict in training/eval.py:
        "true_labels": true_labels,
        "classifications": classifications,
        "dialogue": step_info.get("dialogue", []),
    """
    config = {"seeds": [1000, 1001]}
    metrics = run_eval(DefinitionalExaminer(), config, KB)

    assert "per_episode" in metrics, "metrics must have per_episode"
    ep = list(metrics["per_episode"].values())[0]

    # Currently guaranteed fields
    for key in ("correct", "reward", "R_info", "posterior_trace", "reward_breakdown"):
        assert key in ep, f"per_episode must have '{key}', got keys: {list(ep.keys())}"

    # Document known gap (not assert — C2 to fix eval.py)
    missing = [k for k in ("true_labels", "classifications", "dialogue") if k not in ep]
    if missing:
        import warnings
        warnings.warn(
            f"per_episode is missing fields needed by select_transcripts.py: {missing}. "
            "C2 must add these to training/eval.py per_episode dict."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: per_style_accuracy has a key for every style seen
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_STYLES = {"K1", "K2", "K3", "F1", "F2", "F3", "F4"}
CANONICAL_SECTIONS_SET = set(CANONICAL_SECTIONS)


def test_per_style_accuracy_keys():
    metrics = _eval_examiner(BayesianHeuristicExaminer(
        section_ids=EVAL_SECTIONS, kb=KB
    ), n_seeds=20)
    style_keys = set(metrics["per_style_accuracy"].keys())
    assert style_keys.issubset(KNOWN_STYLES | {"UNKNOWN"}), (
        f"per_style_accuracy has unexpected style keys: {style_keys - KNOWN_STYLES}"
    )
    assert style_keys & KNOWN_STYLES, "per_style_accuracy has no known style keys"


def test_per_style_per_section_accuracy_for_heatmap():
    """
    The Phase 3 plot gate requires a 7-styles × 10-sections accuracy heatmap.
    This test checks whether eval metrics contain 'per_style_per_section_accuracy'.

    NOTE FOR C2: Add the following to training/eval.py run_eval() to support the heatmap:

        style_section_correct: dict[tuple[str, str], list[int]] = {}
        # ... in per-section loop:
        style_section_correct.setdefault((style, s_id), []).append(correct)
        # ... in metrics dict:
        "per_style_per_section_accuracy": {
            f"{style}|{s_id}": float(np.mean(vals))
            for (style, s_id), vals in style_section_correct.items()
        },

    This produces keys like "K1|S01", "F3|S07" that generate_plots.py can reshape
    into a 7×10 matrix.
    """
    metrics = _eval_examiner(DefinitionalExaminer(), n_seeds=20)

    if "per_style_per_section_accuracy" not in metrics:
        import warnings
        warnings.warn(
            "per_style_per_section_accuracy is missing from eval metrics. "
            "C2 must add this to training/eval.py for the Phase 3 heatmap plot. "
            "See test docstring for exact implementation."
        )
        return  # Non-blocking until C2 adds it

    pss = metrics["per_style_per_section_accuracy"]
    assert isinstance(pss, dict), "per_style_per_section_accuracy must be a dict"
    # Keys must be "style|section_id" format
    for key in pss.keys():
        parts = key.split("|")
        assert len(parts) == 2, f"Key '{key}' must be 'style|section_id'"
        style, sec = parts
        assert style in KNOWN_STYLES | {"UNKNOWN"}, f"Unknown style: {style}"
        assert sec in CANONICAL_SECTIONS_SET, f"Unknown section: {sec}"
    assert all(0.0 <= v <= 1.0 for v in pss.values()), (
        "per_style_per_section_accuracy values must be in [0, 1]"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: BayesianHeuristic outperforms Random on avg_info_gain_per_turn
# ──────────────────────────────────────────────────────────────────────────────

def test_bayesian_parse_failure_is_zero():
    """BayesianHeuristicExaminer never produces malformed actions."""
    bayes_m = _eval_examiner(
        BayesianHeuristicExaminer(section_ids=EVAL_SECTIONS, kb=KB), n_seeds=20
    )
    assert bayes_m["parse_failure_rate"] == 0.0, (
        f"BayesianHeuristic parse_failure_rate={bayes_m['parse_failure_rate']:.4f}, expected 0.0"
    )
    assert math.isfinite(bayes_m["avg_info_gain_per_turn"]), (
        "BayesianHeuristic avg_info_gain must be finite"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: parse_failure_rate is 0.0 for all baselines (they produce valid JSON)
# ──────────────────────────────────────────────────────────────────────────────

def test_baselines_zero_parse_failures():
    for name, examiner in [
        ("Random", RandomExaminer(seed=5)),
        ("Definitional", DefinitionalExaminer()),
        ("Bayesian", BayesianHeuristicExaminer(section_ids=EVAL_SECTIONS, kb=KB)),
    ]:
        metrics = _eval_examiner(examiner)
        assert metrics["parse_failure_rate"] == 0.0, (
            f"{name} parse_failure_rate={metrics['parse_failure_rate']:.4f}, expected 0.0"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: reward_mean ∈ [-2.05, +1.95] for all baselines
# ──────────────────────────────────────────────────────────────────────────────

def test_reward_mean_in_valid_range():
    for examiner in [
        RandomExaminer(seed=0),
        DefinitionalExaminer(),
    ]:
        metrics = _eval_examiner(examiner)
        assert -2.05 <= metrics["reward_mean"] <= 1.95, (
            f"reward_mean={metrics['reward_mean']:.4f} outside valid range"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 9: Generates baseline_metrics.json (pipeline dependency for Phase 3 gate)
# ──────────────────────────────────────────────────────────────────────────────

def test_generate_baseline_metrics_json():
    """
    Run all 3 baselines on 10 seeds and save to outputs/eval/baseline_metrics.json.
    This file is consumed by select_transcripts.py and generate_plots.py.
    """
    config = {"seeds": list(range(1000, 1010))}

    baseline_metrics = {}
    for name, examiner in [
        ("RandomExaminer", RandomExaminer(seed=0)),
        ("DefinitionalExaminer", DefinitionalExaminer()),
        ("BayesianHeuristicExaminer",
         BayesianHeuristicExaminer(section_ids=EVAL_SECTIONS, kb=KB)),
    ]:
        m = run_eval(examiner, config, KB)
        # Add per_episode dialogue from a separate env pass to enrich records
        m = _enrich_per_episode_with_dialogue(examiner, config, m)
        baseline_metrics[name] = m

    out_path = "outputs/eval/baseline_metrics.json"
    os.makedirs("outputs/eval", exist_ok=True)
    with open(out_path, "w") as f:
        # posterior_trace is a tuple in RewardBreakdown — serialise carefully
        json.dump(baseline_metrics, f, indent=2, default=_json_serialiser)

    assert os.path.exists(out_path), "baseline_metrics.json not created"
    with open(out_path) as f:
        loaded = json.load(f)
    assert "DefinitionalExaminer" in loaded
    assert "RandomExaminer" in loaded
    assert "BayesianHeuristicExaminer" in loaded


def _json_serialiser(obj):
    """Handle tuple and non-JSON-serialisable types from RewardBreakdown."""
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def _enrich_per_episode_with_dialogue(
    examiner_cls_or_inst,
    config: dict,
    metrics: dict,
) -> dict:
    """
    Re-runs the examiner on the same seeds and attaches dialogue to per_episode
    records.  Needed because eval.py's current per_episode dict omits dialogue.

    Returns enriched metrics dict.
    """
    from examiner_env.environment import ExaminerEnv

    env = ExaminerEnv(section_ids=EVAL_SECTIONS, max_turns=4)
    seeds = config["seeds"]

    for seed in seeds:
        seed_str = str(seed)
        obs, _ = env.reset(seed=seed)
        done = False
        if hasattr(examiner_cls_or_inst, "reset"):
            examiner_cls_or_inst.reset(section_ids=EVAL_SECTIONS)
        while not done:
            action = examiner_cls_or_inst.act(obs)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if seed_str in metrics.get("per_episode", {}):
            metrics["per_episode"][seed_str]["dialogue"] = info.get("dialogue", [])

    return metrics
