"""
training/eval.py — Frozen eval suite runner.
C2 owns. Implements run_eval() which evaluates any examiner on the frozen eval suite
and returns all 15 required metrics including ECE, Brier, per-style accuracy.

DEPENDS ON: examiner_env (C1) — do not implement before C1 Phase 1 gate clears.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# ECE helper — 10-bin standard implementation
# ---------------------------------------------------------------------------

def compute_ece(posteriors: list[float], labels: list[int], n_bins: int = 10) -> float:
    """
    Expected Calibration Error.
    posteriors: list of p_T(s) values (one per section per episode).
    labels:     1 if KNOWS (positive class), 0 if FAKING.
    Returns ECE in [0, 1].
    """
    if not posteriors:
        return float("nan")
    posteriors_arr = np.array(posteriors)
    labels_arr = np.array(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (posteriors_arr >= bins[i]) & (posteriors_arr < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidence = np.mean(posteriors_arr[mask])
        bin_accuracy = np.mean(labels_arr[mask])
        ece += mask.sum() * abs(bin_confidence - bin_accuracy)
    return float(ece / len(posteriors))


def compute_brier(posteriors: list[float], labels: list[int]) -> float:
    """Brier score: mean squared error between posteriors and true labels."""
    if not posteriors:
        return float("nan")
    return float(np.mean((np.array(posteriors) - np.array(labels)) ** 2))


# ---------------------------------------------------------------------------
# Main eval runner — STUB until C1 Phase 1 gate clears
# ---------------------------------------------------------------------------

def run_eval(
    examiner: Any,
    eval_config: dict,
    kb: Any,
    output_path: str | None = None,
) -> dict:
    """
    Run examiner on frozen eval suite. Returns full metrics dict.

    Required metrics (ALL computed):
    - classification_accuracy: float
    - per_section_accuracy: dict[str, float]
    - false_accusation_rate: float  (KNOWS→FAKING)
    - false_exoneration_rate: float  (FAKING→KNOWS)
    - avg_turns_to_classify: float
    - avg_info_gain_per_turn: float
    - terminal_posterior_correctness: float
    - calibration_ECE: float  (10-bin)
    - calibration_brier: float
    - mean_R_qual: float
    - mean_R_info: float
    - mean_R_cal: float
    - parse_failure_rate: float
    - reward_mean: float
    - reward_std: float
    - per_style_accuracy: dict[str, float]

    NOTE: This stub will be replaced once examiner_env (C1) is available.
    """
    # ── Deferred imports from C1 (fail gracefully if not yet built) ──
    try:
        from examiner_env.environment import ExaminerEnv
        from examiner_env.reward import RewardBreakdown
    except ImportError as e:
        raise RuntimeError(
            "examiner_env not yet available — wait for C1 Phase 1 gate to clear "
            "before running eval. Original error: " + str(e)
        )

    seeds = eval_config["seeds"]

    # Accumulation buffers
    all_rewards: list[float] = []
    all_turns: list[int] = []
    all_info_gains: list[float] = []
    section_correct: dict[str, list[int]] = {}
    style_correct: dict[str, list[int]] = {}
    all_posteriors: list[float] = []
    all_labels_binary: list[int] = []
    all_r_qual: list[float] = []
    all_r_info: list[float] = []
    all_r_cal: list[float] = []
    total_steps = 0
    total_malformed = 0
    false_accusations = 0
    false_exonerations = 0
    total_classified = 0
    posterior_correct = 0
    posterior_total = 0
    episode_results: list[dict] = []

    env = ExaminerEnv(kb=kb, config=None)
    n_seeds = len(seeds)

    for ep_idx, seed in enumerate(seeds):
        print(f"[eval] Episode {ep_idx + 1}/{n_seeds} (seed={seed})...", flush=True)
        obs, info = env.reset(seed=seed)
        done = False
        ep_turns = 0

        # Let the examiner act until episode ends
        if hasattr(examiner, "reset"):
            examiner.reset()

        while not done:
            ep_turns += 1
            print(f"[eval]   turn {ep_turns} ...", flush=True)
            action_text = examiner.act(obs)
            obs, reward, terminated, truncated, step_info = env.step(action_text)
            done = terminated or truncated
        print(f"[eval]   done in {ep_turns} turns | reward={reward:.3f}", flush=True)

        # Collect RewardBreakdown from final step_info
        bd: RewardBreakdown | None = step_info.get("reward_breakdown")
        true_labels: dict[str, str] = step_info.get("true_labels", {})
        classifications: dict[str, str] = step_info.get("classifications", {})
        style_assignments: dict[str, str] = step_info.get("style_assignments", {})

        if bd is None:
            continue

        all_rewards.append(bd.R_total)
        all_turns.append(ep_turns)
        # Filter NaN from info_gain_per_turn — can occur when posterior is degenerate
        all_info_gains.extend(g for g in bd.info_gain_per_turn if np.isfinite(g))
        all_r_qual.append(bd.R_qual if np.isfinite(bd.R_qual) else 0.0)
        all_r_info.append(bd.R_info if np.isfinite(bd.R_info) else 0.0)
        all_r_cal.append(bd.R_cal if np.isfinite(bd.R_cal) else 0.0)
        total_malformed += abs(int(round(bd.P_malformed / 0.20))) if bd.P_malformed < 0 else 0
        total_steps += ep_turns

        # Per-section accuracy / false rates
        for s_id, pred in classifications.items():
            truth = true_labels.get(s_id, "")
            correct = int(pred == truth)
            section_correct.setdefault(s_id, []).append(correct)
            if truth == "KNOWS" and pred == "FAKING":
                false_accusations += 1
            if truth == "FAKING" and pred == "KNOWS":
                false_exonerations += 1
            total_classified += 1

            style = style_assignments.get(s_id, "UNKNOWN")
            style_correct.setdefault(style, []).append(correct)

        # Posterior calibration
        if bd.posterior_trace:
            final_posteriors = bd.posterior_trace[-1]
            for s_id, p in final_posteriors.items():
                truth = true_labels.get(s_id, "")
                label_bin = 1 if truth == "KNOWS" else 0
                all_posteriors.append(p)
                all_labels_binary.append(label_bin)
                # terminal posterior correctness: does sign(p-0.5) match truth?
                pred_knows = p > 0.5
                posterior_correct += int(pred_knows == (truth == "KNOWS"))
                posterior_total += 1

        ep_record = {
            "seed": seed,
            "reward": bd.R_total,
            "turns": ep_turns,
            "classifications": classifications,
            "true_labels": true_labels,
            "correct": classifications == true_labels,
            "R_info": bd.R_info,
            "reward_breakdown": {
                "R_acc": bd.R_acc, "R_asym": bd.R_asym, "R_cal": bd.R_cal,
                "R_eff": bd.R_eff, "R_cov": bd.R_cov, "R_info": bd.R_info,
                "R_qual": bd.R_qual, "R_div": bd.R_div,
                "P_malformed": bd.P_malformed,
            },
            "posterior_trace": bd.posterior_trace,
            "style_assignments": style_assignments,
            # Dialogue history from final observation — required by select_transcripts.py
            "dialogue": obs.get("dialogue_history", []),
        }
        episode_results.append(ep_record)

    # ── Aggregate metrics ──
    n = len(seeds)
    metrics = {
        "classification_accuracy": (
            float(np.mean([r["correct"] for r in episode_results])) if episode_results else float("nan")
        ),
        "per_section_accuracy": {
            s: float(np.mean(vals)) for s, vals in section_correct.items()
        },
        "false_accusation_rate": false_accusations / max(total_classified, 1),
        "false_exoneration_rate": false_exonerations / max(total_classified, 1),
        "avg_turns_to_classify": float(np.mean(all_turns)) if all_turns else 0.0,
        "avg_info_gain_per_turn": float(np.mean(all_info_gains)) if all_info_gains else 0.0,
        "terminal_posterior_correctness": posterior_correct / max(posterior_total, 1),
        "calibration_ECE": compute_ece(all_posteriors, all_labels_binary),
        "calibration_brier": compute_brier(all_posteriors, all_labels_binary),
        "mean_R_qual": float(np.mean(all_r_qual)) if all_r_qual else 0.0,
        "mean_R_info": float(np.mean(all_r_info)) if all_r_info else 0.0,
        "mean_R_cal": float(np.mean(all_r_cal)) if all_r_cal else 0.0,
        "parse_failure_rate": total_malformed / max(total_steps, 1),
        "reward_mean": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "reward_std": float(np.std(all_rewards)) if all_rewards else 0.0,
        "per_style_accuracy": {
            style: float(np.mean(vals)) for style, vals in style_correct.items()
        },
        "per_episode": {
            str(r["seed"]): {
                "correct": r["correct"],
                "R_info": r["R_info"],
                "reward": r["reward"],
                "reward_breakdown": r["reward_breakdown"],
                "posterior_trace": r["posterior_trace"],
                # Required by select_transcripts.py for before/after transcript export
                "classifications": r["classifications"],
                "true_labels": r["true_labels"],
                "dialogue": r["dialogue"],
            }
            for r in episode_results
        },
        "n_episodes": len(episode_results),
    }

    # Sanity: warn on non-finite scalars, clamp to 0.0 rather than crashing.
    # A hard crash here would discard all training progress — a warning is safer.
    scalar_keys = [
        "classification_accuracy", "false_accusation_rate", "false_exoneration_rate",
        "avg_turns_to_classify", "avg_info_gain_per_turn", "terminal_posterior_correctness",
        "calibration_ECE", "calibration_brier", "mean_R_qual", "mean_R_info", "mean_R_cal",
        "parse_failure_rate", "reward_mean", "reward_std",
    ]
    for k in scalar_keys:
        v = metrics[k]
        if not np.isfinite(v):
            import warnings
            warnings.warn(
                f"run_eval: metric '{k}' is not finite ({v}) — clamping to 0.0. "
                "This usually means all episodes had degenerate posteriors or parse failures.",
                RuntimeWarning, stacklevel=2,
            )
            metrics[k] = 0.0

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics
