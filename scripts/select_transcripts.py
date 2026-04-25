"""
scripts/select_transcripts.py — Behavior-based transcript selection.
C2 owns.

Selection logic (from §10 of architecture):
  1. Load frozen eval results for DefinitionalExaminer and TrainedExaminer.
  2. Find episodes where DefinitionalExaminer was WRONG and TrainedExaminer was CORRECT.
  3. Among those, select the episode with the LARGEST R_info gap (Trained - Definitional).
  4. Export: before_transcript.json (Definitional) and after_transcript.json (Trained),
     both on the SAME episode seed.

DO NOT select by episode index. Always by behavioral quality metric.
"""

from __future__ import annotations

import json
import os
import sys


def select_transcripts(
    baseline_path: str = "outputs/eval/baseline_metrics.json",
    final_path: str = "outputs/eval/final_metrics.json",
    eval_config_path: str = "eval_config.json",
    out_dir: str = "outputs/transcripts",
) -> dict:
    """
    Select the best before/after transcript pair based on behavioral quality.
    Returns a dict with the selected episode seed and metadata.
    """
    # ── Load eval results ──
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(final_path) as f:
        final = json.load(f)
    with open(eval_config_path) as f:
        eval_config = json.load(f)

    seeds = [str(s) for s in eval_config["seeds"]]

    def_per_ep = baseline.get("DefinitionalExaminer", {}).get("per_episode", {})
    trained_per_ep = final.get("TrainedExaminer", {}).get("per_episode", {})

    if not def_per_ep:
        raise ValueError(
            "baseline_metrics.json has no per_episode data for DefinitionalExaminer. "
            "Make sure run_eval() populates per_episode records."
        )
    if not trained_per_ep:
        raise ValueError(
            "final_metrics.json has no per_episode data for TrainedExaminer."
        )

    # ── Find correctness-flip episodes ──
    flip_seeds = []
    for seed in seeds:
        def_correct = def_per_ep.get(seed, {}).get("correct", True)
        trained_correct = trained_per_ep.get(seed, {}).get("correct", False)
        if not def_correct and trained_correct:
            flip_seeds.append(seed)

    if not flip_seeds:
        # Fallback: largest R_info gap regardless of correctness flip
        print(
            "WARNING: No correctness-flip episodes found. "
            "Falling back to largest R_info gap (may not show correctness flip)."
        )
        flip_seeds = seeds

    # ── Select episode with largest R_info gap ──
    best_seed = max(
        flip_seeds,
        key=lambda s: (
            trained_per_ep.get(s, {}).get("R_info", 0.0)
            - def_per_ep.get(s, {}).get("R_info", 0.0)
        ),
    )

    def_ep = def_per_ep[best_seed]
    trained_ep = trained_per_ep[best_seed]

    before = {
        "episode_seed": int(best_seed),
        "examiner": "DefinitionalExaminer",
        "dialogue": def_ep.get("dialogue", []),
        "classification": def_ep.get("classifications", {}),
        "true_labels": def_ep.get("true_labels", {}),
        "reward_breakdown": def_ep.get("reward_breakdown", {}),
        "posterior_trace": def_ep.get("posterior_trace", []),
        "R_info": def_ep.get("R_info", 0.0),
        "R_total": def_ep.get("reward", 0.0),
        "correct": def_ep.get("correct", False),
        "_selection_note": "Selected because DefinitionalExaminer was wrong on this seed.",
    }

    after = {
        "episode_seed": int(best_seed),
        "examiner": "TrainedExaminer",
        "dialogue": trained_ep.get("dialogue", []),
        "classification": trained_ep.get("classifications", {}),
        "true_labels": trained_ep.get("true_labels", {}),
        "reward_breakdown": trained_ep.get("reward_breakdown", {}),
        "posterior_trace": trained_ep.get("posterior_trace", []),
        "R_info": trained_ep.get("R_info", 0.0),
        "R_total": trained_ep.get("reward", 0.0),
        "correct": trained_ep.get("correct", True),
        "_selection_note": "Selected because TrainedExaminer was correct on the same seed with higher R_info.",
    }

    # ── Sanity checks ──
    assert before["episode_seed"] == after["episode_seed"], "Seed mismatch!"
    r_info_gap = after["R_info"] - before["R_info"]
    print(
        f"Selected episode seed: {best_seed}\n"
        f"  DefinitionalExaminer: correct={before['correct']}, R_info={before['R_info']:.4f}\n"
        f"  TrainedExaminer:      correct={after['correct']},  R_info={after['R_info']:.4f}\n"
        f"  R_info gap: {r_info_gap:.4f}"
    )

    # ── Save ──
    os.makedirs(out_dir, exist_ok=True)
    before_path = os.path.join(out_dir, "before_transcript.json")
    after_path = os.path.join(out_dir, "after_transcript.json")

    with open(before_path, "w") as f:
        json.dump(before, f, indent=2)
    with open(after_path, "w") as f:
        json.dump(after, f, indent=2)

    print(f"✓ before_transcript.json → {before_path}")
    print(f"✓ after_transcript.json  → {after_path}")

    return {"episode_seed": int(best_seed), "R_info_gap": r_info_gap}


if __name__ == "__main__":
    result = select_transcripts()
    print(f"\nDone. Selected seed={result['episode_seed']}, R_info_gap={result['R_info_gap']:.4f}")
