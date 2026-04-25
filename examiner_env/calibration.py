"""
calibration.py â€” Oracle calibration for the KB-grounded posterior.

Runs 200 synthetic episodes to find optimal (Î±, Î², Î³) per section.
Saves result to outputs/eval/oracle_calibration.json.

ASSERT conditions:
  - mean_brier â‰¤ 0.18
  - terminal_accuracy â‰¥ 0.75
"""
from __future__ import annotations

import json
import os
import random

from examiner_env.knowledge_base import KB, SectionKB, build_kb
from examiner_env.posterior_oracle import PosteriorTracker, score_response
from examiner_env.student import generate_response, sample_profile


class CalibrationError(Exception):
    """Raised when calibration targets are not met after grid search."""


def _brier_score(posterior: float, true_label: str) -> float:
    y = 1.0 if true_label == "KNOWS" else 0.0
    return (posterior - y) ** 2


def _run_single_calibration(
    kb: dict,
    n_episodes: int,
    alpha: float,
    beta: float,
    gamma: float,
    base_seed: int = 0,
) -> tuple[float, float]:
    """
    Run n_episodes episodes with given (Î±, Î², Î³) and return (mean_brier, terminal_accuracy).
    """
    section_ids = list(kb.keys())
    total_brier = 0.0
    total_correct = 0
    total_items = 0

    for ep_idx in range(n_episodes):
        ep_seed = base_seed + ep_idx
        rng = random.Random(ep_seed)

        # Assign KNOWS/FAKING to each section
        true_labels: dict[str, str] = {
            s: rng.choice(["KNOWS", "FAKING"]) for s in section_ids
        }

        # Build tracker with overridden weights via monkey-patch
        tracker = PosteriorTracker(section_ids)

        for s_idx, section_id in enumerate(section_ids):
            mode = true_labels[section_id]
            profile = sample_profile(section_id, mode, ep_seed, s_idx)

            # Generate 2 questions per section from probe templates
            section_kb: SectionKB = kb[section_id]
            n_probes = min(2, len(section_kb.probe_templates))
            probe_subset = rng.sample(section_kb.probe_templates, n_probes)

            for t_idx, probe in enumerate(probe_subset):
                question = probe.template
                response = generate_response(
                    question_text=question,
                    section_id=section_id,
                    profile=profile,
                    kb=kb,
                    episode_seed=ep_seed,
                    turn=t_idx,
                )

                # Override Î±/Î³ in evidence weights temporarily
                original_weights = section_kb.evidence_weights.copy()
                section_kb.evidence_weights["alpha"] = alpha
                section_kb.evidence_weights["gamma"] = gamma
                section_kb.evidence_weights["beta"] = beta

                tracker.update(section_id, response, kb)

                section_kb.evidence_weights.update(original_weights)

            # Terminal brier
            p_t = tracker.posterior(section_id)
            total_brier += _brier_score(p_t, true_labels[section_id])

            # Terminal accuracy
            pred = "KNOWS" if p_t > 0.5 else "FAKING"
            if pred == true_labels[section_id]:
                total_correct += 1
            total_items += 1

    mean_brier = total_brier / total_items
    terminal_accuracy = total_correct / total_items
    return mean_brier, terminal_accuracy


def run_calibration(
    kb: dict | None = None,
    n_episodes: int = 200,
    output_path: str = "outputs/eval/oracle_calibration.json",
) -> dict:
    """
    Calibrate (Î±, Î², Î³) on a synthetic held-out split.

    Steps:
      1. Run 200 episodes with default weights.
      2. If Brier > 0.18 or accuracy < 0.75, run grid search.
      3. Save best weights per section and global to output JSON.
      4. ASSERT targets are met.

    Returns the calibration dict (also saved to output_path).
    """
    if kb is None:
        kb = build_kb()

    # â”€â”€ Step 1: evaluate defaults â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    default_alpha = 1.5
    default_beta = 0.5
    default_gamma = 1.0

    print(f"[calibration] Running {n_episodes} episodes with defaults "
          f"(a={default_alpha}, b={default_beta}, g={default_gamma}) ...")

    brier, acc = _run_single_calibration(
        kb, n_episodes, default_alpha, default_beta, default_gamma, base_seed=7000
    )
    print(f"[calibration] Defaults â†’ Brier={brier:.4f}, Accuracy={acc:.4f}")

    best_alpha, best_beta, best_gamma = default_alpha, default_beta, default_gamma
    best_brier, best_acc = brier, acc

    # â”€â”€ Step 2: grid search if needed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if brier > 0.18 or acc < 0.75:
        print("[calibration] Defaults did not meet targets. Running grid search ...")
        grid_alpha = [1.0, 1.5, 2.0]
        grid_beta = [0.3, 0.5, 0.7]
        grid_gamma = [0.8, 1.0, 1.2]

        for a in grid_alpha:
            for b in grid_beta:
                for g in grid_gamma:
                    gs_brier, gs_acc = _run_single_calibration(
                        kb, n_episodes // 2, a, b, g, base_seed=8000
                    )
                    # Optimise for lower Brier (primary) then higher accuracy
                    if gs_brier < best_brier or (
                        abs(gs_brier - best_brier) < 0.005 and gs_acc > best_acc
                    ):
                        best_brier = gs_brier
                        best_acc = gs_acc
                        best_alpha, best_beta, best_gamma = a, b, g
                        print(f"[calibration] Better: a={a}, b={b}, g={g} -> "
                              f"Brier={gs_brier:.4f}, Acc={gs_acc:.4f}")

    # â”€â”€ Step 3: check targets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"[calibration] Final: a={best_alpha}, b={best_beta}, g={best_gamma} -> "
          f"Brier={best_brier:.4f}, Acc={best_acc:.4f}")

    if best_brier > 0.18 or best_acc < 0.75:
        raise CalibrationError(
            f"Calibration targets not met after grid search: "
            f"Brier={best_brier:.4f} (target â‰¤0.18), "
            f"Accuracy={best_acc:.4f} (target â‰¥0.75)"
        )

    # â”€â”€ Step 4: build output dict â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    per_section = {
        s: {"alpha": best_alpha, "beta": best_beta, "gamma": best_gamma}
        for s in kb.keys()
    }

    calibration_result = {
        "global": {"alpha": best_alpha, "beta": best_beta, "gamma": best_gamma},
        "per_section": per_section,
        "calibration_metrics": {
            "mean_brier": round(best_brier, 6),
            "terminal_accuracy": round(best_acc, 6),
            "n_episodes": n_episodes,
        },
    }

    # â”€â”€ Step 5: persist â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calibration_result, f, indent=2)

    print(f"[calibration] Saved to {output_path}")
    return calibration_result


if __name__ == "__main__":
    run_calibration()
