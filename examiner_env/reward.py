"""
reward.py — Full reward computation for The Examiner (all 11 components).

HARD RULES (from guardrails.md):
  - DO NOT normalise R_total — the trainer owns normalisation.
  - DO NOT call any LLM.
  - DO NOT import from training/.
  - LLR clip is enforced in posterior_oracle.py — do not re-clip here.
  - Decomposition ASSERT tolerance: 1e-9.
  - Bounds ASSERT: -2.05 ≤ R_total ≤ 1.95.

Reward components and exact weights (architecture.md §6.2):

  R_acc           in [-1.00, +1.00]
  R_asym          in [-0.50,  0.00]
  R_cal           in [-0.40, +0.40]
  R_eff           in [ 0.00, +0.20]
  R_cov           in [-0.35,  0.00]
  R_info          in [ 0.00, +0.40]
  R_qual          in [ 0.00, +0.10]
  R_div           in [ 0.00, +0.05]
  P_malformed     ≤ 0
  P_repetition    ≤ 0
  P_invalid_sec   ≤ 0

  R_total = sum of all above   bounded in [-2.05, +1.95]
"""
from __future__ import annotations

from examiner_env.models import EpisodeResult, RewardBreakdown
from examiner_env.question_features import compute_r_qual

# ──────────────────────────────────────────────────────────────────────────────
# Asymmetric error cost constants
# ──────────────────────────────────────────────────────────────────────────────
LAMBDA_FA: float = 0.5   # KNOWS classified as FAKING  (false accusation)
LAMBDA_FE: float = 0.3   # FAKING classified as KNOWS  (false exoneration)


# ──────────────────────────────────────────────────────────────────────────────
# Individual component calculators
# ──────────────────────────────────────────────────────────────────────────────

def _r_acc(classifications: dict, true_labels: dict) -> float:
    n = len(true_labels)
    if n == 0:
        return 0.0
    return sum(
        1.0 if classifications.get(s) == true_labels[s] else -1.0
        for s in true_labels
    ) / n


def _r_asym(classifications: dict, true_labels: dict) -> float:
    n = len(true_labels)
    if n == 0:
        return 0.0
    n_fa = sum(
        1 for s in true_labels
        if true_labels[s] == "KNOWS" and classifications.get(s) == "FAKING"
    )
    n_fe = sum(
        1 for s in true_labels
        if true_labels[s] == "FAKING" and classifications.get(s) == "KNOWS"
    )
    return -(LAMBDA_FA * n_fa + LAMBDA_FE * n_fe) / n


def _r_cal(
    classifications: dict,
    true_labels: dict,
    terminal_posteriors: dict[str, float],
) -> float:
    """
    R_cal rewards having correct calibration direction at episode end.

    R_cal = (0.4 / N) * Σ_s sign(correct(s) − 0.5) × |2·p_T(s) − 1|
    where correct(s) = 1.0 if classified[s] == true_labels[s] else 0.0
    """
    n = len(true_labels)
    if n == 0:
        return 0.0

    def _sign(x: float) -> float:
        return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

    total = 0.0
    for s in true_labels:
        correct = 1.0 if classifications.get(s) == true_labels[s] else 0.0
        p_t = terminal_posteriors.get(s, 0.5)
        total += _sign(correct - 0.5) * abs(2.0 * p_t - 1.0)

    return (0.4 / n) * total


def _r_eff(turns_used: int, max_turns: int, r_acc: float) -> float:
    """R_eff = 0.20 × max(0, (MAX_TURNS - turns_used) / MAX_TURNS) × 1[R_acc > 0]."""
    if r_acc <= 0 or max_turns == 0:
        return 0.0
    return 0.20 * max(0.0, (max_turns - turns_used) / max_turns)


def _r_cov(dialogue_history: list[dict], section_ids: list[str]) -> float:
    """
    R_cov = -0.30 × (1 if any_section_missing else 0) − 0.05 × (n_missing / 10)
    """
    asked_sections = {turn["section_id"] for turn in dialogue_history}
    missing = [s for s in section_ids if s not in asked_sections]
    n_missing = len(missing)
    if n_missing == 0:
        return 0.0
    return -0.30 - 0.05 * (n_missing / 10.0)


def _r_info(info_gains: list[float]) -> float:
    """R_info = 0.40 × clip(Σ_t ΔH_t, 0, 1)."""
    total_gain = sum(info_gains)
    return 0.40 * max(0.0, min(1.0, total_gain))


def _r_qual(dialogue_history: list[dict], kb: dict) -> float:
    """R_qual = 0.10 × mean_over_asks(question_features(q, kb, section_id))."""
    ask_turns = [
        turn for turn in dialogue_history
        if turn.get("action_type") != "classify"
    ]
    if not ask_turns:
        return 0.0
    scores = [
        compute_r_qual(turn["question"], turn["section_id"], kb)
        for turn in ask_turns
    ]
    return 0.10 * (sum(scores) / len(scores))


def _r_div(dialogue_history: list[dict], turns_used: int) -> float:
    """R_div = 0.05 × (n_unique_sections_asked / min(turns_used, 10))."""
    if turns_used == 0:
        return 0.0
    unique = len({turn["section_id"] for turn in dialogue_history})
    return 0.05 * (unique / min(turns_used, 10))


def _p_malformed(n_malformed: int) -> float:
    return -0.20 * n_malformed


def _p_repetition(n_repetition: int) -> float:
    return -0.10 * n_repetition


def _p_invalid_sec(n_invalid_sec: int) -> float:
    return -0.10 * n_invalid_sec


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def compute_reward(episode: EpisodeResult, kb: dict) -> RewardBreakdown:
    """
    Compute all 11 reward components and return a frozen RewardBreakdown.

    Raises ValueError if R_total falls outside [-2.05, +1.95] or the
    component decomposition does not sum to R_total within 1e-9.
    """
    tracker = episode.posterior_tracker
    terminal_posteriors = tracker.current_posteriors()
    entropy_gains: list[float] = tracker.entropy_gains()

    r_acc = _r_acc(episode.classifications, episode.true_labels)
    r_asym = _r_asym(episode.classifications, episode.true_labels)
    r_cal = _r_cal(episode.classifications, episode.true_labels, terminal_posteriors)
    r_eff = _r_eff(episode.turns_used, episode.max_turns, r_acc)
    r_cov = _r_cov(episode.dialogue_history, episode.section_ids)
    r_info = _r_info(entropy_gains)
    r_qual = _r_qual(episode.dialogue_history, kb)
    r_div = _r_div(episode.dialogue_history, episode.turns_used)
    p_mal = _p_malformed(episode.n_malformed)
    p_rep = _p_repetition(episode.n_repetition)
    p_inv = _p_invalid_sec(episode.n_invalid_sec)

    components = [r_acc, r_asym, r_cal, r_eff, r_cov, r_info, r_qual, r_div,
                  p_mal, p_rep, p_inv]
    r_total = sum(components)

    # Decomposition sanity check
    recomputed = sum(components)
    if abs(recomputed - r_total) > 1e-9:
        raise AssertionError(
            f"Reward decomposition error: components sum to {recomputed:.12f} "
            f"but R_total is {r_total:.12f}, diff={abs(recomputed-r_total):.2e}"
        )

    # Bounds check (strict — from spec)
    if not (-2.05 <= r_total <= 1.95):
        raise ValueError(
            f"R_total={r_total:.4f} is outside the valid range [-2.05, +1.95]. "
            f"Check reward component calculations."
        )

    posterior_trace = tuple(tracker.history())

    return RewardBreakdown(
        R_acc=r_acc,
        R_asym=r_asym,
        R_cal=r_cal,
        R_eff=r_eff,
        R_cov=r_cov,
        R_info=r_info,
        R_qual=r_qual,
        R_div=r_div,
        P_malformed=p_mal,
        P_repetition=p_rep,
        P_invalid_sec=p_inv,
        R_total=r_total,
        posterior_trace=posterior_trace,
        info_gain_per_turn=tuple(entropy_gains),
    )
