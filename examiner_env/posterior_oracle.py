"""
posterior_oracle.py — KB-grounded posterior oracle for R_info and R_cal.

Key rules (from guardrails.md):
  - LLR clipped to [-3.0, +3.0] per update — CRITICAL, never remove this clip.
  - Entropy computed in bits (log base 2) for consistent ΔH_t units.
  - This is a deterministic surrogate, NOT ground truth.  Must not be used
    as a classification signal (only for reward shaping and calibration).
  - No LLM calls inside this module.
"""
from __future__ import annotations

import math
from typing import Literal

from examiner_env.knowledge_base import KB, SectionKB

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LLR_CLIP: float = 3.0          # must match architecture.md
LOG_PRIOR_ODDS: float = 0.0    # log(0.5 / 0.5) = 0  (uniform prior per section)
P0: float = 0.5                # prior P(KNOWS) for each section

# ──────────────────────────────────────────────────────────────────────────────
# Scoring: response → log-likelihood ratio
# ──────────────────────────────────────────────────────────────────────────────

def _count_phrase_hits(response: str, phrases: list, threshold: float) -> float:
    """Count weighted phrase hits in response (case-insensitive substring match)."""
    resp_lower = response.lower()
    total = 0.0
    for item in phrases:
        if item.phrase.lower() in resp_lower:
            total += item.weight
    return total


def score_response(
    response: str,
    section_id: str,
    kb: dict | None = None,
) -> float:
    """
    Compute LLR for a single (response, section) pair.

    LLR > 0 → evidence for KNOWS.
    LLR < 0 → evidence for FAKING.

    LLR = alpha * mech_hits - beta * misc_hits
    (evidence_weights from KB, defaults: alpha=1.5, beta=0.5)
    Clipped to [-LLR_CLIP, +LLR_CLIP].
    """
    if kb is None:
        kb = KB

    section_kb: SectionKB = kb[section_id]
    alpha = section_kb.evidence_weights.get("alpha", 1.5)
    beta = section_kb.evidence_weights.get("beta", 0.5)

    mech_hits = _count_phrase_hits(response, section_kb.mechanism_cues, threshold=0)
    misc_hits = _count_phrase_hits(response, section_kb.common_misconceptions, threshold=0)

    raw_llr = alpha * mech_hits - beta * misc_hits

    # CRITICAL: clip LLR
    return max(-LLR_CLIP, min(LLR_CLIP, raw_llr))


# ──────────────────────────────────────────────────────────────────────────────
# Binary entropy helper
# ──────────────────────────────────────────────────────────────────────────────

def _binary_entropy(p: float) -> float:
    """H(p) in bits.  Safe at boundaries."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


# ──────────────────────────────────────────────────────────────────────────────
# Posterior tracker (stateful per episode)
# ──────────────────────────────────────────────────────────────────────────────

class PosteriorTracker:
    """
    Tracks P_t(KNOWS | s) for every section across a single episode.

    Update rule:
      log_odds_t = log_odds_{t-1} + llr_t(s)
      p_t = sigmoid(log_odds_t)
    where llr_t is clipped to [-3, +3].
    """

    def __init__(self, section_ids: list[str]) -> None:
        self._sections = list(section_ids)
        # log-odds initialised to LOG_PRIOR_ODDS = 0 (uniform)
        self._log_odds: dict[str, float] = {s: LOG_PRIOR_ODDS for s in section_ids}
        self._posteriors: dict[str, float] = {s: P0 for s in section_ids}
        self._history: list[dict[str, float]] = []   # posterior snapshot per turn
        self._entropy_gains: list[float] = []        # ΔH_t per update call

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        section_id: str,
        response: str,
        kb: dict | None = None,
    ) -> float:
        """
        Update posterior for one section given a student response.
        Returns ΔH_t (entropy reduction in bits, positive = information gain).
        """
        if section_id not in self._log_odds:
            raise KeyError(f"section_id '{section_id}' not tracked in this episode")

        h_before = _binary_entropy(self._posteriors[section_id])

        llr = score_response(response, section_id, kb)
        self._log_odds[section_id] += llr   # already clipped by score_response
        self._posteriors[section_id] = self._sigmoid(self._log_odds[section_id])

        h_after = _binary_entropy(self._posteriors[section_id])
        delta_h = h_before - h_after   # positive = reduced uncertainty = info gained
        self._entropy_gains.append(delta_h)

        return delta_h

    def snapshot(self) -> None:
        """Record current posteriors.  Call once per turn."""
        self._history.append(dict(self._posteriors))

    def current_posteriors(self) -> dict[str, float]:
        """Return copy of current P(KNOWS|s) for all sections."""
        return dict(self._posteriors)

    def posterior(self, section_id: str) -> float:
        return self._posteriors[section_id]

    def history(self) -> list[dict[str, float]]:
        """Return list of posterior snapshots (one per turn after snapshot() call)."""
        return list(self._history)

    def entropy_gains(self) -> list[float]:
        """Return ΔH_t per update() call (all turns, not just snapshots)."""
        return list(self._entropy_gains)

    def total_entropy(self) -> float:
        """Current total entropy in bits across all sections."""
        return sum(_binary_entropy(p) for p in self._posteriors.values())

    def initial_entropy(self) -> float:
        """Total entropy at prior (all 0.5) in bits."""
        return len(self._sections) * _binary_entropy(P0)

    def total_info_gain(self) -> float:
        """Total entropy reduction from prior to current state, in bits."""
        return self.initial_entropy() - self.total_entropy()

    def calibration_error(self, true_labels: dict[str, Literal["KNOWS", "FAKING"]]) -> float:
        """
        Expected Calibration Error (ECE) — mean absolute difference between
        posterior P(KNOWS|s) and the indicator 1[true_label==KNOWS].
        """
        errors: list[float] = []
        for s, label in true_labels.items():
            p_knows = self._posteriors.get(s, P0)
            indicator = 1.0 if label == "KNOWS" else 0.0
            errors.append(abs(p_knows - indicator))
        return sum(errors) / len(errors) if errors else 0.0

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)
