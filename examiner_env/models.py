"""
models.py — Pydantic v2 schemas for The Examiner RL environment.

All data types used across the codebase live here so other modules can
import without circular dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
CANONICAL_SECTIONS: list[str] = [
    "S01", "S02", "S03", "S04", "S05",
    "S06", "S07", "S08", "S09", "S10",
]
CANONICAL_SECTIONS_SET: frozenset[str] = frozenset(CANONICAL_SECTIONS)

SECTION_TITLES: dict[str, str] = {
    "S01": "Gradient Descent and Optimization",
    "S02": "Backpropagation",
    "S03": "Overfitting and Regularization",
    "S04": "Attention Mechanisms",
    "S05": "Transformer Architecture",
    "S06": "Loss Functions and Their Geometry",
    "S07": "Batch Normalization",
    "S08": "Convolutional Neural Networks",
    "S09": "Reinforcement Learning Basics",
    "S10": "Embeddings and Representation Learning",
}

# ──────────────────────────────────────────────────────────────────────────────
# Action schemas
# ──────────────────────────────────────────────────────────────────────────────

class AskAction(BaseModel):
    """Valid Ask action produced by the examiner."""
    action_type: Literal["ask"]
    section_id: str
    question_text: str = Field(min_length=10)

    @field_validator("section_id")
    @classmethod
    def validate_section(cls, v: str) -> str:
        if v not in CANONICAL_SECTIONS_SET:
            raise ValueError(
                f"section_id '{v}' is not a canonical section. "
                f"Must be one of {CANONICAL_SECTIONS}."
            )
        return v


class ClassifyAction(BaseModel):
    """Valid Classify action produced by the examiner (terminates episode)."""
    action_type: Literal["classify"]
    classifications: dict[str, Literal["KNOWS", "FAKING"]]

    @model_validator(mode="after")
    def check_all_sections_valid(self) -> "ClassifyAction":
        """All provided section keys must be canonical; at least 1 required."""
        provided = set(self.classifications.keys())
        invalid = provided - CANONICAL_SECTIONS_SET
        if invalid:
            raise ValueError(
                f"classifications contains invalid section IDs: {sorted(invalid)}. "
                f"Valid sections: {CANONICAL_SECTIONS}"
            )
        if not provided:
            raise ValueError("classifications must not be empty")
        return self


class MalformedAction(BaseModel):
    """Returned by the parser when output is not valid JSON or fails validation."""
    reason: str


# ──────────────────────────────────────────────────────────────────────────────
# Student profile
# ──────────────────────────────────────────────────────────────────────────────

class StudentProfile(BaseModel):
    """Per-section style assignment for the student simulator."""
    knowledge_mode: Literal["KNOWS", "FAKING"]
    style: Literal["K1", "K2", "K3", "F1", "F2", "F3", "F4"]
    section_id: str
    verbosity: Literal["brief", "medium", "verbose"]
    confidence_pattern: Literal["hedging", "neutral", "confident"]
    mechanism_cue_emit_rate: float = Field(ge=0.0, le=1.0)
    misconception_emit_rate: float = Field(ge=0.0, le=1.0)
    style_specific_params: dict = Field(default_factory=dict)
    seed: int


# ──────────────────────────────────────────────────────────────────────────────
# Validation result
# ──────────────────────────────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    """Output of action_parser.validate()."""
    valid: bool
    penalties: list[str] = Field(default_factory=list)
    info: dict = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Episode state (internal, not exposed to examiner)
# ──────────────────────────────────────────────────────────────────────────────

class EpisodeState(BaseModel):
    """Full internal episode state, never serialised into the observation."""
    episode_seed: int
    section_ids: list[str]
    true_labels: dict[str, Literal["KNOWS", "FAKING"]]   # hidden
    style_assignments: dict[str, str]                      # hidden
    dialogue_history: list[dict] = Field(default_factory=list)
    turn: int = 0
    max_turns: int = 4
    done: bool = False
    n_malformed: int = 0
    n_repetition: int = 0
    n_invalid_sec: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Episode result (input to compute_reward)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """Snapshot passed to compute_reward() at episode end."""
    classifications: dict[str, str]            # examiner's final verdict
    true_labels: dict[str, str]                # ground truth
    section_ids: list[str]
    turns_used: int
    max_turns: int
    dialogue_history: list[dict]               # list of {section_id, question, response}
    posterior_tracker: Any                     # PosteriorTracker instance
    n_malformed: int
    n_repetition: int
    n_invalid_sec: int


# ──────────────────────────────────────────────────────────────────────────────
# Reward breakdown (frozen dataclass — immutable after creation)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RewardBreakdown:
    """
    Complete reward decomposition for one episode.

    All 11 components must sum to R_total ± 1e-9.
    R_total is raw, bounded in [−2.05, +1.95] — never normalised here.
    posterior_trace: list[dict[section_id, p_t]] — one dict per turn.
    info_gain_per_turn: list[float] — ΔH_t per turn.
    """
    R_acc: float
    R_asym: float
    R_cal: float
    R_eff: float
    R_cov: float
    R_info: float
    R_qual: float
    R_div: float
    P_malformed: float
    P_repetition: float
    P_invalid_sec: float
    R_total: float
    posterior_trace: tuple      # tuple of dicts for hashability (frozen dataclass)
    info_gain_per_turn: tuple   # tuple of floats

    def as_dict(self) -> dict:
        """Return all numeric fields as a plain dict (for W&B logging)."""
        return {
            "R_acc": self.R_acc,
            "R_asym": self.R_asym,
            "R_cal": self.R_cal,
            "R_eff": self.R_eff,
            "R_cov": self.R_cov,
            "R_info": self.R_info,
            "R_qual": self.R_qual,
            "R_div": self.R_div,
            "P_malformed": self.P_malformed,
            "P_repetition": self.P_repetition,
            "P_invalid_sec": self.P_invalid_sec,
            "R_total": self.R_total,
            "posterior_trace": list(self.posterior_trace),
            "info_gain_per_turn": list(self.info_gain_per_turn),
        }
