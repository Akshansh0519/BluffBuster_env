"""
action_parser.py — Strict JSON action parser for The Examiner.

Rules (enforced, no exceptions):
  - parse() uses json.loads() only — no regex extraction, no coercion.
  - Any failure at any stage returns MalformedAction.
  - validate() checks semantic constraints and returns ValidationResult with penalties.
"""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from examiner_env.models import (
    AskAction,
    CANONICAL_SECTIONS_SET,
    ClassifyAction,
    MalformedAction,
    ValidationResult,
)

# ──────────────────────────────────────────────────────────────────────────────
# parse
# ──────────────────────────────────────────────────────────────────────────────

def parse(text: str) -> AskAction | ClassifyAction | MalformedAction:
    """
    Parse raw model output into a typed action.

    Guarantees:
      - Returns MalformedAction for ANY invalid input.
      - Never coerces, strips, or modifies the input text.
      - Never uses eval() or regex to extract JSON.
    """
    if not text or not text.strip():
        return MalformedAction(reason="empty input")

    # Step 1: JSON parse — no fallback
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return MalformedAction(reason=f"not valid JSON: {exc}")

    if not isinstance(data, dict):
        return MalformedAction(reason="JSON root must be an object, got array or scalar")

    # Step 2: action_type dispatch
    action_type = data.get("action_type")
    if action_type is None:
        return MalformedAction(reason="missing required field 'action_type'")

    if action_type == "ask":
        try:
            return AskAction(**data)
        except (ValidationError, TypeError) as exc:
            return MalformedAction(reason=f"invalid ask action: {exc}")

    if action_type == "classify":
        try:
            return ClassifyAction(**data)
        except (ValidationError, TypeError) as exc:
            return MalformedAction(reason=f"invalid classify action: {exc}")

    return MalformedAction(
        reason=f"unknown action_type '{action_type}' — must be 'ask' or 'classify'"
    )


# ──────────────────────────────────────────────────────────────────────────────
# validate
# ──────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    """Lower-case, strip punctuation, split into word tokens."""
    return set(re.sub(r"[^a-z0-9\s]", "", text.lower()).split())


def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings (punctuation-stripped)."""
    words_a = _tokenise(a)
    words_b = _tokenise(b)
    if not words_a and not words_b:
        return 1.0
    union = words_a | words_b
    if not union:
        return 0.0
    return len(words_a & words_b) / len(union)


def validate(
    action: AskAction | ClassifyAction,
    canonical_sections: list[str],
    history: list[dict],
) -> ValidationResult:
    """
    Validate semantic constraints that cannot be caught by Pydantic alone.

    For AskAction checks:
      - section_id is in canonical_sections → penalty P_invalid_sec
      - near-duplicate question to same section (Jaccard > 0.85) → penalty P_repetition

    For ClassifyAction checks:
      - all canonical sections are present → penalty P_cov (handled by Pydantic, logged here)

    Returns ValidationResult with valid=True iff no penalties were found.
    """
    canonical_set = frozenset(canonical_sections)
    penalties: list[str] = []

    if isinstance(action, AskAction):
        # Invalid section
        if action.section_id not in canonical_set:
            penalties.append("P_invalid_sec")

        # Near-duplicate check: same section in history with Jaccard > 0.85
        prior_questions = [
            turn["question"]
            for turn in history
            if turn.get("section_id") == action.section_id
        ]
        for prior_q in prior_questions:
            if _jaccard(action.question_text, prior_q) > 0.85:
                penalties.append("P_repetition")
                break  # one penalty per action

    elif isinstance(action, ClassifyAction):
        provided = set(action.classifications.keys())
        missing = canonical_set - provided
        if missing:
            penalties.append("P_cov")

    valid = len(penalties) == 0
    return ValidationResult(valid=valid, penalties=penalties, info={})
