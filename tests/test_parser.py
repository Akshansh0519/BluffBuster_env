"""
tests/test_parser.py — Parser unit tests (Phase 0 gate blocker).

All 10 test cases must pass before any training code is written.
"""
from __future__ import annotations

import json

import pytest

from examiner_env.action_parser import parse, validate
from examiner_env.models import AskAction, ClassifyAction, MalformedAction, ValidationResult

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────────

VALID_ASK_JSON = json.dumps({
    "action_type": "ask",
    "section_id": "S01",
    "question_text": "Why does momentum help gradient descent convergence?",
})

VALID_CLASSIFY_JSON = json.dumps({
    "action_type": "classify",
    "classifications": {
        "S01": "KNOWS", "S02": "FAKING", "S03": "KNOWS",
        "S04": "FAKING", "S05": "KNOWS", "S06": "FAKING",
        "S07": "KNOWS", "S08": "FAKING", "S09": "KNOWS",
        "S10": "FAKING",
    },
})

ALL_SECTIONS = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"]


def _make_classify(**overrides) -> str:
    base = {
        "action_type": "classify",
        "classifications": {
            "S01": "KNOWS", "S02": "FAKING", "S03": "KNOWS",
            "S04": "FAKING", "S05": "KNOWS", "S06": "FAKING",
            "S07": "KNOWS", "S08": "FAKING", "S09": "KNOWS",
            "S10": "FAKING",
        },
    }
    base.update(overrides)
    return json.dumps(base)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: empty string
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_string_returns_malformed():
    result = parse("")
    assert isinstance(result, MalformedAction), (
        "parse('') must return MalformedAction"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: valid ask
# ──────────────────────────────────────────────────────────────────────────────

def test_valid_ask_returns_ask_action():
    result = parse(VALID_ASK_JSON)
    assert isinstance(result, AskAction), f"Expected AskAction, got {type(result)}: {result}"
    assert result.section_id == "S01"
    assert "momentum" in result.question_text


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: valid classify (all 10 sections)
# ──────────────────────────────────────────────────────────────────────────────

def test_valid_classify_all_10_sections():
    result = parse(VALID_CLASSIFY_JSON)
    assert isinstance(result, ClassifyAction), f"Expected ClassifyAction, got {type(result)}: {result}"
    assert set(result.classifications.keys()) == set(ALL_SECTIONS)
    assert all(v in ("KNOWS", "FAKING") for v in result.classifications.values())


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: missing action_type
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_action_type_returns_malformed():
    payload = json.dumps({"section_id": "S01", "question_text": "Why does X happen?"})
    result = parse(payload)
    assert isinstance(result, MalformedAction), "Missing action_type must be MalformedAction"


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: missing section_id in ask
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_section_id_in_ask_returns_malformed():
    payload = json.dumps({
        "action_type": "ask",
        "question_text": "Why does gradient descent converge?",
    })
    result = parse(payload)
    assert isinstance(result, MalformedAction), "Ask without section_id must be MalformedAction"


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: wrong label case in classify ("knows" not "KNOWS")
# ──────────────────────────────────────────────────────────────────────────────

def test_wrong_label_case_returns_malformed():
    bad_labels = {s: "knows" if i % 2 == 0 else "FAKING" for i, s in enumerate(ALL_SECTIONS)}
    payload = json.dumps({"action_type": "classify", "classifications": bad_labels})
    result = parse(payload)
    assert isinstance(result, MalformedAction), (
        "Classify with lowercase 'knows' label must be MalformedAction"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: partial classify (only 5 of 10 sections)
# ──────────────────────────────────────────────────────────────────────────────

def test_partial_classify_with_invalid_section_returns_malformed():
    """Classify with an invalid section key (S99) must be MalformedAction."""
    payload = json.dumps({
        "action_type": "classify",
        "classifications": {"S01": "KNOWS", "S99": "FAKING"},
    })
    result = parse(payload)
    assert isinstance(result, MalformedAction), (
        "Classify with invalid section S99 must be MalformedAction"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: non-JSON prose
# ──────────────────────────────────────────────────────────────────────────────

def test_non_json_prose_returns_malformed():
    prose = "I think the student knows gradient descent very well."
    result = parse(prose)
    assert isinstance(result, MalformedAction), "Plain prose must be MalformedAction"


# ──────────────────────────────────────────────────────────────────────────────
# Test 9: JSON embedded inside prose (must NOT be extracted)
# ──────────────────────────────────────────────────────────────────────────────

def test_json_inside_prose_returns_malformed():
    prose = (
        'Sure, here is my action: {"action_type": "ask", '
        '"section_id": "S01", "question_text": "Why does this work?"}'
    )
    result = parse(prose)
    assert isinstance(result, MalformedAction), (
        "JSON embedded in prose must NOT be extracted — must return MalformedAction"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 10: near-duplicate ask to same section triggers P_repetition
# ──────────────────────────────────────────────────────────────────────────────

def test_near_duplicate_ask_triggers_repetition_penalty():
    question_1 = "Why does momentum help gradient descent convergence?"
    question_2 = "Why does momentum help gradient descent convergence speed?"  # very similar

    history = [
        {"section_id": "S01", "question": question_1, "response": "Because..."}
    ]

    action = parse(json.dumps({
        "action_type": "ask",
        "section_id": "S01",
        "question_text": question_2,
    }))
    assert isinstance(action, AskAction), "Near-duplicate ask should still parse as AskAction"

    result = validate(action, ALL_SECTIONS, history)
    assert isinstance(result, ValidationResult)
    assert "P_repetition" in result.penalties, (
        "Near-duplicate ask to same section must produce P_repetition penalty"
    )
    assert result.valid is False


# ──────────────────────────────────────────────────────────────────────────────
# Bonus test: invalid section_id in ask triggers P_invalid_sec
# ──────────────────────────────────────────────────────────────────────────────

def test_invalid_section_in_validate_triggers_penalty():
    """Ask with invalid section_id parses to MalformedAction (Pydantic catches it)."""
    payload = json.dumps({
        "action_type": "ask",
        "section_id": "S99",
        "question_text": "What is a non-existent section?",
    })
    result = parse(payload)
    assert isinstance(result, MalformedAction), (
        "Ask with invalid section_id S99 must be MalformedAction (Pydantic validator rejects it)"
    )
