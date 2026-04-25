"""
question_features.py — R_qual question-side scorer.

Computes a score in [0, 1] for an Ask action based on the diagnostic
quality of the question text.  This is purely QUESTION-SIDE — it does
NOT look at the response.

Features (order-independent):
  f1 — question contains a mechanism or edge-case probe keyword
  f2 — question targets a KB probe_type template match
  f3 — question is not a trivial definitional opener ("what is X?")
  f4 — question mentions a specific mechanism cue from the KB
  f5 — question length is within ideal range (10–50 words)

R_qual = w1*f1 + w2*f2 + w3*f3 + w4*f4 + w5*f5   bounded in [0, 1]
"""
from __future__ import annotations

import re

from examiner_env.knowledge_base import KB, SectionKB

# ──────────────────────────────────────────────────────────────────────────────
# Feature weights (must sum to 1.0 for clean interpretation)
# ──────────────────────────────────────────────────────────────────────────────

_W1 = 0.35  # probe keyword present
_W2 = 0.25  # matches a KB template type
_W3 = 0.15  # non-trivial (not "what is X?")
_W4 = 0.15  # references a specific mechanism cue
_W5 = 0.10  # length in sweet-spot (10–50 words)

assert abs(_W1 + _W2 + _W3 + _W4 + _W5 - 1.0) < 1e-9, "Weights must sum to 1.0"

# ──────────────────────────────────────────────────────────────────────────────
# Pattern constants
# ──────────────────────────────────────────────────────────────────────────────

_PROBE_RE = re.compile(
    r"\b(why|how|mechanism|because|explain|derive|failure|counterexample"
    r"|edge case|what if|when does|under what|walk me through|describe the process)\b",
    re.IGNORECASE,
)

_TRIVIAL_RE = re.compile(
    r"^\s*(what\s+is|define|definition\s+of|tell\s+me\s+about)\b",
    re.IGNORECASE,
)

_MECHANISM_PROBE_TYPES = frozenset({"mechanism", "edge_case", "counterexample"})

# ──────────────────────────────────────────────────────────────────────────────
# Individual feature extractors
# ──────────────────────────────────────────────────────────────────────────────

def _f1_has_probe_keyword(question_text: str) -> float:
    return 1.0 if _PROBE_RE.search(question_text) else 0.0


def _f2_matches_mechanism_template(question_text: str, section_kb: SectionKB) -> float:
    """
    Returns 1.0 if any mechanism/edge_case/counterexample probe template
    from the KB section shares ≥2 key words with the question.
    """
    q_words = set(re.sub(r"[^a-z0-9\s]", "", question_text.lower()).split())

    for template in section_kb.probe_templates:
        if template.probe_type not in _MECHANISM_PROBE_TYPES:
            continue
        t_words = set(re.sub(r"[^a-z0-9\s]", "", template.template.lower()).split())
        overlap = q_words & t_words
        # Ignore stop words for overlap count
        stop = {"what", "is", "the", "a", "an", "in", "of", "does", "do", "how",
                "why", "when", "where", "can", "you", "me", "please"}
        meaningful_overlap = overlap - stop
        if len(meaningful_overlap) >= 2:
            return 1.0
    return 0.0


def _f3_non_trivial(question_text: str) -> float:
    return 0.0 if _TRIVIAL_RE.match(question_text.strip()) else 1.0


def _f4_cites_mechanism_cue(question_text: str, section_kb: SectionKB) -> float:
    """Returns 1.0 if the question directly references any KB mechanism cue phrase."""
    q_lower = question_text.lower()
    for cue in section_kb.mechanism_cues:
        # Check if any 2+ consecutive words from the cue appear in the question
        cue_words = cue.phrase.lower().split()
        for i in range(len(cue_words) - 1):
            bigram = f"{cue_words[i]} {cue_words[i+1]}"
            if bigram in q_lower:
                return 1.0
    return 0.0


def _f5_ideal_length(question_text: str) -> float:
    n_words = len(question_text.split())
    return 1.0 if 10 <= n_words <= 50 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Main scorer
# ──────────────────────────────────────────────────────────────────────────────

def compute_r_qual(
    question_text: str,
    section_id: str,
    kb: dict | None = None,
) -> float:
    """
    Compute R_qual for a single Ask action.

    Returns a score in [0.0, 1.0].
    Must only be called with the question — never uses the response.
    """
    if kb is None:
        kb = KB

    section_kb: SectionKB = kb[section_id]

    f1 = _f1_has_probe_keyword(question_text)
    f2 = _f2_matches_mechanism_template(question_text, section_kb)
    f3 = _f3_non_trivial(question_text)
    f4 = _f4_cites_mechanism_cue(question_text, section_kb)
    f5 = _f5_ideal_length(question_text)

    score = _W1 * f1 + _W2 * f2 + _W3 * f3 + _W4 * f4 + _W5 * f5
    return round(min(1.0, max(0.0, score)), 6)


def question_feature_vector(
    question_text: str,
    section_id: str,
    kb: dict | None = None,
) -> dict[str, float]:
    """Return individual feature values (useful for debugging and logging)."""
    if kb is None:
        kb = KB
    section_kb = kb[section_id]
    return {
        "f1_probe_keyword": _f1_has_probe_keyword(question_text),
        "f2_mechanism_template": _f2_matches_mechanism_template(question_text, section_kb),
        "f3_non_trivial": _f3_non_trivial(question_text),
        "f4_cites_cue": _f4_cites_mechanism_cue(question_text, section_kb),
        "f5_ideal_length": _f5_ideal_length(question_text),
        "R_qual": compute_r_qual(question_text, section_id, kb),
    }
