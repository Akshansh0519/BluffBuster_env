"""
student.py — 7-style adversarial student simulator.

Key design rules (from guardrails.md):
  - All 7 styles implemented with EXACT leak rates from architecture.md §5.
  - Probe sensitivity: mechanism probe → faker mech ×0.5, misc ×1.5.
  - Deterministic RNG: random.Random seeded with (episode_seed, turn, section_id).
  - Same (episode_seed, turn, section_id, question_text) → byte-identical response.
"""
from __future__ import annotations

import random
import re
from typing import Literal

from examiner_env.knowledge_base import KB, SectionKB
from examiner_env.models import StudentProfile

# ──────────────────────────────────────────────────────────────────────────────
# Style specification table (DO NOT CHANGE RATES)
# ──────────────────────────────────────────────────────────────────────────────

_STYLE_SPECS: dict[str, dict] = {
    "K1": {
        "knowledge_mode": "KNOWS",
        "mechanism_cue_emit_rate": 0.85,
        "misconception_emit_rate": 0.05,
        "verbosity_weights": {"brief": 0.10, "medium": 0.40, "verbose": 0.50},
        "confidence_pattern": "confident",
        "style_specific_params": {},
    },
    "K2": {
        "knowledge_mode": "KNOWS",
        "mechanism_cue_emit_rate": 0.55,
        "misconception_emit_rate": 0.05,
        "verbosity_weights": {"brief": 0.60, "medium": 0.35, "verbose": 0.05},
        "confidence_pattern": "neutral",
        "style_specific_params": {},
    },
    "K3": {
        "knowledge_mode": "KNOWS",
        "mechanism_cue_emit_rate": 0.65,
        "misconception_emit_rate": 0.08,
        "verbosity_weights": {"brief": 0.20, "medium": 0.60, "verbose": 0.20},
        "confidence_pattern": "hedging",
        "style_specific_params": {},
    },
    "F1": {
        "knowledge_mode": "FAKING",
        "mechanism_cue_emit_rate": 0.15,
        "misconception_emit_rate": 0.30,
        "verbosity_weights": {"brief": 0.30, "medium": 0.50, "verbose": 0.20},
        "confidence_pattern": "confident",
        "style_specific_params": {"collapse_under_mechanism_probe": 0.80},
    },
    "F2": {
        "knowledge_mode": "FAKING",
        "mechanism_cue_emit_rate": 0.20,
        "misconception_emit_rate": 0.25,
        "verbosity_weights": {"brief": 0.20, "medium": 0.55, "verbose": 0.25},
        "confidence_pattern": "confident",
        "style_specific_params": {"jargon_mirror_rate": 0.70},
    },
    "F3": {
        "knowledge_mode": "FAKING",
        "mechanism_cue_emit_rate": 0.10,
        "misconception_emit_rate": 0.20,
        "verbosity_weights": {"brief": 0.10, "medium": 0.50, "verbose": 0.40},
        "confidence_pattern": "neutral",
        "style_specific_params": {"topic_drift_rate": 0.60},
    },
    "F4": {
        "knowledge_mode": "FAKING",
        "mechanism_cue_emit_rate": 0.05,
        "misconception_emit_rate": 0.40,
        "verbosity_weights": {"brief": 0.40, "medium": 0.45, "verbose": 0.15},
        "confidence_pattern": "confident",
        "style_specific_params": {"specificity_score_cap": 0.20},
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Probe detection
# ──────────────────────────────────────────────────────────────────────────────

_PROBE_KEYWORDS = frozenset({
    "why", "how", "mechanism", "because", "explain",
    "edge case", "counterexample", "what if", "failure",
    "when does", "under what", "why does", "derive",
    "describe the process", "walk me through",
})

_PROBE_RE = re.compile(
    r"\b(why|how|mechanism|because|explain|derive|failure|counterexample)\b"
    r"|edge case|what if|when does|under what|why does|walk me through|describe the process",
    re.IGNORECASE,
)


def _is_probe(question_text: str) -> bool:
    """Return True if question contains a mechanism/edge-case probe keyword."""
    return bool(_PROBE_RE.search(question_text))


# ──────────────────────────────────────────────────────────────────────────────
# Profile sampling
# ──────────────────────────────────────────────────────────────────────────────

def sample_profile(
    section_id: str,
    knowledge_mode: Literal["KNOWS", "FAKING"],
    episode_seed: int,
    section_idx: int,
) -> StudentProfile:
    """
    Sample a StudentProfile deterministically for a given (episode_seed, section).
    Same inputs → same profile across all runs.
    """
    rng = random.Random()
    rng.seed(hash((episode_seed, section_idx, knowledge_mode)))

    if knowledge_mode == "KNOWS":
        style = rng.choice(["K1", "K2", "K3"])
    else:
        style = rng.choice(["F1", "F2", "F3", "F4"])

    spec = _STYLE_SPECS[style]
    verbosity_choices = list(spec["verbosity_weights"].keys())
    verbosity_weights = list(spec["verbosity_weights"].values())
    verbosity = rng.choices(verbosity_choices, weights=verbosity_weights, k=1)[0]

    return StudentProfile(
        knowledge_mode=knowledge_mode,
        style=style,
        section_id=section_id,
        verbosity=verbosity,
        confidence_pattern=spec["confidence_pattern"],
        mechanism_cue_emit_rate=spec["mechanism_cue_emit_rate"],
        misconception_emit_rate=spec["misconception_emit_rate"],
        style_specific_params=spec["style_specific_params"],
        seed=episode_seed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Response assembly helpers
# ──────────────────────────────────────────────────────────────────────────────

def _verbosity_sentences(verbosity: str) -> tuple[int, int]:
    """Return (min_sentences, max_sentences) for a verbosity level."""
    return {"brief": (1, 2), "medium": (2, 4), "verbose": (4, 6)}[verbosity]


def _hedge_prefix(rng: random.Random) -> str:
    hedges = [
        "I believe ", "I think ", "If I recall correctly, ",
        "I'm fairly certain that ", "As far as I understand, ",
    ]
    return rng.choice(hedges)


def _confident_prefix(rng: random.Random) -> str:
    prefixes = [
        "Essentially, ", "The key point is that ", "Simply put, ",
        "At its core, ", "The fundamental idea is that ",
    ]
    return rng.choice(prefixes)


def _build_knows_response(
    question_text: str,
    section_kb: SectionKB,
    profile: StudentProfile,
    rng: random.Random,
    effective_mech_rate: float,
) -> str:
    """Assemble a KNOWS-style response from KB mechanism cues."""
    min_s, max_s = _verbosity_sentences(profile.verbosity)

    # Collect mechanism cue phrases that fire
    selected_cues: list[str] = []
    for cue in section_kb.mechanism_cues:
        prob = effective_mech_rate * cue.weight
        if rng.random() < prob:
            selected_cues.append(cue.phrase)

    # Always include at least one key concept
    key = rng.choice(section_kb.key_concepts)

    # Build sentences
    sentences: list[str] = []
    n_target = rng.randint(min_s, max_s)

    if profile.confidence_pattern == "hedging":
        opener = f"{_hedge_prefix(rng)}the core idea of {key} involves "
    elif profile.confidence_pattern == "confident":
        opener = f"{_confident_prefix(rng)}{key} works by "
    else:
        opener = f"In terms of {key}, "

    if selected_cues:
        first_cue = selected_cues[0]
        sentences.append(f"{opener}{first_cue}.")
        for cue in selected_cues[1:]:
            if len(sentences) >= n_target:
                break
            if profile.confidence_pattern == "hedging":
                sentences.append(f"Furthermore, I think this relies on {cue}.")
            else:
                sentences.append(f"This is because {cue}.")
    else:
        sentences.append(f"{opener}the process described in {key}.")

    if len(sentences) < n_target:
        filler_key = rng.choice(section_kb.key_concepts)
        sentences.append(f"This connects directly to the concept of {filler_key}.")

    if profile.confidence_pattern == "hedging" and rng.random() < 0.4:
        sentences.append("Though I'm less certain about the exact edge cases here.")

    return " ".join(sentences[:n_target])


def _build_faking_response(
    question_text: str,
    section_kb: SectionKB,
    profile: StudentProfile,
    rng: random.Random,
    effective_mech_rate: float,
    effective_misc_rate: float,
    probe_detected: bool,
) -> str:
    """Assemble a FAKING-style response using misconceptions and style params."""
    style = profile.style
    params = profile.style_specific_params
    min_s, max_s = _verbosity_sentences(profile.verbosity)

    # ── F1: definitional bluffer — collapses under mechanism probe ────────────
    if style == "F1":
        if probe_detected and rng.random() < params.get("collapse_under_mechanism_probe", 0.80):
            # Collapse pattern: starts confidently, trails off
            misc = rng.choice(section_kb.common_misconceptions)
            return (
                f"Well, the concept here is basically about {misc.phrase}. "
                "I know the general idea, but the specific mechanism... "
                "I'm not entirely sure about the exact details of how it works step by step."
            )
        else:
            # Definitional recitation
            misc = rng.choice(section_kb.common_misconceptions)
            key = rng.choice(section_kb.key_concepts)
            return (
                f"So {key} is essentially a technique where {misc.phrase}. "
                "It's a fundamental concept in the field."
            )

    # ── F2: jargon-matcher — mirrors terms from question ──────────────────────
    if style == "F2":
        mirror_rate = params.get("jargon_mirror_rate", 0.70)
        # Extract nouns/technical terms from question (simple heuristic: capitalised or ≥6 chars)
        q_words = [w.strip("?.,!") for w in question_text.split() if len(w) >= 5]
        mirrored = ""
        if q_words and rng.random() < mirror_rate:
            term = rng.choice(q_words)
            mirrored = f"Yes, {term} is indeed central here. "

        misc = rng.choice(section_kb.common_misconceptions)
        n = rng.randint(min_s, max_s)
        base = f"{mirrored}In essence, {misc.phrase}."
        if n > 1:
            key = rng.choice(section_kb.key_concepts)
            base += f" This is fundamentally connected to {key}."
        return base

    # ── F3: evasive generaliser — drifts to adjacent topic ───────────────────
    if style == "F3":
        drift_rate = params.get("topic_drift_rate", 0.60)
        if rng.random() < drift_rate:
            # Drift to a neighbouring section's key concepts
            adjacent_ids = [k for k in KB if k != section_kb.section_id]
            adj_id = rng.choice(adjacent_ids)
            adj_key = rng.choice(KB[adj_id].key_concepts)
            misc = rng.choice(section_kb.common_misconceptions)
            return (
                f"That's an interesting question. It's really about {misc.phrase}, "
                f"which connects to broader ideas around {adj_key}. "
                "These concepts are all interrelated in machine learning."
            )
        else:
            misc = rng.choice(section_kb.common_misconceptions)
            return f"At a high level, this is about {misc.phrase}. The details vary by context."

    # ── F4: overconfident handwaver — high-level, zero specificity ───────────
    if style == "F4":
        misc = rng.choice(section_kb.common_misconceptions)
        key = rng.choice(section_kb.key_concepts)
        fillers = [
            f"Honestly, {key} is basically just a standard technique. {misc.phrase}.",
            f"This is essentially just {misc.phrase} — nothing too complicated.",
            f"At the end of the day, {key} boils down to {misc.phrase}. It's quite straightforward.",
        ]
        return rng.choice(fillers)

    # Generic fallback (should not reach here)
    misc = rng.choice(section_kb.common_misconceptions)
    return f"In general, {misc.phrase}."


# ──────────────────────────────────────────────────────────────────────────────
# Main public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_response(
    question_text: str,
    section_id: str,
    profile: StudentProfile,
    kb: dict,
    episode_seed: int,
    turn: int,
) -> str:
    """
    Generate a student response deterministically.

    Guarantee: same (episode_seed, turn, section_id, question_text) →
    byte-identical output across all runs.
    """
    # Seed combines episode, turn, and section for full reproducibility
    rng = random.Random()
    rng.seed(hash((episode_seed, turn, section_id, question_text)))

    section_kb: SectionKB = kb[section_id]
    probe_detected = _is_probe(question_text)

    if profile.knowledge_mode == "KNOWS":
        # Probe does NOT degrade KNOWS styles
        effective_mech_rate = profile.mechanism_cue_emit_rate
        effective_misc_rate = profile.misconception_emit_rate
        return _build_knows_response(
            question_text, section_kb, profile, rng, effective_mech_rate
        )
    else:
        # FAKING: probe degrades mechanism cue emission and increases misconception emission
        if probe_detected:
            effective_mech_rate = profile.mechanism_cue_emit_rate * 0.5
            effective_misc_rate = min(1.0, profile.misconception_emit_rate * 1.5)
        else:
            effective_mech_rate = profile.mechanism_cue_emit_rate
            effective_misc_rate = profile.misconception_emit_rate

        return _build_faking_response(
            question_text, section_kb, profile, rng,
            effective_mech_rate, effective_misc_rate, probe_detected,
        )
