"""
training/prompt_builder.py — Examiner prompt construction.
C2 owns.

CRITICAL INVARIANT: The prompt must NEVER contain:
  - True section labels (KNOWS / FAKING) attached to a specific section ID
  - Student style IDs (K1–K3, F1–F4)
  - Posterior probabilities or confidence scores
  - Any hidden state from the environment

'KNOWS' and 'FAKING' MAY appear in the fixed schema template block (Part B).
They must NEVER appear in the observation-derived block (Part A).

Leakage guard design (fix for Validator blocker):
  The prompt is built in two independent parts:
    Part A — observation-derived: section list lines, turn, remaining turns,
              dialogue history. This is the ONLY part checked for leakage.
    Part B — fixed schema template: instructions + ASK/CLASSIFY JSON examples.
              Never scanned by the guard (KNOWS/FAKING appear there as schema
              literals and are intentional).
  _assert_no_leakage() receives Part A ONLY.
"""

from __future__ import annotations

import re

SECTION_TITLES = {
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

# ── Fixed schema template (Part B) — never scanned for leakage ──
_SCHEMA_BLOCK = """Output ONLY valid JSON matching exactly one of:

ASK:
{"action_type": "ask", "section_id": "<S01-S10>", "question_text": "<your question, min 10 chars>"}

CLASSIFY (all 10 sections required):
{"action_type": "classify", "classifications": {"S01": "KNOWS"|"FAKING", "S02": "KNOWS"|"FAKING", "S03": "KNOWS"|"FAKING", "S04": "KNOWS"|"FAKING", "S05": "KNOWS"|"FAKING", "S06": "KNOWS"|"FAKING", "S07": "KNOWS"|"FAKING", "S08": "KNOWS"|"FAKING", "S09": "KNOWS"|"FAKING", "S10": "KNOWS"|"FAKING"}}

Your JSON:"""

_INSTRUCTIONS_BLOCK = """Instructions:
- Ask targeted diagnostic questions (why/how/edge-case probes expose bluffing).
- When confident enough, issue a CLASSIFY covering ALL 10 sections.
- You MUST classify in the final turn if remaining_turns == 1."""


def build_prompt(observation: dict) -> str:
    """
    Construct the examiner prompt from the current observation.

    observation keys (all from ExaminerEnv._build_observation — no hidden state):
      section_titles   — dict[section_id, title]  (canonical, no labels)
      turn             — int (current turn, 0-indexed)
      remaining_turns  — int
      dialogue_history — list of dicts with keys: section_id, question, response

    Returns a string prompt. Raises AssertionError if observation-derived Part A
    contains a section ID immediately followed by a label (KNOWS/FAKING) — that
    would indicate env leakage of true_labels.
    """
    section_titles: dict = observation.get("section_titles", SECTION_TITLES)
    turn: int = observation.get("turn", 0)
    remaining_turns: int = observation.get("remaining_turns", 4)
    dialogue_history: list = observation.get("dialogue_history", [])

    # ── Part A: observation-derived content (leakage-checked) ────────────
    section_lines = "\n".join(
        f"  {s_id}: {title}" for s_id, title in section_titles.items()
    )

    if dialogue_history:
        history_lines = []
        for i, entry in enumerate(dialogue_history):
            history_lines.append(
                f"  Turn {i + 1} — Asked {entry['section_id']}: \"{entry['question']}\"\n"
                f"           Student: \"{entry['response']}\""
            )
        history_block = "\n".join(history_lines)
    else:
        history_block = "  (no questions asked yet)"

    part_a = (
        f"You are an examiner. Determine which ML sections a student KNOWS vs is FAKING.\n\n"
        f"Sections under examination:\n"
        f"{section_lines}\n\n"
        f"Turn: {turn + 1} | Remaining turns: {remaining_turns}\n\n"
        f"Dialogue History:\n"
        f"{history_block}"
    )

    # ── Leakage guard — runs on Part A ONLY, never on schema template ────
    _assert_no_leakage(part_a, section_titles)

    # ── Part B: fixed schema template (intentionally contains KNOWS/FAKING) ──
    prompt = f"{part_a}\n\n{_INSTRUCTIONS_BLOCK}\n\n{_SCHEMA_BLOCK}"

    return prompt


def _assert_no_leakage(observation_text: str, section_titles: dict) -> None:
    """
    Assert no hidden-state labels appear in the observation-derived block.

    Checks each section's rendered title line to ensure it is not followed
    immediately by KNOWS or FAKING — which would mean the env leaked true_labels
    into the observation dict.

    This function MUST only be called on Part A (observation-derived text).
    It MUST NOT be called on the full prompt (which contains schema literals).

    Raises AssertionError with a clear message on leakage.

    Deliberate-leakage test: if the observation somehow injects
    '  S01: Gradient Descent (KNOWS)' into section_titles, this will catch it.
    """
    for s_id in section_titles:
        # Match: the section ID and label on the same line, as produced by
        # a corrupted observation (e.g. "  S01: title KNOWS" or "S01 — KNOWS").
        # Regex scope: anchored to one line (\n not consumed by .*) so only
        # observation-derived lines are matched — not distant schema text.
        if re.search(rf"^.*\b{re.escape(s_id)}\b.*\b(KNOWS|FAKING)\b", observation_text, re.MULTILINE):
            raise AssertionError(
                f"CRITICAL: Hidden label leaked for section {s_id} in prompt (Part A). "
                f"Observation must not contain true_labels. "
                f"Check ExaminerEnv._build_observation() — it must never include "
                f"true_labels, style_ids, or posterior values."
            )


# ── Smoke test (run with: python -m training.prompt_builder) ─────────────
if __name__ == "__main__":
    print("=== Smoke test: build_prompt ===\n")

    # 1. Minimal observation — no history
    obs_empty = {
        "section_titles": SECTION_TITLES,
        "turn": 0,
        "remaining_turns": 4,
        "dialogue_history": [],
    }
    prompt = build_prompt(obs_empty)
    assert "S01" in prompt, "S01 not in prompt"
    assert "KNOWS" in prompt, "KNOWS schema literal missing"
    assert "FAKING" in prompt, "FAKING schema literal missing"
    assert "Your JSON:" in prompt, "schema footer missing"
    print("[OK] Empty history")
    print(f"  Prompt length: {len(prompt)} chars")

    # 2. With dialogue history
    obs_with_history = {
        "section_titles": SECTION_TITLES,
        "turn": 2,
        "remaining_turns": 2,
        "dialogue_history": [
            {"section_id": "S01", "question": "Why does momentum help?", "response": "It accelerates convergence..."},
            {"section_id": "S02", "question": "Explain the chain rule here.", "response": "Using the Jacobian..."},
        ],
    }
    prompt2 = build_prompt(obs_with_history)
    assert "momentum" in prompt2, "dialogue history not in prompt"
    print("[OK] With history")

    # 3. Deliberate leakage test — must raise AssertionError
    obs_poisoned = {
        "section_titles": {"S01": "Gradient Descent (KNOWS)", "S02": "Backpropagation"},
        "turn": 0,
        "remaining_turns": 4,
        "dialogue_history": [],
    }
    try:
        build_prompt(obs_poisoned)
        raise RuntimeError("FAIL -- poisoned observation did NOT raise AssertionError!")
    except AssertionError as e:
        print(f"[OK] Poisoned observation correctly caught: {e}")

    # 4. Confirm schema still present after leakage guard
    assert '"S01": "KNOWS"|"FAKING"' in prompt, "CLASSIFY schema missing from output"
    print("[OK] Schema literals present in final prompt")

    print("\n=== All smoke tests passed ===")
