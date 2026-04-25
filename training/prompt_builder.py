"""
training/prompt_builder.py — Examiner prompt construction.
C2 owns.

CRITICAL INVARIANT: The prompt must NEVER contain:
  - True section labels (KNOWS / FAKING per section)
  - Student style IDs (K1–K3, F1–F4)
  - Posterior probabilities or confidence scores
  - Any hidden state from the environment

'KNOWS' and 'FAKING' MAY appear ONLY as schema example values in the JSON template.
Validator will run the leakage-check script before Gate 2 merges.
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

_ASK_EXAMPLE = '{"action_type": "ask", "section_id": "<S01-S10>", "question_text": "<your question, min 10 chars>"}'
_CLASSIFY_EXAMPLE = (
    '{"action_type": "classify", "classifications": '
    '{"S01": "KNOWS"|"FAKING", "S02": "KNOWS"|"FAKING", '
    '"S03": "KNOWS"|"FAKING", "S04": "KNOWS"|"FAKING", '
    '"S05": "KNOWS"|"FAKING", "S06": "KNOWS"|"FAKING", '
    '"S07": "KNOWS"|"FAKING", "S08": "KNOWS"|"FAKING", '
    '"S09": "KNOWS"|"FAKING", "S10": "KNOWS"|"FAKING"}}'
)


def build_prompt(observation: dict) -> str:
    """
    Construct the examiner prompt from the current observation.

    observation keys:
      section_titles  — dict[section_id, title]  (canonical, no labels)
      turn            — int (current turn, 0-indexed)
      remaining_turns — int
      dialogue_history — list of dicts with keys: section_id, question, response
    """
    section_titles: dict = observation.get("section_titles", SECTION_TITLES)
    turn: int = observation.get("turn", 0)
    remaining_turns: int = observation.get("remaining_turns", 4)
    dialogue_history: list = observation.get("dialogue_history", [])

    # ── Section list ──
    section_lines = "\n".join(
        f"  {s_id}: {title}" for s_id, title in section_titles.items()
    )

    # ── Dialogue history ──
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

    prompt = f"""You are an examiner. Your goal: determine which ML sections a student genuinely KNOWS vs is FAKING.

Sections under examination:
{section_lines}

Turn: {turn + 1} | Remaining turns: {remaining_turns}

Dialogue History:
{history_block}

Instructions:
- Ask targeted diagnostic questions (why/how/edge-case probes reveal bluffing).
- When confident enough, issue a CLASSIFY action covering ALL 10 sections.
- You MUST classify in the final turn if remaining_turns == 1.

Output ONLY valid JSON matching exactly one of:

ASK:
{_ASK_EXAMPLE}

CLASSIFY (all 10 sections required):
{_CLASSIFY_EXAMPLE}

Your JSON:"""

    # ── Leakage guard (safety net — should never trigger in correct usage) ──
    _assert_no_leakage(prompt, section_titles)

    return prompt


def _assert_no_leakage(prompt: str, section_titles: dict) -> None:
    """
    Assert no hidden-state information is present in the prompt.
    Checks that no section ID has a label attached to it (e.g. 'S01: ... KNOWS').
    Raises AssertionError if leakage is detected.
    """
    for s_id in section_titles:
        # Pattern: section ID followed by KNOWS or FAKING anywhere in same line
        if re.search(rf"{s_id}.*\b(KNOWS|FAKING)\b", prompt):
            raise AssertionError(
                f"CRITICAL: Hidden label leaked for section {s_id} in prompt! "
                "Check build_prompt() — observation must not contain true_labels."
            )
