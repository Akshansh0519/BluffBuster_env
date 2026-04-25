"""
baselines.py — Three reference examiners for evaluation comparison.

All three implement the same interface:
    examiner.act(observation: dict) -> str (JSON string, same as TrainedExaminer)

This matches the env.step() contract: env.step() always receives a raw JSON string.

Observation schema:
    {
      "section_titles": dict[section_id, str],
      "section_ids": list[str],
      "turn": int,
      "remaining_turns": int,
      "dialogue_history": list[dict],   # [{section_id, question, response}, ...]
    }

Return schema (JSON string):
    Ask:      '{"action_type": "ask", "section_id": "...", "question_text": "..."}'
    Classify: '{"action_type": "classify", "classifications": {"S01": "KNOWS", ...}}'
"""
from __future__ import annotations

import json
import random

from examiner_env.knowledge_base import KB
from examiner_env.models import CANONICAL_SECTIONS
from examiner_env.posterior_oracle import PosteriorTracker


class RandomExaminer:
    """
    Random baseline: asks random questions, classifies at halftime with random labels.
    Represents the floor performance.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def reset(self, section_ids: list[str] | None = None) -> None:
        pass

    def act(self, observation: dict) -> str:
        section_ids: list[str] = observation.get(
            "section_ids", list(observation.get("section_titles", {}).keys())
        )
        remaining = observation.get("remaining_turns", 1)
        turn = observation.get("turn", 0)
        max_turns = turn + remaining

        # Classify at the midpoint or on the last turn
        if remaining <= 1 or turn >= max(1, max_turns // 2):
            return json.dumps({
                "action_type": "classify",
                "classifications": {
                    s: self._rng.choice(["KNOWS", "FAKING"]) for s in section_ids
                },
            })

        section_id = self._rng.choice(section_ids)
        section_kb = KB[section_id]
        probe = self._rng.choice(section_kb.probe_templates)
        return json.dumps({
            "action_type": "ask",
            "section_id": section_id,
            "question_text": probe.template,
        })


class DefinitionalExaminer:
    """
    Asks "What is <section_title>?" for each section in order.
    Classifies by response length heuristic: longer → KNOWS, shorter → FAKING.

    Primary storytelling comparison baseline — weak by design.
    """

    def __init__(self) -> None:
        self._response_lengths: dict[str, int] = {}
        self._asked_order: list[str] = []

    def reset(self, section_ids: list[str] | None = None) -> None:
        self._response_lengths = {}
        self._asked_order = []

    def act(self, observation: dict) -> str:
        section_titles: dict = observation.get("section_titles", {})
        section_ids: list[str] = observation.get(
            "section_ids", list(section_titles.keys())
        )
        remaining = observation.get("remaining_turns", 1)
        history: list[dict] = observation.get("dialogue_history", [])

        # Update response lengths from history
        for turn in history:
            s = turn.get("section_id")
            resp = turn.get("response", "")
            if s and s not in self._response_lengths:
                self._response_lengths[s] = len(resp.split())

        already_asked = {t["section_id"] for t in history}
        remaining_unasked = [s for s in section_ids if s not in already_asked]

        # Classify if no more unasked sections or last turn
        if not remaining_unasked or remaining <= 1:
            lengths = list(self._response_lengths.values())
            threshold = sorted(lengths)[len(lengths) // 2] if lengths else 20
            return json.dumps({
                "action_type": "classify",
                "classifications": {
                    s: "KNOWS"
                    if self._response_lengths.get(s, 0) >= threshold
                    else "FAKING"
                    for s in section_ids
                },
            })

        # Ask definitional question for next section
        target = remaining_unasked[0]
        title = section_titles.get(target, target)
        return json.dumps({
            "action_type": "ask",
            "section_id": target,
            "question_text": f"What is {title}?",
        })


class BayesianHeuristicExaminer:
    """
    Uses a PosteriorTracker to pick the most uncertain section and ask
    progressively deeper probes (definitional → mechanism → edge_case → counterexample).
    Classifies when all posteriors exceed 0.4 margin from 0.5 OR on the last turn.
    """

    _PROBE_CYCLE = ["definitional", "mechanism", "edge_case", "counterexample"]

    def __init__(self, section_ids: list[str] | None = None, kb: dict | None = None) -> None:
        self._section_ids = section_ids or CANONICAL_SECTIONS
        self._kb = kb or KB
        self._tracker = PosteriorTracker(self._section_ids)
        self._asked_type_idx: dict[str, int] = {s: 0 for s in self._section_ids}
        self._last_response: dict[str, str] = {}

    def reset(self, section_ids: list[str] | None = None) -> None:
        if section_ids:
            self._section_ids = section_ids
        self._tracker = PosteriorTracker(self._section_ids)
        self._asked_type_idx = {s: 0 for s in self._section_ids}
        self._last_response = {}

    def observe_response(self, section_id: str, response: str) -> None:
        """Call after receiving a student response to update posteriors."""
        self._tracker.update(section_id, response, self._kb)

    def act(self, observation: dict) -> str:
        section_ids: list[str] = observation.get(
            "section_ids", list(observation.get("section_titles", {}).keys())
        )
        remaining = observation.get("remaining_turns", 1)
        history: list[dict] = observation.get("dialogue_history", [])

        # Update tracker from any new history entries we haven't seen
        seen_count = len(self._last_response)
        for turn in history[seen_count:]:
            s = turn.get("section_id", "")
            resp = turn.get("response", "")
            if s and resp:
                self._tracker.update(s, resp, self._kb)
                self._last_response[s] = resp

        posteriors = self._tracker.current_posteriors()

        # Classify conditions: last turn or all sections confident
        all_confident = all(
            abs(posteriors.get(s, 0.5) - 0.5) > 0.4 for s in section_ids
        )

        if remaining <= 1 or all_confident:
            return json.dumps({
                "action_type": "classify",
                "classifications": {
                    s: "KNOWS" if posteriors.get(s, 0.5) > 0.5 else "FAKING"
                    for s in section_ids
                },
            })

        # Find highest-uncertainty section (closest to 0.5) among under-explored
        uncertain_sections = [
            s for s in section_ids if abs(posteriors.get(s, 0.5) - 0.5) <= 0.4
        ]
        if not uncertain_sections:
            uncertain_sections = section_ids

        target = min(
            uncertain_sections,
            key=lambda s: abs(posteriors.get(s, 0.5) - 0.5),
        )

        # Cycle through probe types for this section
        probe_type_idx = self._asked_type_idx.get(target, 0) % len(self._PROBE_CYCLE)
        desired_probe_type = self._PROBE_CYCLE[probe_type_idx]
        self._asked_type_idx[target] = probe_type_idx + 1

        section_kb = self._kb[target]

        # Find a probe template of the desired type; fall back to any template
        matching = [p for p in section_kb.probe_templates if p.probe_type == desired_probe_type]
        probe = matching[0] if matching else section_kb.probe_templates[0]

        return json.dumps({
            "action_type": "ask",
            "section_id": target,
            "question_text": probe.template,
        })
