"""
environment.py — ExaminerEnv: OpenEnv-inheriting diagnostic RL environment.

Inherits from openenv.Env (or falls back to a Gymnasium-compatible base class
if openenv is not yet installed).

CRITICAL rules (guardrails.md):
  - Observation NEVER contains true_labels, style_assignments, or posteriors.
  - Reward = 0.0 for every Ask step; full reward only on Classify/exhaustion.
  - action_space and observation_space defined per OpenEnv interface.
  - Registered as "ExaminerEnv-v0" via openenv.register().
"""
from __future__ import annotations

import json
import random
from typing import Any

# ── OpenEnv / Gymnasium base class resolution ──────────────────────────────
openenv = None  # type: ignore[assignment]
_USE_OPENENV = False
_BaseEnv = None  # type: ignore[assignment]

try:
    from openenv.env import Env as _BaseEnv  # type: ignore
    import openenv  # type: ignore
    _USE_OPENENV = True
except (ImportError, AttributeError):
    pass

if _BaseEnv is None:
    try:
        import gymnasium as gym  # type: ignore
        _BaseEnv = gym.Env  # type: ignore
    except ImportError:
        class _BaseEnv:  # type: ignore[no-redef]
            """Minimal shim when neither openenv nor gymnasium is installed."""
            metadata: dict = {}
            def reset(self, *a, **kw): raise NotImplementedError
            def step(self, *a, **kw): raise NotImplementedError

from examiner_env.action_parser import parse, validate
from examiner_env.knowledge_base import KB, build_kb
from examiner_env.models import (
    AskAction,
    CANONICAL_SECTIONS,
    ClassifyAction,
    EpisodeResult,
    EpisodeState,
    MalformedAction,
    SECTION_TITLES,
)
from examiner_env.posterior_oracle import PosteriorTracker
from examiner_env.reward import compute_reward
from examiner_env.student import generate_response, sample_profile

# ──────────────────────────────────────────────────────────────────────────────
# Action / Observation space descriptors (openenv / gymnasium compatible)
# ──────────────────────────────────────────────────────────────────────────────

try:
    from gymnasium.spaces import Text, Dict as DictSpace  # type: ignore
    _ACTION_SPACE = Text(min_length=2, max_length=4096)
    _OBS_SPACE = DictSpace({})   # observation is untyped dict — varies per turn
except Exception:
    _ACTION_SPACE = None
    _OBS_SPACE = None


class ExaminerEnv(_BaseEnv):
    """
    Single-episode diagnostic environment.

    Episode flow:
      reset(seed) → [step(ask_json), ...] → step(classify_json) → done=True
      OR: turns exhausted → force classify → done=True

    Action: raw JSON string (str) — parsed via action_parser.parse().
    Observation: dict — section_titles, turn, remaining_turns, dialogue_history.
    Reward: 0.0 for Ask steps; R_total (float) on Classify / exhaustion.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        section_ids: list[str] | None = None,
        max_turns: int = 4,
        kb: dict | None = None,
        config: Any = None,      # accepts TrainingConfig or None — C2 compatibility
    ) -> None:
        super().__init__()
        self.section_ids = section_ids or CANONICAL_SECTIONS
        self.max_turns = max_turns
        self.kb = kb or build_kb()

        if _ACTION_SPACE is not None:
            self.action_space = _ACTION_SPACE
            self.observation_space = _OBS_SPACE

        # Episode state — initialised in reset()
        self._state: EpisodeState | None = None
        self._tracker: PosteriorTracker | None = None
        self._profiles: dict = {}

    # ── Public interface ──────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """
        Sample a new episode.  Returns (observation, info).
        """
        if seed is None:
            seed = random.randint(0, 2**31)

        rng = random.Random(seed)

        # Sample KNOWS/FAKING partition
        true_labels: dict[str, str] = {
            s: rng.choice(["KNOWS", "FAKING"]) for s in self.section_ids
        }

        # Sample style per section
        style_assignments: dict[str, str] = {}
        self._profiles = {}
        for idx, s in enumerate(self.section_ids):
            profile = sample_profile(s, true_labels[s], seed, idx)
            self._profiles[s] = profile
            style_assignments[s] = profile.style

        # Initialise state
        self._state = EpisodeState(
            episode_seed=seed,
            section_ids=list(self.section_ids),
            true_labels=true_labels,
            style_assignments=style_assignments,
            max_turns=self.max_turns,
        )

        # Initialise oracle tracker
        self._tracker = PosteriorTracker(self.section_ids)

        obs = self._build_observation()
        info = {"episode_seed": seed}
        return obs, info

    def step(
        self,
        action_text: str,
    ) -> tuple[dict, float, bool, bool, dict]:
        """
        Process one action.

        Returns (observation, reward, terminated, truncated, info).
        reward is 0.0 for Ask steps; R_total for Classify / exhaustion.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        state = self._state
        action = parse(action_text)

        # ── Malformed action ────────────────────────────────────────────────
        if isinstance(action, MalformedAction):
            state.n_malformed += 1
            state.turn += 1
            if state.turn >= state.max_turns:
                return self._force_classify()
            obs = self._build_observation()
            return obs, 0.0, False, False, {"malformed_reason": action.reason}

        # ── Ask action ──────────────────────────────────────────────────────
        if isinstance(action, AskAction):
            val = validate(action, state.section_ids, state.dialogue_history)
            if "P_repetition" in val.penalties:
                state.n_repetition += 1
            if "P_invalid_sec" in val.penalties:
                state.n_invalid_sec += 1

            # Only generate a response if section is valid
            if action.section_id in self.kb:
                profile = self._profiles[action.section_id]
                response = generate_response(
                    question_text=action.question_text,
                    section_id=action.section_id,
                    profile=profile,
                    kb=self.kb,
                    episode_seed=state.episode_seed,
                    turn=state.turn,
                )
                self._tracker.update(action.section_id, response, self.kb)
                self._tracker.snapshot()

                state.dialogue_history.append({
                    "section_id": action.section_id,
                    "question": action.question_text,
                    "response": response,
                    "action_type": "ask",
                })
            else:
                state.n_invalid_sec += 1

            state.turn += 1
            if state.turn >= state.max_turns:
                return self._force_classify()

            obs = self._build_observation()
            return obs, 0.0, False, False, {}

        # ── Classify action ─────────────────────────────────────────────────
        if isinstance(action, ClassifyAction):
            return self._finalise(action.classifications)

        # Should never reach here
        state.n_malformed += 1
        state.turn += 1
        obs = self._build_observation()
        return obs, 0.0, state.turn >= state.max_turns, False, {}

    def render(self, mode: str = "human") -> None:
        if self._state is None:
            print("No active episode.")
            return
        print(f"Turn {self._state.turn}/{self._state.max_turns}")
        for h in self._state.dialogue_history[-3:]:
            print(f"  [{h['section_id']}] Q: {h['question'][:60]}...")
            print(f"          A: {h['response'][:60]}...")

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_observation(self) -> dict:
        """Return observation dict — NEVER includes labels, styles, or posteriors."""
        state = self._state
        return {
            "section_titles": {s: SECTION_TITLES.get(s, s) for s in state.section_ids},
            "section_ids": list(state.section_ids),
            "turn": state.turn,
            "remaining_turns": max(0, state.max_turns - state.turn),
            "dialogue_history": list(state.dialogue_history),
        }

    def _force_classify(self) -> tuple[dict, float, bool, bool, dict]:
        """Force-classify all sections as FAKING (penalty for not classifying)."""
        forced = {s: "FAKING" for s in self._state.section_ids}
        return self._finalise(forced, truncated=True)

    def _finalise(
        self,
        classifications: dict[str, str],
        truncated: bool = False,
    ) -> tuple[dict, float, bool, bool, dict]:
        state = self._state

        episode = EpisodeResult(
            classifications=classifications,
            true_labels=state.true_labels,
            section_ids=state.section_ids,
            turns_used=state.turn,
            max_turns=state.max_turns,
            dialogue_history=state.dialogue_history,
            posterior_tracker=self._tracker,
            n_malformed=state.n_malformed,
            n_repetition=state.n_repetition,
            n_invalid_sec=state.n_invalid_sec,
        )

        breakdown = compute_reward(episode, self.kb)
        state.done = True

        obs = self._build_observation()
        info = breakdown.as_dict()
        # Expose episode ground truth in info (NOT in obs — guardrail compliant)
        info["true_labels"] = dict(state.true_labels)
        info["classifications"] = dict(classifications)
        info["style_assignments"] = dict(state.style_assignments)
        info["reward_breakdown"] = breakdown   # raw dataclass for eval.py
        return obs, breakdown.R_total, True, truncated, info


# ──────────────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────────────

if _USE_OPENENV:
    try:
        openenv.register(
            id="ExaminerEnv-v0",
            entry_point="examiner_env.environment:ExaminerEnv",
            max_episode_steps=10,
        )
    except Exception:
        pass
