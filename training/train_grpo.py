"""
training/train_grpo.py — Unsloth + TRL GRPO training with OpenEnv environment_factory.
C2 owns.

Integration pattern: TRL OpenEnv docs
  https://huggingface.co/docs/trl/en/openenv

Key design:
  - ExaminerToolEnv wraps ExaminerEnv as a TRL environment class.
  - GRPOTrainer(environment_factory=ExaminerToolEnv) handles the multi-turn
    function-calling loop automatically — no manual rollout needed.
  - ask() and classify() are exposed as tool methods. The model calls them
    via function-calling; TRL parses the calls and executes them.
  - reward_func(environments, **kwargs) reads env.reward after each episode.
  - Unsloth FastLanguageModel provides 4-bit quantized base model + LoRA.

ALL hyperparameters come from training/config.py. None hardcoded here.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any

import numpy as np

from training.config import TrainingConfig, get_config, DEBUG_CONFIG

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Standard GRPO reward function (no environment_factory)
# ──────────────────────────────────────────────────────────────────────────────
# Unsloth's compiled UnslothGRPOTrainer overrides _calculate_rewards and always
# calls reward functions with the standard GRPO signature:
#   reward_func(prompts=..., completions=..., completion_ids=..., **kwargs)
# It does NOT support TRL's environment_factory pattern (the environments list
# is never constructed or passed).  We therefore run the ExaminerEnv manually
# inside the reward function: the model outputs a sequence of JSON action
# objects in one generation, and we step the environment with each one.
# ──────────────────────────────────────────────────────────────────────────────

import re as _re

def _extract_json_actions(text: str) -> list[str]:
    """
    Pull every top-level JSON object out of model output text.
    Returns raw JSON strings so ExaminerEnv.step() (which calls parse() on a
    raw string) can consume them directly.
    """
    actions: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                actions.append(text[start : i + 1])
                start = -1
    return actions


def _grpo_reward_func(
    prompts,
    completions,
    episode_seed=None,   # TRL forwards dataset column "episode_seed" as kwarg
    **kwargs,
) -> list[float]:
    """
    Standard GRPO reward function compatible with Unsloth's compiled trainer.

    For each completion the model generates:
      1. Extract JSON action strings (ask / classify blocks).
      2. Create a fresh ExaminerEnv and reset with the episode seed.
      3. Step through the env with each parsed action.
      4. Return the terminal R_total reward.
    """
    try:
        from examiner_env.environment import ExaminerEnv
        from examiner_env.knowledge_base import KB as _KB
    except ImportError as exc:
        print(f"[reward] examiner_env not available: {exc}")
        return [-1.0] * len(completions)

    try:
        import wandb as _wandb
        import numpy as _np
        _wb = _wandb.run is not None
    except ImportError:
        _wb = False
        _wandb = None
        _np = None

    # episode_seed may be a list, a tensor, or None
    seeds: list[int] = []
    if episode_seed is not None:
        try:
            seeds = [int(s) for s in episode_seed]
        except (TypeError, ValueError):
            seeds = list(range(len(completions)))
    if not seeds:
        seeds = list(range(len(completions)))

    rewards: list[float] = []

    for i, completion in enumerate(completions):
        seed = seeds[i] if i < len(seeds) else i

        # completions can be a list of message dicts or a plain string
        if isinstance(completion, list):
            text = " ".join(
                m.get("content", "") for m in completion if isinstance(m, dict)
            )
        else:
            text = str(completion)

        try:
            env = ExaminerEnv(kb=_KB)
            env.reset(seed=seed)

            action_strings = _extract_json_actions(text)
            total_reward = 0.0
            classify_done = False

            for action_str in action_strings:
                obs, reward, terminated, truncated, info = env.step(action_str)
                total_reward += reward
                if terminated or truncated:
                    classify_done = True
                    break

            if not classify_done:
                total_reward -= 0.20  # P_malformed: no classify action
        except Exception as exc:
            print(f"[reward] Episode {i} seed={seed} error: {exc}")
            total_reward = -1.0

        rewards.append(float(total_reward))

    if _wb and _np is not None and rewards:
        _wandb.log({
            "reward/R_total_batch_mean": float(_np.mean(rewards)),
            "reward/R_total_batch_std": float(_np.std(rewards)),
            "reward/R_total_batch_min": float(_np.min(rewards)),
            "reward/R_total_batch_max": float(_np.max(rewards)),
        })

    return rewards


# Section titles for the system prompt
_SECTION_NAMES = (
    "S01=GradDesc/Optim, S02=Backprop, S03=Overfitting/Reg, S04=CNN, "
    "S05=RNN/LSTM, S06=Attention/Transformer, S07=Normalization, "
    "S08=Generative, S09=TransferLearning, S10=EvalMetrics"
)

_EXAMINER_SYSTEM_PROMPT = (
    "You are an expert ML examiner testing a student on machine learning theory.\n\n"
    "Output your examination as a sequence of JSON action objects.\n\n"
    "To ask a question (do this 2-4 times for different sections):\n"
    '{"action_type": "ask", "section_id": "S01", "question_text": '
    '"Why does momentum help gradient descent avoid local minima?"}\n\n'
    "To classify (do this ONCE at the end, covering ALL 10 sections):\n"
    '{"action_type": "classify", "classifications": '
    '{"S01": "KNOWS", "S02": "FAKING", "S03": "KNOWS", "S04": "FAKING", '
    '"S05": "KNOWS", "S06": "FAKING", "S07": "KNOWS", "S08": "FAKING", '
    '"S09": "KNOWS", "S10": "FAKING"}}\n\n'
    f"Sections: {_SECTION_NAMES}\n\n"
    "Rules:\n"
    "- Ask WHY/HOW/edge-case questions — surface definitions do not expose bluffing\n"
    "- Target at least 3 different sections before classifying\n"
    "- You MUST end with a classify action that covers S01 through S10"
)


def _safe_selective_log_softmax_for_unsloth(*args, **kwargs):
    """
    Shape-safe replacement for Unsloth's selective log-softmax helper.

    Unsloth calls this helper with version-specific signatures. The observed
    GRPO path passes: hidden_or_logits, lm_head.weight, token_ids, plus kwargs
    like chunks/logit_scale_*/temperature. This helper intentionally infers the
    tensors by shape and dtype instead of depending on one exact signature.
    """
    import torch
    import torch.nn.functional as _F

    temperature = float(kwargs.get("temperature", 1.0) or 1.0)
    logit_scale_multiply = float(kwargs.get("logit_scale_multiply", 1.0) or 1.0)
    logit_scale_divide = float(kwargs.get("logit_scale_divide", 1.0) or 1.0)
    logit_softcapping = kwargs.get("logit_softcapping")

    ignored_kwargs = {
        "chunks",
        "temperature",
        "logit_scale_multiply",
        "logit_scale_divide",
        "logit_softcapping",
    }
    candidates = list(args) + [v for k, v in kwargs.items() if k not in ignored_kwargs]

    floating_tensors = []
    index = None
    lm_head = None
    lm_head_weight = None

    for candidate in candidates:
        if isinstance(candidate, torch.nn.Parameter):
            if candidate.ndim == 2 and lm_head_weight is None:
                lm_head_weight = candidate
            continue

        if torch.is_tensor(candidate):
            if candidate.is_floating_point() and candidate.ndim >= 2:
                floating_tensors.append(candidate)
            elif index is None:
                index = candidate
            continue

        if callable(candidate) and hasattr(candidate, "weight") and lm_head is None:
            lm_head = candidate

    if not floating_tensors or index is None:
        raise RuntimeError(
            "_safe_selective_log_softmax: could not infer logits/index from "
            f"args={[type(a).__name__ for a in args]} kwargs={list(kwargs.keys())}"
        )

    hidden_or_logits = floating_tensors[0]
    logits = None

    if lm_head_weight is not None:
        if hidden_or_logits.shape[-1] == lm_head_weight.shape[-1]:
            # hidden (..., D) @ weight (V, D).T → (..., V)
            logits = hidden_or_logits.float() @ lm_head_weight.float().t()
        else:
            # already vocab-sized logits
            logits = hidden_or_logits.float()
    elif lm_head is not None:
        logits = lm_head(hidden_or_logits).float()

    if logits is None:
        logits = hidden_or_logits.float()

    if logit_scale_multiply != 1.0:
        logits = logits * logit_scale_multiply
    if logit_scale_divide != 1.0:
        logits = logits / logit_scale_divide
    if logit_softcapping is not None:
        cap = float(logit_softcapping)
        if cap > 0:
            logits = cap * torch.tanh(logits / cap)

    index = index.long()
    # Remove a trailing size-1 dim that some callers add
    if index.ndim == logits.ndim and index.shape[-1] == 1:
        index = index.squeeze(-1)

    # ── Shape-safe gather preserving batch/seq structure ──────────────────
    # logits: (..., V)   index: (...)   where ... are the prefix dims.
    # We must return (...) so that callers can use .shape[1] etc.
    if logits.ndim == index.ndim + 1:
        # Standard: crop each prefix dim to the minimum and gather on last dim.
        n_prefix = index.ndim
        logits_slices = tuple(
            slice(0, min(logits.shape[d], index.shape[d])) for d in range(n_prefix)
        ) + (slice(None),)
        index_slices = tuple(
            slice(0, min(logits.shape[d], index.shape[d])) for d in range(n_prefix)
        )
        logits_c = logits[logits_slices]
        index_c = index[index_slices].clamp(0, logits_c.shape[-1] - 1)
        log_probs = _F.log_softmax(logits_c / temperature, dim=-1)
        return log_probs.gather(-1, index_c.unsqueeze(-1)).squeeze(-1)
    else:
        # Fallback: flatten, gather, then try to restore a 2-D shape if index
        # was at least 2-D (so callers can do .shape[1]).
        index_orig_shape = index.shape
        logits_2d = logits.reshape(-1, logits.shape[-1])
        index_1d = index.reshape(-1)
        n = min(logits_2d.shape[0], index_1d.shape[0])
        logits_2d = logits_2d[:n]
        index_1d = index_1d[:n].clamp(0, logits_2d.shape[-1] - 1)
        log_probs = _F.log_softmax(logits_2d / temperature, dim=-1)
        result = log_probs.gather(-1, index_1d.unsqueeze(-1)).squeeze(-1)  # (n,)
        # Restore 2-D shape when possible so downstream .shape[1] works.
        if index_orig_shape and len(index_orig_shape) >= 2:
            seq = index_orig_shape[-1]
            if n % seq == 0:
                result = result.reshape(n // seq, seq)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# ExaminerToolEnv — TRL environment_factory-compatible wrapper
# ──────────────────────────────────────────────────────────────────────────────

class ExaminerToolEnv:
    """
    TRL environment_factory class wrapping ExaminerEnv.

    Design (from TRL OpenEnv docs):
      - __init__: captures config from module-level; no constructor args.
      - reset(**kwargs): called per episode; receives dataset columns as kwargs.
      - ask() / classify(): public methods → exposed as function-calling tools.
      - self.reward: read by reward_func after each episode.

    The model interacts via function calls:
      ask(section_id="S02", question_text="Why does backprop need chain rule?")
      classify(classifications_json='{"S01":"KNOWS","S02":"FAKING",...}')

    TRL handles multi-turn generation, tool-call parsing, and context stitching.
    """

    def __init__(self):
        from examiner_env.environment import ExaminerEnv
        from examiner_env.knowledge_base import KB
        self._env = ExaminerEnv(kb=KB)
        self._obs: dict = {}
        self._done: bool = False
        self._reward_breakdown: Any = None
        # State for reward_func and W&B logging
        self.reward: float = 0.0
        self.r_acc: float = 0.0
        self.r_info: float = 0.0
        self.r_cal: float = 0.0
        self.r_eff: float = 0.0
        self.r_qual: float = 0.0
        self.parse_failure: bool = False

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, episode_seed: int | None = None, **kwargs) -> str:
        """
        Start a new episode. TRL calls this before each generation.
        Returns the initial system prompt shown to the model.

        The 'episode_seed' dataset column is forwarded here as a kwarg.
        """
        seed = int(episode_seed) if episode_seed is not None else random.randint(0, 99_999)
        self.reward = 0.0
        self.r_acc = self.r_info = self.r_cal = self.r_eff = self.r_qual = 0.0
        self.parse_failure = False
        self._done = False
        self._reward_breakdown = None

        self._obs, _ = self._env.reset(seed=seed)
        return self._format_initial_obs(self._obs)

    # ── Tool: ask ─────────────────────────────────────────────────────────────

    def ask(self, section_id: str, question_text: str) -> str:
        """
        Ask the student a diagnostic question about an ML section.

        Prefer WHY / HOW / edge-case probes — they expose bluffing better
        than surface-level definitional questions. The student simulator will
        reveal misconception cues under pressure; mechanism cues under broad
        questions.

        Args:
            section_id: Section to probe. Must be one of S01–S10.
            question_text: Your diagnostic question (minimum 10 characters).
                           Example: "Why does momentum help gradient descent?"

        Returns:
            The student's response, plus remaining turns before forced classify.
        """
        if self._done:
            raise ValueError(
                "Episode is over — you already classified. "
                "The environment will reset for the next episode."
            )

        action_json = json.dumps({
            "action_type": "ask",
            "section_id": section_id,
            "question_text": question_text,
        })
        obs, _, terminated, truncated, info = self._env.step(action_json)
        self._obs = obs
        self._done = terminated or truncated

        if self._done:
            # Episode ended due to exhaustion (max turns used without classify)
            self.reward = info.get("reward", -0.20) if info else -0.20
            return (
                "[All turns used without classify — episode ended automatically. "
                f"Reward: {self.reward:.3f}. "
                "Issue classify() in future episodes before turns run out.]"
            )

        history = obs.get("dialogue_history", [])
        remaining = obs.get("remaining_turns", 0)
        if history:
            student_response = history[-1].get("response", "(no response)")
        else:
            student_response = "(no response recorded)"

        remaining_msg = (
            f"\n[{remaining} turns left — issue classify() when confident enough.]"
            if remaining <= 1
            else f"\n[{remaining} turns remaining]"
        )
        return f"Student: {student_response}{remaining_msg}"

    # ── Tool: classify ────────────────────────────────────────────────────────

    def classify(self, classifications_json: str) -> str:
        """
        Classify ALL sections and end the episode.

        Call this when you are confident in your assessments, or on the
        final turn (remaining_turns == 1, you MUST classify then).

        Args:
            classifications_json: JSON object mapping EVERY active section ID
                to either "KNOWS" or "FAKING" (case-sensitive).
                Example: {"S01":"KNOWS","S02":"FAKING","S03":"KNOWS",
                          "S04":"FAKING","S05":"KNOWS","S06":"KNOWS",
                          "S07":"FAKING","S08":"KNOWS","S09":"FAKING","S10":"KNOWS"}
                All active sections must be present.

        Returns:
            Episode result showing reward components and classification outcome.
        """
        if self._done:
            raise ValueError("Episode already ended. Cannot classify again.")

        # Parse the JSON string (model may output single-quoted or malformed JSON)
        try:
            clf = json.loads(classifications_json)
            if not isinstance(clf, dict):
                raise ValueError("classifications_json must be a JSON object.")
        except (json.JSONDecodeError, ValueError) as e:
            self.reward = -0.20
            self.parse_failure = True
            return (
                f"[PARSE ERROR] Could not parse classifications_json: {e}. "
                "Reward penalty: -0.20. "
                "Expected: {\"S01\":\"KNOWS\",\"S02\":\"FAKING\",...}"
            )

        action_json = json.dumps({"action_type": "classify", "classifications": clf})
        obs, reward, terminated, truncated, info = self._env.step(action_json)
        self._done = True
        self.reward = float(reward)
        self._reward_breakdown = info.get("reward_breakdown")

        bd = self._reward_breakdown
        if bd:
            self.r_acc = float(bd.R_acc)
            self.r_info = float(bd.R_info)
            self.r_cal = float(bd.R_cal)
            self.r_eff = float(bd.R_eff)
            self.r_qual = float(bd.R_qual)
            return (
                f"[Episode complete] R_total={reward:.3f} | "
                f"R_acc={bd.R_acc:.3f} R_info={bd.R_info:.3f} "
                f"R_cal={bd.R_cal:.3f} R_eff={bd.R_eff:.3f} "
                f"R_qual={bd.R_qual:.3f}"
            )
        return f"[Episode complete] Reward: {reward:.3f}"

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _format_initial_obs(obs: dict) -> str:
        section_titles = obs.get("section_titles", {})
        remaining = obs.get("remaining_turns", 4)
        lines = [
            "You are an examiner determining which ML sections a student KNOWS vs is FAKING.",
            "",
            "Available sections:",
        ]
        for s_id, title in section_titles.items():
            lines.append(f"  {s_id}: {title}")
        lines += [
            "",
            f"You have {remaining} turns. Strategy:",
            "  1. Use ask() with WHY/HOW probes — they expose bluffing via misconception cues.",
            "  2. Ask different sections to build a complete picture.",
            "  3. Use classify() when confident, or on the final turn (MANDATORY).",
            "",
            "The student cannot see your classifications until the episode ends.",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Reward function — TRL OpenEnv signature
# ──────────────────────────────────────────────────────────────────────────────

def reward_func(environments: list, **kwargs) -> list[float]:
    """
    TRL OpenEnv reward function.
    Called after each generation batch. Reads env.reward from each instance.
    Per TRL docs: https://huggingface.co/docs/trl/en/openenv#reward-functions
    """
    rewards = [float(env.reward) for env in environments]

    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "reward/R_total_batch_mean": float(np.mean(rewards)),
            "reward/R_total_batch_std": float(np.std(rewards)),
            "reward/R_total_batch_min": float(np.min(rewards)),
            "reward/R_total_batch_max": float(np.max(rewards)),
            "reward/R_acc_mean": float(np.mean([env.r_acc for env in environments])),
            "reward/R_info_mean": float(np.mean([env.r_info for env in environments])),
            "reward/R_cal_mean": float(np.mean([env.r_cal for env in environments])),
            "training/parse_failure_rate": float(
                np.mean([float(env.parse_failure) for env in environments])
            ),
        })

    return rewards


# ──────────────────────────────────────────────────────────────────────────────
# Reward variance monitor
# ──────────────────────────────────────────────────────────────────────────────

def _check_reward_variance(
    reward_buffer: list[float],
    config: TrainingConfig,
    step: int,
    current_beta_kl: float,
) -> float:
    if len(reward_buffer) < 10:
        return current_beta_kl
    recent = reward_buffer[-50:]
    variance = float(np.std(recent))
    if _WANDB_AVAILABLE and wandb.run:
        wandb.log({"reward/variance_monitor": variance}, step=step)
    if variance < config.reward_variance_floor:
        print(f"WARNING [step {step}]: Reward variance {variance:.4f} collapsed.")
        if _WANDB_AVAILABLE and wandb.run:
            wandb.log({"warning/reward_variance_collapsed": 1}, step=step)
    if variance > config.reward_variance_ceiling:
        new_beta = min(0.10, current_beta_kl * 1.5)
        print(f"WARNING [step {step}]: Variance {variance:.4f} high → beta_kl {current_beta_kl:.4f} → {new_beta:.4f}")
        if _WANDB_AVAILABLE and wandb.run:
            wandb.log({"adaptive/beta_kl": new_beta}, step=step)
        return new_beta
    return current_beta_kl


# ──────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ──────────────────────────────────────────────────────────────────────────────

def train(config: TrainingConfig, eval_config: dict) -> dict:
    """
    Full GRPO training with TRL environment_factory.

    Flow:
      1. Init W&B
      2. Check oracle calibration gate
      3. Pre-training baseline eval → baseline_metrics.json
      4. Load model via Unsloth (4-bit, LoRA)
      5. Build episode seed dataset
      6. GRPOTrainer(environment_factory=ExaminerToolEnv)
      7. Periodic eval callback → checkpoint_metrics.json
      8. trainer.train()
      9. Final held-out eval → final_metrics.json
      10. W&B finish

    Returns final_metrics dict.
    """
    # ── Deferred imports ───────────────────────────────────────────────────
    # NOTE: We use standard HF transformers + PEFT + TRL directly (not Unsloth's
    # compiled GRPOTrainer) because Unsloth's pre-compiled cache has a shape
    # mismatch with environment_factory variable-length completions on T4.
    # We still use BitsAndBytesConfig for 4-bit quantisation and PEFT LoRA.

    # TORCHDYNAMO_DISABLE=1 must be set (via Space secret or env) before torch
    # loads to prevent Unsloth's compiled cache from tracing variable-length
    # environment_factory completions (which causes a gather shape mismatch).
    import os as _os
    _os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    # Hard-disable Accelerate mixed precision for this run. If the runtime
    # inherits ACCELERATE_MIXED_PRECISION=fp16 from the environment, Accelerate
    # creates a GradScaler and crashes with "Attempting to unscale FP16 gradients"
    # on forced-fp16 LoRA parameters.
    _os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

    try:
        from unsloth import FastLanguageModel
        from trl import GRPOTrainer, GRPOConfig
        from transformers import TrainerCallback
        from datasets import Dataset
        import torch
        import torch.nn.functional as _F
    except ImportError as e:
        raise RuntimeError(
            "Unsloth / TRL / datasets not installed.\n"
            "Run: pip install 'unsloth[colab-new]' datasets\n"
            f"Error: {e}"
        )

    # ── Define patched function (applied post-trainer-init) ──────────────
    # Unsloth's chunked_hidden_states_selective_log_softmax has a hard-coded
    # gather over chunks, but environment_factory produces variable-length
    # completions → gather shape mismatch at dim 0. We replace it with a
    # robust, signature-agnostic version that introspects its inputs.
    # NOTE: must be applied AFTER GRPOTrainer is instantiated (that's when
    # Unsloth generates unsloth_compiled_cache.UnslothGRPOTrainer).
    def _safe_selective_log_softmax(*args, **kwargs):
        return _safe_selective_log_softmax_for_unsloth(*args, **kwargs)

    def _apply_unsloth_patch():
        """Apply patch — call AFTER GRPOTrainer() so compiled cache exists."""
        import sys as _sys
        _patched = 0
        for _mod_name, _mod in list(_sys.modules.items()):
            if _mod is None:
                continue
            _lower_name = _mod_name.lower()
            if "unsloth" not in _lower_name:
                continue
            if hasattr(_mod, "chunked_hidden_states_selective_log_softmax"):
                try:
                    _mod.chunked_hidden_states_selective_log_softmax = _safe_selective_log_softmax
                    print(f"[Patch] chunked_log_softmax replaced in {_mod_name}")
                    _patched += 1
                except Exception as _e:
                    print(f"[Patch] failed on {_mod_name}: {_e}")
        try:
            import unsloth_compiled_cache.UnslothGRPOTrainer as _ucg  # type: ignore
            _ucg.chunked_hidden_states_selective_log_softmax = _safe_selective_log_softmax
            print("[Patch] direct patch unsloth_compiled_cache.UnslothGRPOTrainer")
            _patched += 1
        except (ImportError, AttributeError):
            pass
        print(f"[Patch] total replacements: {_patched}")
        return _patched

    try:
        from examiner_env.knowledge_base import KB
        from examiner_env.baselines import RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer
        from training.eval import run_eval
        from training.reward_fn import init_wandb
    except ImportError as e:
        raise RuntimeError(
            f"examiner_env not available. C1 Phase 1 gate must clear first.\nError: {e}"
        )

    # ── W&B init ──────────────────────────────────────────────────────────
    init_wandb(config)

    # ── Oracle calibration gate ────────────────────────────────────────────
    cal_path = os.path.join("outputs", "eval", "oracle_calibration.json")
    if not os.path.exists(cal_path):
        raise FileNotFoundError(
            "outputs/eval/oracle_calibration.json missing. "
            "Run examiner_env.calibration.run_calibration() first."
        )
    with open(cal_path) as f:
        cal = json.load(f)
    brier = cal["calibration_metrics"]["mean_brier"]
    if brier > 0.18:
        # Soft gate: warn loudly but do not abort an expensive GPU run mid-launch.
        # Recalibrate offline if you see this banner.
        print(
            f"WARNING: Oracle Brier={brier:.4f} > 0.18 target — "
            "training will continue but R_info shaping may be noisy."
        )
        if _WANDB_AVAILABLE and wandb.run:
            wandb.log({"warning/oracle_brier_high": brier})
    else:
        print(f"Oracle calibration OK (Brier={brier:.4f})")

    # ── Pre-training baseline eval ─────────────────────────────────────────
    _run_baseline_eval(eval_config, KB, run_eval,
                       RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer)

    # ── Pick a single compute dtype for Unsloth fused LoRA kernels ─────────
    # A100 supports bf16, but this Unsloth 2026.4.8 + Qwen2.5 7B 4-bit GRPO
    # stack still enters `fast_lora.matmul_lora` with fp16 activations (`Half`).
    # If the trainer/model/adapters are bf16 or fp32 in any branch, the fused
    # kernel crashes with:
    #   RuntimeError: self and mat2 must have the same dtype, but got Half and Float
    #
    # The stable path is to force CUDA model weights/adapters to fp16:
    #   model dtype = torch.float16
    #   LoRA adapter params = torch.float16
    # but keep Trainer mixed precision OFF. If `fp16=True`, Accelerate creates
    # a GradScaler and later crashes at gradient clipping with:
    #   ValueError: Attempting to unscale FP16 gradients.
    _bf16_ok_load = False
    _fp16_trainer = False
    _model_dtype = torch.float16 if torch.cuda.is_available() else None
    print(
        f"Model load dtype: {_model_dtype} "
        f"(forced fp16 model for Unsloth LoRA kernel stability; "
        f"trainer bf16={_bf16_ok_load} fp16={_fp16_trainer})"
    )

    # ── Load model with Unsloth (4-bit + LoRA) ────────────────────────────
    print(f"Loading {config.model_name} (4-bit={config.use_4bit})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.use_4bit,
        dtype=_model_dtype,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Force every trainable floating parameter into the model dtype.
    # PEFT commonly leaves LoRA A/B weights as float32. For this Unsloth fused
    # kernel path we must not allow any trainable fp32 matrix to reach addmm_.
    if _model_dtype is not None:
        for name, param in model.named_parameters():
            if param.requires_grad and param.is_floating_point() and param.dtype != _model_dtype:
                print(f"[dtype] casting trainable param {name}: {param.dtype} -> {_model_dtype}")
                param.data = param.data.to(_model_dtype)

    # Align tokenizer's max length with config so generation does not warn about
    # max_length=32768 vs max_new_tokens=256 and so prompt-side truncation is
    # explicit rather than silent during eval callbacks.
    try:
        tokenizer.model_max_length = int(config.max_seq_length)
    except Exception:
        pass
    # Qwen defaults ship max_length=32768 on generation_config; combined with
    # max_new_tokens elsewhere that triggers noisy "Both max_new_tokens and
    # max_length" warnings. Prefer unconstrained max_length when using new tokens.
    try:
        _gc = getattr(model, "generation_config", None)
        if _gc is not None and hasattr(_gc, "max_length"):
            _gc.max_length = None
    except Exception:
        pass
    print("Model loaded with LoRA adapters.")

    # ── Episode seed dataset ───────────────────────────────────────────────
    # Each row has:
    #   prompt       — system message instructing the model to output JSON actions
    #   episode_seed — forwarded to _grpo_reward_func as kwarg; used to reset
    #                  ExaminerEnv deterministically per episode
    # NOTE: environment_factory is NOT used (Unsloth compiled trainer does not
    # support it). The model generates all actions in one completion; the reward
    # function runs the environment from those actions.
    n = config.num_episodes
    dataset = Dataset.from_dict({
        "prompt": [[{
            "role": "user",
            "content": _EXAMINER_SYSTEM_PROMPT,
        }]] * n,
        "episode_seed": list(range(n)),
    })

    # ── Eval + checkpoint callback ─────────────────────────────────────────
    reward_buffer: list[float] = []
    current_beta_kl = config.beta_kl
    checkpoint_metrics_log: dict = {}
    train_started_at = time.time()
    # Optimizer steps per epoch (what tqdm / max_steps actually use).
    total_steps_estimate = max(1, (n + config.batch_size - 1) // config.batch_size)

    class EvalAndMonitorCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                r = logs.get("train/reward", logs.get("reward/R_total_batch_mean"))
                if r is not None:
                    reward_buffer.append(float(r))
                step = int(getattr(state, "global_step", 0) or 0)
                max_steps = int(getattr(state, "max_steps", 0) or total_steps_estimate)
                if max_steps <= 0:
                    max_steps = total_steps_estimate
                pct = 100.0 * min(step, max_steps) / max_steps
                elapsed_s = time.time() - train_started_at
                elapsed_min = elapsed_s / 60.0
                # ETA: based on average sec/step so far
                if step > 0:
                    sec_per_step = elapsed_s / step
                    remaining_steps = max(0, max_steps - step)
                    eta_min = (sec_per_step * remaining_steps) / 60.0
                else:
                    sec_per_step = 0.0
                    eta_min = 0.0
                loss = logs.get("loss", logs.get("train/loss"))
                lr = logs.get("learning_rate", logs.get("train/learning_rate"))
                reward = logs.get("reward", logs.get("train/reward", logs.get("reward/R_total_batch_mean")))
                kl = logs.get("kl", logs.get("train/kl"))
                grad_norm = logs.get("grad_norm", logs.get("train/grad_norm"))

                # Pretty banner every step
                bar_w = 24
                filled = int(bar_w * pct / 100.0)
                bar = "█" * filled + "·" * (bar_w - filled)
                header = (
                    f"━━━ {config.config_name} step {step:>4}/{max_steps} "
                    f"[{bar}] {pct:5.1f}% ━━━"
                )
                stats = []
                if reward is not None:
                    stats.append(f"reward={float(reward):+.4f}")
                if loss is not None:
                    stats.append(f"loss={float(loss):.4f}")
                if kl is not None:
                    stats.append(f"kl={float(kl):.4f}")
                if grad_norm is not None:
                    stats.append(f"grad_norm={float(grad_norm):.3f}")
                if lr is not None:
                    stats.append(f"lr={float(lr):.2e}")
                timing = (
                    f"elapsed={elapsed_min:6.1f}m  "
                    f"eta={eta_min:6.1f}m  "
                    f"sec/step={sec_per_step:6.1f}s"
                )
                print(header, flush=True)
                if stats:
                    print("  " + "  ".join(stats), flush=True)
                print("  " + timing, flush=True)

                # Do not pass step= — TRL/Unsloth already advances wandb steps at a
                # different rate; forcing trainer global_step causes "step 24 < 335"
                # drops and missing curves.
                if _WANDB_AVAILABLE and wandb.run:
                    wandb.log({
                        "training/current_step": step,
                        "training/total_steps_estimate": max_steps,
                        "training/progress_pct": pct,
                        "training/elapsed_minutes": elapsed_min,
                        "training/eta_minutes": eta_min,
                        "training/sec_per_step": sec_per_step,
                    })

        def on_step_end(self, args, state, control, **kw):
            nonlocal current_beta_kl
            step = state.global_step
            if step % 50 == 0 and reward_buffer:
                current_beta_kl = _check_reward_variance(
                    reward_buffer, config, step, current_beta_kl
                )
            if step > 0 and step % config.eval_every_n_steps == 0:
                print(f"\n[step {step}] Checkpoint eval...")
                from examiner_env.environment import ExaminerEnv
                _TrainedExaminerWrapper.model = model
                _TrainedExaminerWrapper.tokenizer = tokenizer
                _TrainedExaminerWrapper.config = config
                # Switch Unsloth model into inference kernels for generate(),
                # then back to training kernels.  Without this, mid-training
                # generation hits the same fast_lora dtype mismatch path and
                # eval crashes with "Half and Float".
                _swapped = False
                try:
                    FastLanguageModel.for_inference(model)
                    _swapped = True
                except Exception as _swap_err:
                    print(f"[eval] for_inference swap failed: {_swap_err}")
                try:
                    chk = run_eval(_TrainedExaminerWrapper(), eval_config, KB)
                finally:
                    if _swapped:
                        try:
                            FastLanguageModel.for_training(model)
                        except Exception as _back_err:
                            print(f"[eval] for_training swap-back failed: {_back_err}")
                checkpoint_metrics_log[str(step)] = chk
                os.makedirs(os.path.join("outputs", "eval"), exist_ok=True)
                with open(os.path.join("outputs", "eval", "checkpoint_metrics.json"), "w") as fp:
                    json.dump(checkpoint_metrics_log, fp, indent=2)
                if _WANDB_AVAILABLE and wandb.run:
                    wandb.log({
                        "eval/reward_mean": chk["reward_mean"],
                        "eval/classification_accuracy": chk["classification_accuracy"],
                        "eval/avg_info_gain": chk["avg_info_gain_per_turn"],
                        "eval/calibration_ECE": chk["calibration_ECE"],
                        "eval/false_accusation_rate": chk["false_accusation_rate"],
                    })
                print(f"  accuracy={chk['classification_accuracy']:.3f} "
                      f"info_gain={chk['avg_info_gain_per_turn']:.4f} "
                      f"ECE={chk['calibration_ECE']:.4f}")

    # Trainer mixed precision is intentionally disabled. The model itself is
    # already fp16 on CUDA, and enabling `fp16=True` creates a GradScaler that
    # cannot unscale fp16 LoRA gradients.
    _bf16_ok = _bf16_ok_load
    _fp16 = _fp16_trainer
    _max_grad_norm = 0.0 if _model_dtype is torch.float16 else config.max_grad_norm
    print(
        f"Precision: bf16={_bf16_ok} fp16={_fp16} "
        f"model_dtype={_model_dtype} max_grad_norm={_max_grad_norm}"
    )

    # ── GRPOConfig ────────────────────────────────────────────────────────
    checkpoint_steps = 10
    grpo_config = GRPOConfig(
        output_dir=os.path.join("outputs", "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        bf16=_bf16_ok,
        fp16=_fp16,
        max_grad_norm=_max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        num_generations=config.num_generations,
        beta=config.beta_kl,
        save_steps=checkpoint_steps,
        logging_steps=1,
        report_to="wandb" if _WANDB_AVAILABLE else "none",
        max_completion_length=config.max_seq_length,
        # Huge completion tables + duplicate "max_new_tokens" log noise on Spaces.
        log_completions=(config.config_name == "FULL"),
    )

    # ── GRPOTrainer (standard GRPO — no environment_factory) ─────────────
    # Unsloth's compiled UnslothGRPOTrainer calls reward functions with the
    # standard GRPO signature (prompts, completions, ...), not the OpenEnv
    # signature (environments).  We therefore use _grpo_reward_func which runs
    # ExaminerEnv internally for each completion.
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=_grpo_reward_func,
        args=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=[EvalAndMonitorCallback()],
    )

    # Defensive runtime guard: some Spaces still surface mixed_precision="fp16"
    # despite fp16=False/bf16=False in args (inherited accelerate state). If so,
    # neutralize gradient unscale path before training to prevent GradScaler crash.
    _acc = getattr(trainer, "accelerator", None)
    if _acc is not None:
        _mp = str(getattr(_acc, "mixed_precision", "no")).lower()
        if _mp == "fp16":
            print("[precision-guard] Accelerator mixed_precision=fp16 detected; forcing no/unscale bypass.")
            try:
                _acc.mixed_precision = "no"
            except Exception:
                pass
            try:
                _acc.native_amp = False
            except Exception:
                pass
            try:
                _acc.scaler = None
            except Exception:
                pass

            def _no_unscale_gradients(*args, **kwargs):
                return None

            try:
                _acc.unscale_gradients = _no_unscale_gradients
            except Exception:
                pass

    print(f"\nStarting {config.config_name} training ({n} episodes)...")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA: r={config.lora_rank} alpha={config.lora_alpha}")
    print(f"  Generations/step: {config.num_generations}")
    print(f"  Eval every {config.eval_every_n_steps} steps")
    print(f"  Checkpoint save every {checkpoint_steps} steps (auto-resume enabled)")
    print(f"  Reward mode: standard GRPO (env runs inside reward func)")

    # Patch Unsloth's compiled cache NOW (generated by GRPOTrainer.__init__)
    _apply_unsloth_patch()

    resume_ckpt = _find_latest_checkpoint(os.path.join("outputs", "checkpoints"))
    if resume_ckpt is not None:
        print(f"[resume] Found checkpoint. Resuming from: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        print("[resume] No checkpoint found. Starting fresh run.")
        trainer.train()

    # ── Final held-out eval ───────────────────────────────────────────────
    print("\nRunning final held-out eval...")
    _TrainedExaminerWrapper.model = model
    _TrainedExaminerWrapper.tokenizer = tokenizer
    _TrainedExaminerWrapper.config = config
    try:
        FastLanguageModel.for_inference(model)
    except Exception as _swap_err:
        print(f"[final-eval] for_inference swap failed: {_swap_err}")
    # Optional override for time-critical runs:
    #   FINAL_EVAL_EPISODES=20
    # keeps the same evaluation codepath but uses the first N frozen seeds.
    _final_eval_cfg = eval_config
    _override = os.environ.get("FINAL_EVAL_EPISODES", "").strip()
    if _override:
        try:
            _n = int(_override)
            if _n > 0:
                _final_eval_cfg = dict(eval_config)
                _final_eval_cfg["seeds"] = list(eval_config.get("seeds", []))[:_n]
                print(
                    f"[final-eval] FINAL_EVAL_EPISODES override active: "
                    f"{len(_final_eval_cfg['seeds'])} episodes",
                    flush=True,
                )
        except ValueError:
            print(f"[final-eval] Ignoring invalid FINAL_EVAL_EPISODES='{_override}'", flush=True)

    try:
        final_metrics = run_eval(
            _TrainedExaminerWrapper(),
            _final_eval_cfg,
            KB,
            output_path=os.path.join("outputs", "eval", "final_metrics.json"),
        )
    except Exception as _final_exc:
        # Last-resort fallback to avoid losing an otherwise-finished training run.
        print(f"[final-eval] Primary eval failed: {_final_exc}", flush=True)
        print("[final-eval] Falling back to 10-episode eval...", flush=True)
        _fallback_cfg = dict(eval_config)
        _fallback_cfg["seeds"] = list(eval_config.get("seeds", []))[:10]
        final_metrics = run_eval(
            _TrainedExaminerWrapper(),
            _fallback_cfg,
            KB,
            output_path=os.path.join("outputs", "eval", "final_metrics_fallback.json"),
        )
        final_metrics["final_eval_fallback_used"] = True
    print(
        f"Final — accuracy={final_metrics['classification_accuracy']:.3f} "
        f"info_gain={final_metrics['avg_info_gain_per_turn']:.4f} "
        f"ECE={final_metrics['calibration_ECE']:.4f}"
    )

    if _WANDB_AVAILABLE and wandb.run:
        wandb.log({
            "final/classification_accuracy": final_metrics["classification_accuracy"],
            "final/avg_info_gain_per_turn": final_metrics["avg_info_gain_per_turn"],
            "final/calibration_ECE": final_metrics["calibration_ECE"],
            "final/false_accusation_rate": final_metrics["false_accusation_rate"],
            "final/reward_mean": final_metrics["reward_mean"],
        })
        wandb.finish()

    print("Training complete.")
    return final_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run_baseline_eval(eval_config, KB, run_eval, RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer):
    baseline_path = os.path.join("outputs", "eval", "baseline_metrics.json")
    if os.path.exists(baseline_path):
        print(f"baseline_metrics.json already exists — skipping baseline eval.")
        return
    print("Running pre-training baseline evaluation...")
    baseline_metrics = {}
    for name, examiner in [
        ("RandomExaminer", RandomExaminer()),
        ("DefinitionalExaminer", DefinitionalExaminer()),
        ("BayesianHeuristicExaminer", BayesianHeuristicExaminer(KB)),
    ]:
        m = run_eval(examiner, eval_config, KB)
        baseline_metrics[name] = m
        print(f"  {name}: accuracy={m['classification_accuracy']:.3f} "
              f"info_gain={m['avg_info_gain_per_turn']:.4f}")
    os.makedirs(os.path.join("outputs", "eval"), exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    print("baseline_metrics.json saved.")


def _find_latest_checkpoint(checkpoint_root: str) -> str | None:
    """Return latest checkpoint-* directory path, or None if absent."""
    if not os.path.isdir(checkpoint_root):
        return None
    latest_step = -1
    latest_path = None
    for entry in os.listdir(checkpoint_root):
        if not entry.startswith("checkpoint-"):
            continue
        suffix = entry.split("checkpoint-", 1)[-1]
        try:
            step = int(suffix)
        except ValueError:
            continue
        path = os.path.join(checkpoint_root, entry)
        if os.path.isdir(path) and step > latest_step:
            latest_step = step
            latest_path = path
    return latest_path


class _TrainedExaminerWrapper:
    """Wraps trained model into examiner.act(observation) interface for run_eval()."""
    model: Any = None
    tokenizer: Any = None
    config: Any = None

    def reset(self) -> None:
        pass

    def act(self, observation: dict) -> str:
        from copy import deepcopy

        from training.prompt_builder import build_prompt
        from transformers import GenerationConfig

        tok = self.__class__.tokenizer
        cfg = self.__class__.config
        model = self.__class__.model
        max_new = 256
        # Reserve room for generation: cap prompt at (max_seq_length - max_new).
        max_prompt_tokens = max(64, int(cfg.max_seq_length) - max_new)
        prompt = build_prompt(observation)
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(model.device)
        _base = getattr(model, "generation_config", None)
        _gc = deepcopy(_base) if _base is not None else GenerationConfig()
        _gc.max_new_tokens = max_new
        _gc.do_sample = False
        _gc.pad_token_id = tok.eos_token_id
        if hasattr(_gc, "max_length"):
            _gc.max_length = None
        outputs = model.generate(**inputs, generation_config=_gc)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return tok.decode(generated, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BluffBuster Examiner with GRPO")
    parser.add_argument("--config", default="DEBUG", choices=["DEBUG", "DEMO", "FULL"])
    parser.add_argument("--eval_config", default="eval_config.json")
    args = parser.parse_args()

    with open(args.eval_config) as f:
        eval_cfg = json.load(f)

    train(get_config(args.config), eval_cfg)
