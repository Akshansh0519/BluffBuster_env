"""
training/reward_fn.py — GRPO reward functions.
C2 owns.

Two reward function interfaces are provided:

1. reward_func(environments, **kwargs) — TRL OpenEnv pattern (PRIMARY).
   Used with GRPOTrainer(environment_factory=ExaminerToolEnv).
   Reads env.reward from each ExaminerToolEnv instance after the episode.
   Per TRL docs: https://huggingface.co/docs/trl/en/openenv#reward-functions

2. reward_fn(completions, prompts, **kwargs) — legacy bridge (kept for compat).
   Used only if falling back to a non-environment-factory training loop.
   DELEGATES to examiner_env.reward.compute_reward(). Never re-implements logic.

Also contains: log_reward_breakdown() and init_wandb() helpers.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np

# Deferred import — examiner_env built by C1
try:
    from examiner_env.reward import compute_reward, RewardBreakdown
    from examiner_env.action_parser import parse
    from examiner_env.models import MalformedAction
except ImportError:
    compute_reward = None  # type: ignore[assignment]
    RewardBreakdown = None  # type: ignore[assignment]
    parse = None  # type: ignore[assignment]
    MalformedAction = None  # type: ignore[assignment]

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

_global_step = 0


# ---------------------------------------------------------------------------
# TRL OpenEnv reward function (PRIMARY — used with environment_factory)
# ---------------------------------------------------------------------------

def reward_func(environments: list, **kwargs) -> list[float]:
    """
    TRL OpenEnv reward function signature.
    Called after each generation batch by GRPOTrainer when environment_factory
    is used. Reads env.reward from each ExaminerToolEnv instance.

    Per TRL docs pattern (Wordle/Echo/Sudoku examples):
      def reward_func(environments, **kwargs):
          return [env.reward for env in environments]

    This version also logs per-component rewards to W&B.
    """
    rewards = [float(env.reward) for env in environments]

    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "reward/R_total_batch_mean": float(np.mean(rewards)),
            "reward/R_total_batch_std": float(np.std(rewards)),
            "reward/R_total_batch_min": float(np.min(rewards)),
            "reward/R_total_batch_max": float(np.max(rewards)),
            "reward/R_acc_mean": float(np.mean([getattr(env, "r_acc", 0.0) for env in environments])),
            "reward/R_info_mean": float(np.mean([getattr(env, "r_info", 0.0) for env in environments])),
            "reward/R_cal_mean": float(np.mean([getattr(env, "r_cal", 0.0) for env in environments])),
            "reward/R_eff_mean": float(np.mean([getattr(env, "r_eff", 0.0) for env in environments])),
            "reward/R_qual_mean": float(np.mean([getattr(env, "r_qual", 0.0) for env in environments])),
            "training/parse_failure_rate": float(
                np.mean([float(getattr(env, "parse_failure", False)) for env in environments])
            ),
        })

    return rewards


# ---------------------------------------------------------------------------
# W&B reward breakdown logger — called every training step
# ---------------------------------------------------------------------------

def log_reward_breakdown(breakdown: Any, step: int) -> None:
    """
    Log all 11 per-component rewards to W&B.
    Must be called with a real RewardBreakdown object, not a mock.
    Raises if W&B is not initialized.
    """
    if not _WANDB_AVAILABLE:
        return
    wandb.log(
        {
            "reward/R_total": breakdown.R_total,
            "reward/R_acc": breakdown.R_acc,
            "reward/R_asym": breakdown.R_asym,
            "reward/R_cal": breakdown.R_cal,
            "reward/R_eff": breakdown.R_eff,
            "reward/R_cov": breakdown.R_cov,
            "reward/R_info": breakdown.R_info,
            "reward/R_qual": breakdown.R_qual,
            "reward/R_div": breakdown.R_div,
            "reward/P_malformed": breakdown.P_malformed,
            "reward/P_repetition": breakdown.P_repetition,
            "reward/P_invalid_sec": breakdown.P_invalid_sec,
        },
        step=step,
    )


# ---------------------------------------------------------------------------
# GRPO reward function — TRL GRPOTrainer interface
# ---------------------------------------------------------------------------

def reward_fn(
    completions: list[str],
    prompts: list[str],
    **kwargs: Any,
) -> list[float]:
    """
    TRL GRPOTrainer reward function interface.

    For each completion:
      1. parse(completion) → action
      2. If MalformedAction → -0.20 (P_malformed for 1 malformed action)
      3. Else → delegate to compute_reward() via episode_result in kwargs

    NEVER defines reward logic here. Always delegates to examiner_env.reward.
    """
    global _global_step

    if compute_reward is None or parse is None:
        raise RuntimeError(
            "examiner_env not available — ensure C1 Phase 1 gate has cleared "
            "before running training."
        )

    rewards: list[float] = []
    n_malformed = 0
    episode_results: dict = kwargs.get("episode_results", {})

    for completion in completions:
        action = parse(completion)

        if isinstance(action, MalformedAction):
            rewards.append(-0.20)
            n_malformed += 1
            continue

        # Retrieve episode result injected by the training loop
        ep_result = episode_results.get(id(completion))
        kb = kwargs.get("kb")

        if ep_result is None or kb is None:
            # Fallback: malformed penalty when episode context unavailable
            rewards.append(-0.20)
            n_malformed += 1
            continue

        breakdown = compute_reward(ep_result, kb)
        rewards.append(breakdown.R_total)

        # Log per-component breakdown for this completion
        log_reward_breakdown(breakdown, step=_global_step)

    # Batch-level W&B logging
    if _WANDB_AVAILABLE and rewards:
        wandb.log(
            {
                "reward/R_total_batch_mean": float(np.mean(rewards)),
                "reward/R_total_batch_std": float(np.std(rewards)),
                "training/parse_failure_rate": n_malformed / len(completions),
            },
            step=_global_step,
        )

    _global_step += 1
    return rewards


# ---------------------------------------------------------------------------
# W&B initialization helper (used by train_grpo.py)
# ---------------------------------------------------------------------------

def init_wandb(config: Any) -> None:
    """Initialize W&B run. config must be a TrainingConfig dataclass."""
    if not _WANDB_AVAILABLE:
        print("WARNING: wandb not installed. Metrics will not be logged.")
        return

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "WANDB_API_KEY not set. Set it as an env variable or Colab secret before training."
        )

    wandb.init(
        project="bluffbuster-examiner",
        name=f"run-{config.config_name}-{datetime.now().strftime('%m%d-%H%M')}",
        config=vars(config),
        tags=[config.config_name, "grpo", "bluffbuster"],
    )
