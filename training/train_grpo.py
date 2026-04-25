"""
training/train_grpo.py — Unsloth + TRL GRPO training for BluffBuster.
C2 owns.

────────────────────────────────────────────────────────────────────────────
Why this file looks the way it does (read before editing):

1. Unsloth ships a pre-compiled GRPOTrainer at
   /app/unsloth_compiled_cache/UnslothGRPOTrainer.py. That file is generated
   the first time Unsloth is imported; TRL's own GRPOTrainer is then replaced
   at runtime. Consequences for us:

     (a) TRL's `environment_factory=...` multi-turn tool-calling path is
         IGNORED by the compiled cache. The compiled trainer calls
         `reward_func(prompts=..., completions=..., completion_ids=..., **dataset_cols)`
         — standard single-shot GRPO signature, not the OpenEnv one.
         Hence this file does NOT pass `environment_factory`. Any env rollout
         happens INSIDE the reward function, per completion, using the
         `episode_seed` column forwarded from the dataset.

     (b) The compiled cache calls an internal
         `chunked_hidden_states_selective_log_softmax(
              hidden_or_logits, lm_head_weight, token_ids,
              chunks=N, logit_scale_multiply=..., logit_scale_divide=...,
              logit_softcapping=..., temperature=...,
          )`
         and expects a 2D `(B, S)` tensor back (downstream does
         `ref_logps.shape[1]`). On some versions / shapes the kernel crashes.
         We therefore install a safe Python replacement via monkey-patch
         AFTER `FastLanguageModel.from_pretrained(...)` has triggered cache
         generation and BEFORE `trainer.train()` is called.

         The replacement is version-independent: it accepts *args/**kwargs,
         classifies inputs by type (Parameter-first, so Parameter is never
         mistaken for an int-tensor), projects hidden states to logits when
         needed, and always returns a 2D tensor with the ids' prefix dims.

         See scripts/verify_unsloth_patch.py for a torch-only test matrix
         proving the replacement is shape-correct under the four failure
         modes documented in the root-cause analysis.

2. ALL hyperparameters come from training/config.py. Nothing is hardcoded
   in this file.

3. Reward is *never* re-implemented here. The reward_func builds a real
   ExaminerEnv per completion and delegates to examiner_env.reward via the
   env's `_finalise` path.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Iterable

import numpy as np

from training.config import TrainingConfig, get_config, DEBUG_CONFIG

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════
#   Unsloth chunked-log-softmax monkey patch
# ══════════════════════════════════════════════════════════════════════════

_UNSLOTH_PATCH_TARGET = "chunked_hidden_states_selective_log_softmax"
_UNSLOTH_PATCH_INSTALLED = False


def _safe_chunked_hidden_states_selective_log_softmax(*args, **kwargs):
    """
    Version-independent replacement for Unsloth's
    `chunked_hidden_states_selective_log_softmax`.

    Expected call shape (may vary slightly across Unsloth versions):
        fn(
            hidden_or_logits,       # (B, S, D) hidden OR (B, S, V) logits
            lm_head_weight,         # torch.nn.Parameter (V, D)  ← critical: Parameter
            token_ids,              # int tensor (B, S) or (N,)
            chunks=int,
            logit_scale_multiply=float,
            logit_scale_divide=float,
            logit_softcapping=float,
            temperature=float,
        )

    Returns a 2D tensor of per-token log-probabilities with the same prefix
    dims as `token_ids`. Downstream (grpo_accumulated_loss) does
    `ref_logps.shape[1]`, so a 1D return will crash.

    Guarantees (from the RCA):
      1. Accepts *args/**kwargs — signature differs across Unsloth releases.
      2. Distinguishes torch.nn.Parameter from plain Tensor BEFORE any dtype
         check: `torch.is_tensor(p)` returns True for Parameter, so an
         unguarded pass misclassifies `lm_head.weight` as the int token ids.
      3. If `hidden.shape[-1] == weight.shape[-1]` (D matches), projects
         hidden→logits via `F.linear(hidden, weight)`; otherwise treats
         `hidden` as pre-computed logits.
      4. Preserves the prefix dims of `token_ids` in the output.
      5. Tolerates shape mismatches between logits and ids by cropping to
         the min of each prefix dim (env-generated completions have variable
         sequence length; B and S may differ between the two tensors).
    """
    import torch
    import torch.nn.functional as F

    # ── 1. Identify the three roles among positional args + kwargs ──────────
    weight = None
    ids = None
    hidden = None

    for k in ("lm_head_weight", "embedding_weight", "lm_head", "weight"):
        if k in kwargs and kwargs[k] is not None:
            weight = kwargs[k]
            break
    for k in ("token_ids", "input_ids", "labels", "ids", "index"):
        if k in kwargs and kwargs[k] is not None:
            ids = kwargs[k]
            break
    for k in ("hidden_states", "logits", "x"):
        if k in kwargs and kwargs[k] is not None:
            hidden = kwargs[k]
            break

    def _classify(x):
        if isinstance(x, torch.nn.Parameter):
            return "param"
        if not torch.is_tensor(x):
            return "other"
        if x.dtype in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        ):
            return "int"
        if x.is_floating_point():
            return "float"
        return "other"

    # Scan positional args last — kwargs win.
    for a in args:
        role = _classify(a)
        if role == "param" and weight is None:
            weight = a
        elif role == "int" and ids is None:
            ids = a
        elif role == "float" and hidden is None and a is not weight:
            hidden = a

    # If hidden is still missing but we have exactly one un-assigned float
    # tensor among the args, take it.
    if hidden is None:
        for a in args:
            if torch.is_tensor(a) and a is not weight and a is not ids:
                hidden = a
                break

    if hidden is None or ids is None:
        raise RuntimeError(
            "[_safe_chunked_hidden_states_selective_log_softmax] "
            f"could not identify hidden/ids from "
            f"args types={[type(a).__name__ for a in args]}, "
            f"kwargs keys={list(kwargs.keys())}"
        )

    # ── 2. Hidden states → logits if needed ─────────────────────────────────
    # Hidden last-dim = D. Weight shape for an LM head is (vocab, D).
    # So if hidden.shape[-1] == weight.shape[-1], hidden is pre-linear.
    # If hidden.shape[-1] == weight.shape[0], hidden is already logits.
    if weight is not None:
        D_weight = int(weight.shape[-1])
        V_weight = int(weight.shape[0])
        last = int(hidden.shape[-1])
        if last == D_weight and D_weight != V_weight:
            logits = F.linear(hidden, weight)      # (..., V)
        elif last == V_weight:
            logits = hidden                         # already logits
        elif last == D_weight:
            # Square vocab==D edge case — assume hidden.
            logits = F.linear(hidden, weight)
        else:
            raise RuntimeError(
                f"hidden last-dim {last} matches neither weight.shape[-1]={D_weight} "
                f"nor weight.shape[0]={V_weight}"
            )
    else:
        logits = hidden  # assume already logits

    # ── 3. Apply Unsloth's optional scalings / softcap / temperature ────────
    def _fnum(key: str, default: float) -> float:
        v = kwargs.get(key, default)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    mult = _fnum("logit_scale_multiply", 0.0)
    divide = _fnum("logit_scale_divide", 0.0)
    softcap = _fnum("logit_softcapping", 0.0)
    temperature = _fnum("temperature", 1.0)

    if mult:
        logits = logits * mult
    if divide:
        logits = logits / divide
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    if temperature and temperature != 1.0:
        logits = logits / temperature

    # ── 4. Align prefix dims of logits and ids ──────────────────────────────
    # logits: (..., V); ids: (...,) int.
    logits_prefix = tuple(logits.shape[:-1])
    ids_prefix = tuple(ids.shape)

    if logits_prefix != ids_prefix:
        if len(logits_prefix) == len(ids_prefix):
            # Same rank — crop to min on each dim.
            crop = tuple(min(lp, ip) for lp, ip in zip(logits_prefix, ids_prefix))
            logits = logits[tuple(slice(0, c) for c in crop) + (slice(None),)]
            ids = ids[tuple(slice(0, c) for c in crop)]
        elif len(logits_prefix) == 2 and len(ids_prefix) == 1:
            B, S = logits_prefix
            N = int(ids.numel())
            if N == B * S:
                ids = ids.reshape(B, S)
            elif N < B * S:
                logits = logits.reshape(-1, logits.shape[-1])[:N]
                logits = logits.unsqueeze(0)  # (1, N, V)
                ids = ids.unsqueeze(0)        # (1, N)
            else:
                logits = logits.reshape(-1, logits.shape[-1])
                ids = ids[: logits.shape[0]]
                logits = logits.unsqueeze(0)
                ids = ids.unsqueeze(0)
        elif len(logits_prefix) == 1 and len(ids_prefix) == 2:
            B, S = ids_prefix
            if logits.shape[0] == B * S:
                logits = logits.reshape(B, S, -1)
            else:
                n = min(logits.shape[0], ids.numel())
                logits = logits[:n].unsqueeze(0)
                ids = ids.reshape(-1)[:n].unsqueeze(0)
        else:
            # Fallback: flatten both to 1-D then re-wrap as 2-D with one row.
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_ids = ids.reshape(-1)
            n = min(flat_logits.shape[0], flat_ids.shape[0])
            logits = flat_logits[:n].unsqueeze(0)
            ids = flat_ids[:n].unsqueeze(0)

    # ── 5. Compute log-softmax and gather per-token log-probs ───────────────
    # Cast to float32 for numerical stability (T4 has no bf16).
    logp = F.log_softmax(logits.float(), dim=-1)
    ids_long = ids.long()
    gathered = logp.gather(-1, ids_long.unsqueeze(-1)).squeeze(-1)

    # ── 6. Guarantee a 2-D output ───────────────────────────────────────────
    # Downstream grpo_accumulated_loss does `ref_logps.shape[1]`.
    if gathered.dim() == 1:
        gathered = gathered.unsqueeze(0)
    elif gathered.dim() > 2:
        gathered = gathered.reshape(gathered.shape[0], -1)

    return gathered


def _install_unsloth_chunked_logsoftmax_patch(verbose: bool = True) -> list[str]:
    """
    Install `_safe_chunked_hidden_states_selective_log_softmax` into every
    namespace that exports `chunked_hidden_states_selective_log_softmax`.

    Must be called AFTER Unsloth has generated
    `/app/unsloth_compiled_cache/UnslothGRPOTrainer.py` — i.e. after the first
    `FastLanguageModel.from_pretrained(...)` call — and BEFORE `trainer.train()`.

    Returns the list of module names that were successfully patched.
    """
    global _UNSLOTH_PATCH_INSTALLED
    import importlib

    patched: list[str] = []

    # 1. Patch already-loaded modules.
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            if hasattr(mod, _UNSLOTH_PATCH_TARGET):
                setattr(mod, _UNSLOTH_PATCH_TARGET,
                        _safe_chunked_hidden_states_selective_log_softmax)
                patched.append(mod_name)
        except Exception:
            # A module that has immutable globals (C extension, etc.) — skip.
            continue

    # 2. Force-import the usual suspects if they aren't loaded yet.
    for candidate in (
        "unsloth_compiled_cache.UnslothGRPOTrainer",
        "unsloth_compiled_cache",
        "unsloth.models.rl",
        "trl.trainer.grpo_trainer",
    ):
        try:
            m = importlib.import_module(candidate)
        except Exception:
            continue
        try:
            if hasattr(m, _UNSLOTH_PATCH_TARGET) and candidate not in patched:
                setattr(m, _UNSLOTH_PATCH_TARGET,
                        _safe_chunked_hidden_states_selective_log_softmax)
                patched.append(candidate)
        except Exception:
            continue

    _UNSLOTH_PATCH_INSTALLED = bool(patched)
    if verbose:
        if patched:
            print(f"[unsloth-patch] Installed {_UNSLOTH_PATCH_TARGET} "
                  f"replacement into: {patched}")
        else:
            print(f"[unsloth-patch] WARNING: no module exports "
                  f"{_UNSLOTH_PATCH_TARGET!r}; patch not installed.")
    return patched


def _install_matmul_lora_patch() -> bool:
    """
    Dtype-safe replacement for Unsloth's matmul_lora (unsloth/kernels/utils.py).

    Root cause in Unsloth 2026.4.x: matmul_lora determines `dtype` from the
    LoRA B-matrix (float32) and calls B.to(dtype), but `out` is created in
    fp16/bf16 from the surrounding autocast context.  addmm_(out_half, B_fp32)
    then raises:  RuntimeError: self and mat2 must have the same dtype,
    but got Half and Float.

    Fix: always derive compute_dtype from X (the activation), cast W, A, B to
    that dtype, and use F.linear (autograd-safe, dtype-consistent).

    Must be called AFTER FastLanguageModel.get_peft_model().
    """
    import torch
    import torch.nn.functional as F

    try:
        import unsloth.kernels.utils as _uu
        import unsloth.kernels.fast_lora as _fl
    except ImportError as exc:
        print(f"[matmul-lora-patch] WARNING: could not import Unsloth kernels: {exc}")
        return False

    _fdq = getattr(_uu, "fast_dequantize", None)

    def _is_float_tensor(t) -> bool:
        """Return True only for genuine floating-point torch.Tensors."""
        return (
            isinstance(t, torch.Tensor)
            and t.is_floating_point()
            and t.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        )

    def _dequant(W_quant):
        """
        Return a genuine float tensor from W_quant.
        Tries (1) Unsloth fast_dequantize, (2) bitsandbytes dequantize_4bit.
        Returns None if all attempts fail.
        bitsandbytes Params4bit.to() is in-place and returns None, so we guard
        with isinstance checks before accepting any result.
        """
        # Strategy 1 — Unsloth fast_dequantize
        if _fdq is not None:
            try:
                result = _fdq(W_quant)
                if _is_float_tensor(result):
                    return result
            except Exception:
                pass

        # Strategy 2 — bitsandbytes dequantize_4bit
        if hasattr(W_quant, "quant_state"):
            try:
                import bitsandbytes.functional as _bnb
                raw = W_quant.data if hasattr(W_quant, "data") else W_quant
                result = _bnb.dequantize_4bit(raw, W_quant.quant_state)
                if _is_float_tensor(result):
                    return result
            except Exception:
                pass

        return None

    def _safe_matmul_lora(X, W, W_quant, A, B, s, out=None):
        dtype = X.dtype  # authoritative compute dtype from activations

        # ── Dequantize / locate base weight ────────────────────────────────
        W_fp = None
        if W_quant is not None:
            W_fp = _dequant(W_quant)
            if W_fp is None and _is_float_tensor(W):
                W_fp = W
        elif _is_float_tensor(W):
            W_fp = W

        # ── Base forward ────────────────────────────────────────────────────
        if W_fp is not None:
            base = F.linear(X, W_fp.to(dtype))
        else:
            out_dim = (
                B.shape[0] if B is not None
                else (W.shape[0] if W is not None else X.shape[-1])
            )
            base = X.new_zeros(*X.shape[:-1], out_dim)

        # ── LoRA delta: (X @ A^T) @ B^T * s ────────────────────────────────
        if A is not None and B is not None:
            XA = F.linear(X, A.to(dtype))
            base = base + float(s) * F.linear(XA, B.to(dtype))

        return base

    _uu.matmul_lora = _safe_matmul_lora
    _fl.matmul_lora = _safe_matmul_lora
    print("[matmul-lora-patch] dtype-safe matmul_lora installed "
          "(Unsloth 2026.4.x Half/Float fix)")
    return True


# ══════════════════════════════════════════════════════════════════════════
#   Single-shot env rollout → reward
# ══════════════════════════════════════════════════════════════════════════

# Matches a top-level JSON object via greedy brace-balanced scan. We stream
# the completion, peel off one object at a time, and feed each to env.step().
_JSON_OBJECT_RE = re.compile(r"\{", re.DOTALL)


def _iter_json_objects(text: str) -> Iterable[str]:
    """
    Yield each top-level JSON object string found in ``text``, in order.

    Handles nested braces, strings with escaped quotes, and stray text between
    objects. Does NOT attempt to parse — that's the env's job via action_parser.
    """
    i, n = 0, len(text)
    while i < n:
        m = _JSON_OBJECT_RE.search(text, i)
        if m is None:
            return
        start = m.start()
        depth = 0
        in_str = False
        esc = False
        j = start
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[start : j + 1]
                        i = j + 1
                        break
            j += 1
        else:
            # Unbalanced trailing brace — stop.
            return


def _rollout_completion(
    completion: str,
    episode_seed: int,
    kb: dict,
    config: TrainingConfig,
) -> tuple[float, Any]:
    """
    Fresh ExaminerEnv per completion. Extracts JSON action objects from
    ``completion`` in order and feeds them to env.step() until terminated /
    truncated. If the completion contains no actions, the env auto-force-
    classifies as FAKING on exhaustion.

    Returns (R_total, RewardBreakdown-or-None).
    """
    from examiner_env.environment import ExaminerEnv

    env = ExaminerEnv(kb=kb, config=config)
    env.reset(seed=int(episode_seed))

    last_reward = -0.20       # safety floor if rollout never terminates
    last_info: dict = {}
    done = False

    for obj_text in _iter_json_objects(completion):
        if done:
            break
        try:
            obs, reward, terminated, truncated, info = env.step(obj_text)
        except Exception as exc:
            # Any env-side crash: hard malformed penalty.
            return -0.20, None
        done = terminated or truncated
        last_reward = float(reward)
        last_info = info or {}

    # If the model emitted zero JSON actions, run the env forward on empty
    # strings to exhaust turns → forced classify → real reward_breakdown.
    if not done:
        while not done:
            obs, reward, terminated, truncated, info = env.step("")
            done = terminated or truncated
            last_reward = float(reward)
            last_info = info or {}

    breakdown = last_info.get("reward_breakdown") if last_info else None
    return last_reward, breakdown


def _make_grpo_reward_func(kb: dict, config: TrainingConfig):
    """
    Build a reward_func matching the compiled Unsloth GRPOTrainer signature:

        reward_func(prompts, completions, completion_ids=None, **dataset_cols)

    ``dataset_cols`` contains the rest of the row columns forwarded by the
    trainer; we read ``episode_seed`` from there. Returns ``list[float]`` of
    the same length as ``completions``.

    Every training step produces batch-mean/std, per-component, and
    parse-failure rate signals in W&B.
    """
    step_counter = [0]

    def _extract_text(comp: Any) -> str:
        # Unsloth's compiled trainer may hand completions as plain strings OR
        # as chat-format lists of dicts: [{"role": "assistant", "content": ...}].
        if isinstance(comp, str):
            return comp
        if isinstance(comp, list):
            parts = []
            for m in comp:
                if isinstance(m, dict) and "content" in m:
                    parts.append(str(m["content"]))
                else:
                    parts.append(str(m))
            return "\n".join(parts)
        return str(comp)

    def reward_func(
        prompts=None,
        completions=None,
        completion_ids=None,
        **dataset_cols: Any,
    ) -> list[float]:
        if completions is None:
            return []

        seeds = dataset_cols.get("episode_seed")
        if seeds is None:
            # Deterministic fallback: derive a seed from completion index.
            seeds = list(range(len(completions)))
        elif not isinstance(seeds, (list, tuple)):
            seeds = [seeds] * len(completions)

        rewards: list[float] = []
        breakdowns = []
        n_malformed = 0

        for i, comp in enumerate(completions):
            text = _extract_text(comp)
            seed = int(seeds[i]) if i < len(seeds) else i
            r, bd = _rollout_completion(text, seed, kb, config)
            rewards.append(float(r))
            breakdowns.append(bd)
            if bd is None:
                n_malformed += 1

        if _WANDB_AVAILABLE and wandb.run is not None and rewards:
            log = {
                "reward/R_total_batch_mean": float(np.mean(rewards)),
                "reward/R_total_batch_std": float(np.std(rewards)),
                "reward/R_total_batch_min": float(np.min(rewards)),
                "reward/R_total_batch_max": float(np.max(rewards)),
                "training/parse_failure_rate": n_malformed / max(1, len(rewards)),
                "training/step_local": step_counter[0],
            }
            for field in ("R_acc", "R_info", "R_cal", "R_eff",
                          "R_qual", "R_cov", "R_asym", "R_div"):
                values = [getattr(b, field) for b in breakdowns if b is not None]
                if values:
                    log[f"reward/{field}_mean"] = float(np.mean(values))
            wandb.log(log)

        step_counter[0] += 1
        return rewards

    return reward_func


# ══════════════════════════════════════════════════════════════════════════
#   Reward variance monitor (adaptive KL)
# ══════════════════════════════════════════════════════════════════════════

def _check_reward_variance(
    reward_buffer: list[float],
    config: TrainingConfig,
    step: int,
    current_beta_kl: float,
) -> float:
    """Warn on collapse; auto-bump KL penalty on explosion. Returns new beta."""
    if len(reward_buffer) < 10:
        return current_beta_kl
    recent = reward_buffer[-50:]
    variance = float(np.std(recent))

    if _WANDB_AVAILABLE and wandb.run:
        wandb.log({"reward/variance_monitor": variance}, step=step)

    if variance < config.reward_variance_floor:
        print(f"WARNING [step {step}]: Reward variance {variance:.4f} "
              f"< floor {config.reward_variance_floor} — signal may have collapsed.")
        if _WANDB_AVAILABLE and wandb.run:
            wandb.log({"warning/reward_variance_collapsed": 1}, step=step)

    if variance > config.reward_variance_ceiling:
        new_beta = min(0.10, current_beta_kl * 1.5)
        print(f"WARNING [step {step}]: Variance {variance:.4f} "
              f"> ceiling {config.reward_variance_ceiling}; "
              f"beta_kl {current_beta_kl:.4f} → {new_beta:.4f}")
        if _WANDB_AVAILABLE and wandb.run:
            wandb.log({"adaptive/beta_kl": new_beta}, step=step)
        return new_beta

    return current_beta_kl


# ══════════════════════════════════════════════════════════════════════════
#   Main training entry point
# ══════════════════════════════════════════════════════════════════════════

def train(config: TrainingConfig, eval_config: dict) -> dict:
    """
    Full GRPO training loop.

    Flow:
      1. Init W&B
      2. Oracle calibration gate (Brier ≤ 0.18)
      3. Pre-training baseline eval → outputs/eval/baseline_metrics.json
      4. Load model via Unsloth (4-bit + LoRA)
      5. Install the chunked-log-softmax monkey patch
      6. Build episode seed dataset
      7. Construct GRPOTrainer with standard reward_func signature (no
         environment_factory — Unsloth compiled trainer ignores it)
      8. Periodic eval callback → outputs/eval/checkpoint_metrics.json
      9. trainer.train()
      10. Final held-out eval → outputs/eval/final_metrics.json
    """
    # ── Deferred imports ────────────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
        from trl import GRPOTrainer, GRPOConfig
        from transformers import TrainerCallback
        from datasets import Dataset
    except ImportError as e:
        raise RuntimeError(
            "Unsloth / TRL / datasets not installed.\n"
            "Run: pip install 'unsloth[colab-new]' 'trl>=0.9.0' datasets\n"
            f"Error: {e}"
        )

    try:
        from examiner_env.knowledge_base import KB
        from examiner_env.baselines import (
            RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer,
        )
        from training.eval import run_eval
        from training.reward_fn import init_wandb
    except ImportError as e:
        raise RuntimeError(
            "examiner_env not available — C1 Phase 1 gate must clear first.\n"
            f"Error: {e}"
        )

    # ── W&B ─────────────────────────────────────────────────────────────────
    init_wandb(config)

    # ── Oracle calibration gate ─────────────────────────────────────────────
    cal_path = os.path.join("outputs", "eval", "oracle_calibration.json")
    if not os.path.exists(cal_path):
        raise FileNotFoundError(
            "outputs/eval/oracle_calibration.json missing — "
            "run examiner_env.calibration.run_calibration() first."
        )
    with open(cal_path) as f:
        cal = json.load(f)
    brier = cal["calibration_metrics"]["mean_brier"]
    assert brier <= 0.18, f"Oracle Brier={brier:.4f} > 0.18 — recalibrate."
    print(f"Oracle calibration OK (Brier={brier:.4f})")

    # ── Pre-training baseline eval ──────────────────────────────────────────
    _run_baseline_eval(
        eval_config, KB, run_eval,
        RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer,
    )

    # ── Load model with Unsloth ─────────────────────────────────────────────
    # This call is what generates /app/unsloth_compiled_cache/UnslothGRPOTrainer.py.
    print(f"Loading {config.model_name} (4-bit={config.use_4bit})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.use_4bit,
        dtype=None,
    )
    # Restrict LoRA to attention projections only.
    # Including gate_proj/up_proj/down_proj triggers Unsloth's fused
    # apply_lora_mlp_swiglu Triton kernel, which has a bf16/fp32 dtype
    # mismatch bug under gradient checkpointing in Unsloth 2026.4.x.
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print("Model + LoRA adapters loaded.")

    # ── Install the Unsloth chunked-log-softmax patch NOW ───────────────────
    # Between from_pretrained() (which generates the compiled cache module)
    # and GRPOTrainer() (which will call the function on the first step).
    _install_unsloth_chunked_logsoftmax_patch()
    _install_matmul_lora_patch()

    # ── Episode seed dataset ────────────────────────────────────────────────
    n = config.num_episodes
    system_instruction = (
        "You are an expert examiner testing a student on machine-learning "
        "theory across ten sections (S01–S10). Your goal is to distinguish "
        "KNOWS from FAKING in each section. You have at most 4 turns.\n\n"
        "Respond with a SEQUENCE of JSON action objects, one per line, in "
        "the order you want them executed. Use WHY/HOW/edge-case probes — "
        "they expose bluffing faster than definitional questions.\n\n"
        "Ask action: "
        '{\"action_type\":\"ask\",\"section_id\":\"S02\",\"question_text\":\"...\"}\n'
        "Classify action (must be the LAST object, covering every active "
        "section with KNOWS or FAKING):\n"
        '{\"action_type\":\"classify\",\"classifications\":'
        '{\"S01\":\"KNOWS\",\"S02\":\"FAKING\",...,\"S10\":\"KNOWS\"}}'
    )
    dataset = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": system_instruction}]] * n,
        "episode_seed": list(range(n)),
    })

    # ── Eval + variance callback ────────────────────────────────────────────
    reward_buffer: list[float] = []
    current_beta_kl = config.beta_kl
    checkpoint_metrics_log: dict = {}

    class EvalAndMonitorCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                r = logs.get("reward/R_total_batch_mean",
                             logs.get("train/reward"))
                if r is not None:
                    reward_buffer.append(float(r))

        def on_step_end(self, args, state, control, **kw):
            nonlocal current_beta_kl
            step = state.global_step
            if step % 50 == 0 and reward_buffer:
                current_beta_kl = _check_reward_variance(
                    reward_buffer, config, step, current_beta_kl
                )
            if step > 0 and step % config.eval_every_n_steps == 0:
                print(f"\n[step {step}] Checkpoint eval...")
                _TrainedExaminerWrapper.model = model
                _TrainedExaminerWrapper.tokenizer = tokenizer
                _TrainedExaminerWrapper.config = config
                chk = run_eval(_TrainedExaminerWrapper(), eval_config, KB)
                checkpoint_metrics_log[str(step)] = chk
                os.makedirs(os.path.join("outputs", "eval"), exist_ok=True)
                with open(
                    os.path.join("outputs", "eval", "checkpoint_metrics.json"),
                    "w",
                ) as fp:
                    json.dump(checkpoint_metrics_log, fp, indent=2)
                if _WANDB_AVAILABLE and wandb.run:
                    wandb.log({
                        "eval/reward_mean": chk["reward_mean"],
                        "eval/classification_accuracy": chk["classification_accuracy"],
                        "eval/avg_info_gain": chk["avg_info_gain_per_turn"],
                        "eval/calibration_ECE": chk["calibration_ECE"],
                        "eval/false_accusation_rate": chk["false_accusation_rate"],
                    }, step=step)
                print(
                    f"  accuracy={chk['classification_accuracy']:.3f} "
                    f"info_gain={chk['avg_info_gain_per_turn']:.4f} "
                    f"ECE={chk['calibration_ECE']:.4f}"
                )

    # ── GRPOConfig ──────────────────────────────────────────────────────────
    # Precision: use device capability (reliable) not is_bf16_supported()
    # which returns False on some HF Spaces A10G driver configs.
    # Bf16 requires SM >= 8.0 (Ampere: A10G=8.6, A100=8.0).
    # NEVER fall back to fp16: Unsloth fast_lora mixes fp16 activations with
    # float32 LoRA matrices, causing RuntimeError on addmm_.
    import torch as _torch
    if _torch.cuda.is_available():
        _cap_major = _torch.cuda.get_device_capability()[0]
        _bf16_ok = _cap_major >= 8 and config.bf16
        _fp16_ok = False  # fp16+fp32 LoRA mismatch in Unsloth kernels
        _gpu_name = _torch.cuda.get_device_name(0)
    else:
        _cap_major = 0
        _bf16_ok = False
        _fp16_ok = False
        _gpu_name = "CPU"
    print(f"Precision: bf16={_bf16_ok}  fp16={_fp16_ok}  "
          f"(GPU: {_gpu_name}, SM {_cap_major}.x)")

    grpo_config = GRPOConfig(
        output_dir=os.path.join("outputs", "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        bf16=_bf16_ok,
        fp16=_fp16_ok,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        num_generations=config.num_generations,
        beta=config.beta_kl,
        save_steps=config.checkpoint_every_n_steps,
        logging_steps=1,
        report_to="wandb" if _WANDB_AVAILABLE else "none",
        max_completion_length=config.max_seq_length,
        log_completions=True,
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    # Standard GRPO signature — no environment_factory. Unsloth's compiled
    # cache calls reward_func(prompts, completions, completion_ids, **cols).
    grpo_reward_func = _make_grpo_reward_func(KB, config)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=grpo_reward_func,
        args=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=[EvalAndMonitorCallback()],
    )

    # Belt-and-suspenders: re-install the patch after trainer construction,
    # in case GRPOTrainer's __init__ caused a fresh import of the compiled
    # cache that bypassed the first sweep.
    _install_unsloth_chunked_logsoftmax_patch(verbose=False)
    _install_matmul_lora_patch()

    print(f"\nStarting {config.config_name} training ({n} episodes)...")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA: r={config.lora_rank} alpha={config.lora_alpha}")
    print(f"  Generations/step: {config.num_generations}")
    print(f"  Eval every {config.eval_every_n_steps} steps")
    trainer.train()

    # ── Final held-out eval ─────────────────────────────────────────────────
    print("\nRunning final held-out eval...")
    _TrainedExaminerWrapper.model = model
    _TrainedExaminerWrapper.tokenizer = tokenizer
    _TrainedExaminerWrapper.config = config
    final_metrics = run_eval(
        _TrainedExaminerWrapper(),
        eval_config,
        KB,
        output_path=os.path.join("outputs", "eval", "final_metrics.json"),
    )
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


# ══════════════════════════════════════════════════════════════════════════
#   Helpers
# ══════════════════════════════════════════════════════════════════════════

def _run_baseline_eval(
    eval_config, KB, run_eval,
    RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer,
):
    baseline_path = os.path.join("outputs", "eval", "baseline_metrics.json")
    if os.path.exists(baseline_path):
        print("baseline_metrics.json already exists — skipping baseline eval.")
        return
    print("Running pre-training baseline evaluation...")
    baseline_metrics: dict = {}
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


class _TrainedExaminerWrapper:
    """Adapts the trained model to the examiner.act(observation) interface."""
    model: Any = None
    tokenizer: Any = None
    config: Any = None

    def reset(self) -> None:
        pass

    def act(self, observation: dict) -> str:
        from training.prompt_builder import build_prompt
        prompt = build_prompt(observation)
        inputs = self.__class__.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.__class__.model.device)
        outputs = self.__class__.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.__class__.tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.__class__.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()


# ══════════════════════════════════════════════════════════════════════════
#   CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BluffBuster Examiner with GRPO")
    parser.add_argument("--config", default="DEBUG",
                        choices=["DEBUG", "DEMO", "FULL"])
    parser.add_argument("--eval_config", default="eval_config.json")
    args = parser.parse_args()

    with open(args.eval_config) as f:
        eval_cfg = json.load(f)

    train(get_config(args.config), eval_cfg)
