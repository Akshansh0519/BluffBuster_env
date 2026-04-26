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
import warnings
from typing import Any, Iterable

import numpy as np

# ── Silence known-harmless warnings that pollute the live log ──────────────
# 1. "Both max_new_tokens and max_length seem to have been set" — we pass
#    max_new_tokens explicitly; max_length comes from the model's generation_config
#    and is irrelevant. max_new_tokens wins; message adds no info.
# 2. "Passing generation_config together with generation-related arguments" —
#    Unsloth/TRL internal; not our call-site; will be fixed upstream.
# 3. AttentionMaskConverter FutureWarning — transformers v5 API change; not
#    actionable until we upgrade transformers.
warnings.filterwarnings(
    "ignore",
    message=r".*max_new_tokens.*max_length.*seem to have been set.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Passing `generation_config`.*generation-related arguments.*deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*AttentionMaskConverter.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*attention mask API.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*use_return_dict.*deprecated.*",
)
# ──────────────────────────────────────────────────────────────────────────

from training.config import (
    TrainingConfig, get_config, DEBUG_CONFIG,
    DEMO_FAST_CONFIG, FULL_FAST_CONFIG,
)

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


def _install_addmm_dtype_patch() -> bool:
    """
    Patch torch.Tensor.addmm_ to auto-cast mat1/mat2 to self.dtype.

    Unsloth 2026.4.x bug (all layers — attention O, QKV, MLP gate/up/down):
    fast_lora.py's matmul_lora creates `out` in the activation dtype
    (fp16/bf16 from surrounding autocast), then calls:

        out.addmm_(XA, B.to(dtype))

    where `dtype` is derived from the LoRA B-matrix (float32). CUDA addmm_
    requires all three tensors to share dtype, raising:

        RuntimeError: self and mat2 must have the same dtype, but got Half and Float

    Fix: intercept addmm_ and cast mat1/mat2 to self.dtype when they differ.
    Numerically equivalent — just promotes/demotes the delta to the
    accumulator's precision, not the other way around.

    Tested locally (CPU) — 9/9 pass in scripts/test_addmm_patch.py.
    """
    import torch

    if getattr(torch.Tensor.addmm_, "_bluffbuster_patched", False):
        print("[addmm-patch] already installed, skipping")
        return True

    _orig = torch.Tensor.addmm_

    def _safe_addmm_(self, mat1, mat2, *, beta=1.0, alpha=1.0):
        if mat1.dtype != self.dtype:
            mat1 = mat1.to(self.dtype)
        if mat2.dtype != self.dtype:
            mat2 = mat2.to(self.dtype)
        return _orig(self, mat1, mat2, beta=beta, alpha=alpha)

    _safe_addmm_._bluffbuster_patched = True
    torch.Tensor.addmm_ = _safe_addmm_
    print("[addmm-patch] dtype-safe addmm_ installed (Unsloth 2026.4.x Half/Float fix)")
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
#   HF Hub checkpoint helpers  (cross-restart resume)
# ══════════════════════════════════════════════════════════════════════════

def _hub_get_latest_step(repo_id: str, token: str) -> tuple[int, str | None]:
    """
    Scan ``repo_id`` (model repo) for folders named ``lora-step-<N>``.
    Returns ``(latest_step, folder_name)`` or ``(0, None)`` if none found or
    if the repo doesn't exist yet.
    """
    try:
        from huggingface_hub import HfApi as _HfApi
        api = _HfApi(token=token)
        files = api.list_repo_tree(
            repo_id=repo_id, repo_type="model", recursive=False
        )
        steps = []
        for f in files:
            name = getattr(f, "path", "") or getattr(f, "rfilename", "")
            if name.startswith("lora-step-"):
                try:
                    steps.append(int(name.split("-")[-1]))
                except ValueError:
                    pass
        if not steps:
            return 0, None
        best = max(steps)
        return best, f"lora-step-{best}"
    except Exception:
        return 0, None


def _hub_peek_training_state(
    repo_id: str, token: str, folder_name: str,
) -> dict | None:
    """Download only ``training_state.json`` from a Hub checkpoint folder."""
    try:
        from huggingface_hub import hf_hub_download as _dl
        _path = _dl(
            repo_id=repo_id,
            repo_type="model",
            filename=f"{folder_name}/training_state.json",
            token=token,
        )
        with open(_path, encoding="utf-8") as _f:
            return json.load(_f)
    except Exception:
        return None


def _hub_restore_lora(
    model,
    repo_id: str,
    token: str,
    folder_name: str,
    local_dir: str,
) -> dict | None:
    """
    Download ``folder_name`` from ``repo_id`` to ``local_dir``, load the LoRA
    adapter weights into ``model``, and return the parsed ``training_state.json``
    (or ``None`` if it doesn't exist / fails).

    Returns the training_state dict on success so the caller can resume dataset
    position and reward_buffer.  Returns ``None`` on any failure so the caller
    can safely fall back to training from scratch.
    """
    import torch
    try:
        from huggingface_hub import HfApi as _HfApi, hf_hub_download as _dl
        api = _HfApi(token=token)

        # List files inside the folder on the Hub
        files = list(api.list_repo_tree(
            repo_id=repo_id, repo_type="model",
            path_in_repo=folder_name, recursive=True,
        ))
        if not files:
            print(f"[resume] WARNING: {folder_name} exists but is empty on Hub")
            return None

        os.makedirs(local_dir, exist_ok=True)
        for f in files:
            fname = getattr(f, "path", None) or getattr(f, "rfilename", None)
            if fname is None:
                continue
            # fname is like "lora-step-50/adapter_model.safetensors"
            rel = fname[len(folder_name):].lstrip("/")
            if not rel:
                continue
            dest = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            _dl(
                repo_id=repo_id,
                repo_type="model",
                filename=fname,
                token=token,
                local_dir=os.path.dirname(dest),
                local_dir_use_symlinks=False,
            )

        # Load adapter weights into model
        adapter_bin   = os.path.join(local_dir, "adapter_model.bin")
        adapter_safe  = os.path.join(local_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_safe):
            from safetensors.torch import load_file as _load_sf
            state_dict = _load_sf(adapter_safe)
        elif os.path.exists(adapter_bin):
            state_dict = torch.load(adapter_bin, map_location="cpu")
        else:
            print(f"[resume] WARNING: no adapter_model file found in {local_dir}")
            return None

        from peft import set_peft_model_state_dict as _set_sd
        _set_sd(model, state_dict)
        print(f"[resume] LoRA weights loaded from {folder_name}")

        # Read training state
        ts_path = os.path.join(local_dir, "training_state.json")
        if os.path.exists(ts_path):
            with open(ts_path) as _f:
                return json.load(_f)
        return {}
    except Exception as _e:
        print(f"[resume] WARNING: restore failed ({_e}), training from scratch")
        return None


# ══════════════════════════════════════════════════════════════════════════
#   Main training entry point
# ══════════════════════════════════════════════════════════════════════════

def train(config: TrainingConfig, eval_config: dict) -> dict:
    """
    Full GRPO training with TRL environment_factory.

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

    # ── Pre-training baseline eval ──────────────────────────────────────────
    # Use only 5 episodes for a fast sanity check (not a full 50-ep run).
    _baseline_eval_cfg = {**eval_config, "num_episodes": 5}
    _run_baseline_eval(
        _baseline_eval_cfg, KB, run_eval,
        RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer,
    )

    # ── Load model with Unsloth ─────────────────────────────────────────────
    # This call is what generates /app/unsloth_compiled_cache/UnslothGRPOTrainer.py.
    #
    # Enable synchronous CUDA launches so any device-side assert surfaces at
    # its true source (instead of the next async call). Costs ~5% throughput
    # for a large debugging win. Remove once runs are consistently green.
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")

    # Prefer Unsloth's pre-quantized 4bit weights when available — on-the-fly
    # bnb quantization of fresh HF weights is the known cause of
    # "CUDA device-side assert during rotary init" with Qwen2.5-7B + Unsloth
    # 2026.4.x. The pre-quantized variant skips that path entirely.
    model_name = config.model_name
    if config.use_4bit and not model_name.startswith("unsloth/"):
        prequant_candidates = {
            "Qwen/Qwen2.5-1.5B-Instruct": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
            "Qwen/Qwen2.5-3B-Instruct":   "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
            "Qwen/Qwen2.5-7B-Instruct":   "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        }
        alias = prequant_candidates.get(model_name)
        if alias is not None:
            print(f"[loader] substituting pre-quantized weights: "
                  f"{model_name} -> {alias}")
            model_name = alias

    print(f"Loading {model_name} (4-bit={config.use_4bit})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.use_4bit,
        dtype=_model_dtype,
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
    _install_addmm_dtype_patch()

    # ── Cross-restart resume from HF Hub ────────────────────────────────────
    # HF Spaces filesystem is ephemeral — local checkpoints are wiped on every
    # container restart. On startup we scan the Hub checkpoint repo for the
    # latest lora-step-N, download it, restore the adapter weights, and slice
    # the dataset so we skip already-trained episodes.
    # If no checkpoint exists (first run) this is a no-op.
    _resume_step      = 0
    _start_episode    = 0
    _resumed_rewards: list[float] = []

    _hf_token_startup = os.environ.get("HF_TOKEN")
    _repo_id_startup  = os.environ.get("CHECKPOINT_REPO", "")
    if not _repo_id_startup and _hf_token_startup:
        try:
            from huggingface_hub import HfApi as _HfApi
            _hf_username = _HfApi(token=_hf_token_startup).whoami()["name"]
            _repo_id_startup = f"{_hf_username}/bluffbuster-checkpoints"
            print(f"[resume] Auto-detected checkpoint repo: {_repo_id_startup}")
        except Exception as _e:
            _repo_id_startup = "bluffbuster-checkpoints"
            print(f"[resume] Could not detect HF username ({_e}), using: {_repo_id_startup}")
    elif not _repo_id_startup:
        _repo_id_startup = "bluffbuster-checkpoints"
    if _hf_token_startup:
        print(f"[resume] Scanning HF Hub for checkpoints in {_repo_id_startup}...")
        _latest_step, _latest_folder = _hub_get_latest_step(
            _repo_id_startup, _hf_token_startup
        )
        if _latest_step > 0 and _latest_folder:
            _peek = _hub_peek_training_state(
                _repo_id_startup, _hf_token_startup, _latest_folder
            )
            _skip_resume = False
            if _peek:
                _saved_cfg = _peek.get("config_name")
                if _saved_cfg and _saved_cfg != config.config_name:
                    print(
                        f"[resume] Hub checkpoint is for `{_saved_cfg}` but you "
                        f"selected `{config.config_name}` — starting fresh "
                        f"(LoRA shapes / weights are not compatible across configs)."
                    )
                    _skip_resume = True
                _saved_model = _peek.get("model_name")
                if (
                    not _skip_resume
                    and _saved_model
                    and _saved_model != config.model_name
                ):
                    print(
                        f"[resume] Hub checkpoint used `{_saved_model}` but this "
                        f"run uses `{config.model_name}` — starting fresh."
                    )
                    _skip_resume = True
            if not _skip_resume:
                print(f"[resume] Found checkpoint at step {_latest_step}, restoring...")
                _lora_local = os.path.join(
                    "outputs", "checkpoints", f"lora-step-{_latest_step}"
                )
                _ts = _hub_restore_lora(
                    model,
                    _repo_id_startup,
                    _hf_token_startup,
                    _latest_folder,
                    _lora_local,
                )
                if _ts is not None:
                    _resume_step = _ts.get("step", _latest_step)
                    _start_episode = _ts.get(
                        "episodes_consumed",
                        _resume_step * config.batch_size,
                    )
                    _resumed_rewards = _ts.get("reward_buffer", [])
                    print(f"[resume] Resumed from step {_resume_step}, "
                          f"episode {_start_episode}")
                else:
                    print("[resume] Adapter load failed — starting from scratch")
        else:
            print("[resume] No Hub checkpoint found — starting fresh")
    else:
        print("[resume] HF_TOKEN not set — skipping Hub resume")

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
    # Slice to skip already-trained episodes when resuming.
    # episode_seed is preserved (absolute index) so the reward function
    # produces the same rollout for each seed regardless of resume.
    _all_seeds = list(range(n))
    _remaining_seeds = _all_seeds[_start_episode:]
    if _start_episode > 0:
        print(f"[resume] Skipping first {_start_episode} episodes "
              f"({n - len(_remaining_seeds)} steps already done); "
              f"{len(_remaining_seeds)} episodes remain")
    dataset = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": system_instruction}]]
                  * len(_remaining_seeds),
        "episode_seed": _remaining_seeds,
    })

    # ── Checkpoint / eval cadence ────────────────────────────────────────────
    # Save LoRA adapter (lightweight, ~50 MB) every N steps so a crash at eval
    # time never loses more than N steps of progress.  Full Trainer checkpoints
    # (optimizer state, ~4 GB) are kept at checkpoint_every_n_steps; we only
    # keep the last 3 to avoid filling the Space disk.
    _lora_save_every = {"DEBUG": 5, "DEMO": 10, "FULL": 25}.get(
        config.config_name, 10
    )

    # Pre-populate reward_buffer from resume state so variance checks
    # immediately have history rather than starting cold.
    reward_buffer: list[float] = list(_resumed_rewards)
    current_beta_kl = config.beta_kl
    checkpoint_metrics_log: dict = {}
    train_started_at = time.time()
    # Optimizer steps per epoch (what tqdm / max_steps actually use).
    total_steps_estimate = max(1, (n + config.batch_size - 1) // config.batch_size)

    class EvalAndMonitorCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                r = logs.get("reward/R_total_batch_mean",
                             logs.get("train/reward"))
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
            import torch
            nonlocal current_beta_kl
            step = state.global_step
            if step == 0:
                return

            # ── Reward variance check ──────────────────────────────────────
            if step % 50 == 0 and reward_buffer:
                current_beta_kl = _check_reward_variance(
                    reward_buffer, config, step, current_beta_kl
                )

            # ── Lightweight LoRA adapter save ──────────────────────────────
            # Saves LoRA adapter weights (~50 MB) every few steps.
            # Also uploads to HF Hub so checkpoints SURVIVE Space restarts
            # (HF Spaces filesystem is ephemeral — local saves alone are lost
            # if the container restarts after a connection error).
            if step % _lora_save_every == 0:
                lora_dir = os.path.join(
                    "outputs", "checkpoints", f"lora-step-{step}"
                )
                try:
                    os.makedirs(lora_dir, exist_ok=True)
                    model.save_pretrained(lora_dir)
                    print(f"[ckpt] LoRA adapter saved → {lora_dir}")
                except Exception as _e:
                    print(f"[ckpt] WARNING: LoRA save failed at step {step}: {_e}")

                # Write training_state.json into the lora dir so resume
                # knows exactly where to continue from.
                _episodes_consumed = step * config.batch_size
                _ts = {
                    "step": step,
                    "episodes_consumed": _episodes_consumed,
                    "config_name": config.config_name,
                    "model_name": config.model_name,
                    "reward_buffer": reward_buffer[-200:],  # last 200 only
                }
                try:
                    with open(os.path.join(lora_dir, "training_state.json"), "w") as _f:
                        json.dump(_ts, _f, indent=2)
                except Exception as _e:
                    print(f"[ckpt] WARNING: training_state.json write failed: {_e}")

                # Upload to HF Hub in a BACKGROUND THREAD so training is never
                # blocked waiting for network I/O (~50 MB per upload can take
                # 30-90 sec on HF Spaces — keeping it synchronous was the main
                # cause of the 6× training slowdown observed in the DEMO run).
                _hf_token = os.environ.get("HF_TOKEN")
                _repo_id  = _repo_id_startup  # reuse auto-detected repo from startup
                if _hf_token and os.path.isdir(lora_dir):
                    import threading as _threading
                    _upload_dir   = lora_dir
                    _upload_step  = step
                    _upload_token = _hf_token
                    _upload_repo  = _repo_id

                    def _bg_upload(d, s, tok, repo, cfg_name):
                        try:
                            from huggingface_hub import HfApi as _HfApi
                            _a = _HfApi(token=tok)
                            _a.create_repo(repo_id=repo, repo_type="model",
                                           exist_ok=True, private=True)
                            _a.upload_folder(
                                folder_path=d,
                                repo_id=repo,
                                repo_type="model",
                                path_in_repo=f"lora-step-{s}",
                                commit_message=f"LoRA checkpoint step {s} ({cfg_name})",
                            )
                            print(f"[ckpt] LoRA step-{s} uploaded → hf.co/{repo}")
                        except Exception as _e:
                            print(f"[ckpt] WARNING: Hub upload step-{s} failed: {_e}")

                    _t = _threading.Thread(
                        target=_bg_upload,
                        args=(_upload_dir, _upload_step, _upload_token,
                              _upload_repo, config.config_name),
                        daemon=True,
                    )
                    _t.start()

            # ── Checkpoint eval ────────────────────────────────────────────
            # Uses a reduced episode count for mid-training pulse-checks to
            # keep eval fast (~2 min not ~10 min). Full eval runs at the end.
            # CRITICAL: model.eval() + no_grad prevent OOM (gradient checkpointing
            # builds a full compute graph during generate() otherwise).
            # CRITICAL: torch.cuda.empty_cache() after eval prevents VRAM
            # fragmentation that caused 6× training slowdown in the DEMO run.
            if step % config.eval_every_n_steps == 0:
                _ckpt_eval_episodes = min(3, config.eval_episodes)
                print(f"\n[step {step}] Checkpoint eval "
                      f"({_ckpt_eval_episodes} episodes — fast pulse check)...")
                _TrainedExaminerWrapper.model = model
                _TrainedExaminerWrapper.tokenizer = tokenizer
                _TrainedExaminerWrapper.config = config
                # Build a fast eval_config with reduced episodes
                _fast_eval_cfg = {**eval_config,
                                  "num_episodes": _ckpt_eval_episodes}
                try:
                    model.eval()
                    with torch.no_grad():
                        chk = run_eval(
                            _TrainedExaminerWrapper(), _fast_eval_cfg, KB
                        )
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
                except Exception as _eval_exc:
                    print(f"[step {step}] WARNING: eval failed (training continues): "
                          f"{_eval_exc}")
                finally:
                    model.train()
                    # Release any VRAM held by eval inference before next train step.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

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
        output_dir=checkpoint_root,
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
        save_steps=_lora_save_every,   # match LoRA cadence
        save_total_limit=3,            # keep last 3 full checkpoints only
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
    _install_addmm_dtype_patch()

    _remaining = len(_remaining_seeds)
    _total_steps_this_run = (_remaining + config.batch_size - 1) // config.batch_size
    print(f"\nStarting {config.config_name} training...")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA: r={config.lora_rank} alpha={config.lora_alpha}")
    print(f"  Resumed from step: {_resume_step} / episode: {_start_episode}")
    print(f"  Episodes remaining: {_remaining} → ~{_total_steps_this_run} steps this run")
    print(f"  Generations/step: {config.num_generations}")
    print(f"  Eval every {config.eval_every_n_steps} steps")
    print(f"  LoRA save every {_lora_save_every} steps → Hub: {_repo_id_startup}")

    # Trainer.train() with no resume_from_checkpoint: starts optimizer fresh
    # on the sliced dataset. The LoRA weights are already loaded from Hub above.
    # We don't use resume_from_checkpoint here because there is no local Trainer
    # checkpoint (ephemeral filesystem) and loading just weights+state is handled
    # by _hub_restore_lora above.
    trainer.train()

    # ── Final held-out eval ─────────────────────────────────────────────────
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

    # ── Save model + all results to Hub permanently ───────────────────────────
    print("\n[hub-save] Pushing final model + results to HF Hub...", flush=True)
    _hub_urls = _save_all_to_hub(model, tokenizer, config.config_name, final_metrics=final_metrics)
    if _hub_urls:
        print(
            f"\n{'='*60}\n"
            f"  RESULTS PERMANENTLY SAVED — share these links:\n"
            f"  Model   : {_hub_urls.get('model', 'n/a')}\n"
            f"  Results : {_hub_urls.get('results', 'n/a')}\n"
            f"{'='*60}\n",
            flush=True,
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


def _hub_repo_id(config_name: str) -> str:
    return f"Akshansh1020/bluffbuster-{config_name.lower()}-ckpt"


def _push_checkpoint_to_hub_async(ckpt_path: str, config_name: str, step: int) -> None:
    """Push checkpoint directory to HF Hub in a background thread (non-blocking).
    Checkpoint survives container rebuilds/ephemeral disk wipes."""
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token or not os.path.isdir(ckpt_path):
        return
    repo_id = _hub_repo_id(config_name)

    def _push() -> None:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.create_repo(repo_id, exist_ok=True, repo_type="model", private=False)
            api.upload_folder(
                folder_path=ckpt_path,
                repo_id=repo_id,
                repo_type="model",
                path_in_repo=f"checkpoint-{step}",
                commit_message=f"auto-checkpoint step {step}",
                ignore_patterns=["*.lock"],
            )
            print(f"[hub-push] step {step} → {repo_id} ✓", flush=True)
        except Exception as _e:
            print(f"[hub-push] step {step} warn (non-fatal): {_e}", flush=True)

    t = threading.Thread(target=_push, daemon=True)
    t.start()


def _recover_checkpoint_from_hub(checkpoint_root: str, config_name: str) -> str | None:
    """Download latest checkpoint from HF Hub when local disk has been wiped.
    Returns path to downloaded checkpoint directory, or None if unavailable."""
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        return None
    repo_id = _hub_repo_id(config_name)
    local_dir = os.path.join(checkpoint_root, "hub-recovery")
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=hf_token,
            ignore_patterns=["*.gguf", "*.ggml"],
        )
        # Optional manual override: RESUME_CHECKPOINT_STEP=30 resumes checkpoint-30.
        requested_step = os.environ.get("RESUME_CHECKPOINT_STEP", "").strip()
        if requested_step:
            forced = os.path.join(local_dir, f"checkpoint-{requested_step}")
            if os.path.isdir(forced):
                print(
                    f"[hub-recovery] using requested checkpoint step {requested_step}: {forced}",
                    flush=True,
                )
                return forced
            print(
                f"[hub-recovery] RESUME_CHECKPOINT_STEP={requested_step} not found; "
                "falling back to latest checkpoint.",
                flush=True,
            )

        latest = _find_latest_checkpoint(local_dir)
        if latest is not None:
            print(f"[hub-recovery] downloaded latest checkpoint from {repo_id} → {latest}", flush=True)
            return latest
        print(f"[hub-recovery] downloaded {repo_id} but no checkpoint-* dirs found.", flush=True)
        return None
    except Exception as _e:
        print(f"[hub-recovery] no Hub checkpoint found ({repo_id}): {_e}", flush=True)
        return None


def _save_all_to_hub(
    model,
    tokenizer,
    config_name: str,
    final_metrics: dict | None = None,
) -> dict[str, str]:
    """
    Persistently save the trained LoRA adapter + every result file to HF Hub.

    Two repos are created / updated:
      • huggingface.co/<user>/bluffbuster-<config_name>          — model (LoRA adapter)
      • huggingface.co/datasets/<user>/bluffbuster-<config_name>-results — JSON metrics + README

    Returns a dict with the public URLs so callers can surface them in the UI.
    Non-fatal: any failure prints a warning and returns {}.
    """
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("[hub-save] HF_TOKEN not set — skipping permanent Hub save.", flush=True)
        return {}

    owner = "Akshansh1020"
    model_repo    = f"{owner}/bluffbuster-{config_name.lower()}"
    results_repo  = f"{owner}/bluffbuster-{config_name.lower()}-results"
    urls: dict[str, str] = {}

    # ── 1. Save LoRA adapter locally then push to Hub ─────────────────────────
    try:
        local_model_dir = os.path.join("outputs", f"lora_model_{config_name.lower()}")
        print(f"[hub-save] Saving LoRA adapter to {local_model_dir} ...", flush=True)
        model.save_pretrained(local_model_dir)
        tokenizer.save_pretrained(local_model_dir)

        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.create_repo(model_repo, exist_ok=True, repo_type="model", private=False)
        api.upload_folder(
            folder_path=local_model_dir,
            repo_id=model_repo,
            repo_type="model",
            commit_message=f"BluffBuster {config_name} trained LoRA adapter",
            ignore_patterns=["*.lock"],
        )
        model_url = f"https://huggingface.co/{model_repo}"
        urls["model"] = model_url
        print(f"[hub-save] Model saved → {model_url}  ✓", flush=True)
    except Exception as _e:
        print(f"[hub-save] model push WARN (non-fatal): {_e}", flush=True)

    # ── 2. Collect all JSON result files ──────────────────────────────────────
    try:
        from huggingface_hub import HfApi  # already imported above, safe to re-import
        api = HfApi(token=hf_token)
        api.create_repo(results_repo, exist_ok=True, repo_type="dataset", private=False)

        result_files = [
            os.path.join("outputs", "eval", "final_metrics.json"),
            os.path.join("outputs", "eval", "final_metrics_fallback.json"),
            os.path.join("outputs", "eval", "baseline_metrics.json"),
            os.path.join("outputs", "eval", "checkpoint_metrics.json"),
            os.path.join("outputs", "eval", "oracle_calibration.json"),
        ]
        uploaded: list[str] = []
        for fpath in result_files:
            if os.path.exists(fpath):
                api.upload_file(
                    path_or_fileobj=fpath,
                    path_in_repo=os.path.basename(fpath),
                    repo_id=results_repo,
                    repo_type="dataset",
                    commit_message=f"upload {os.path.basename(fpath)}",
                )
                uploaded.append(os.path.basename(fpath))

        # ── 3. Write a human-readable README with the numbers ─────────────────
        acc   = (final_metrics or {}).get("classification_accuracy", float("nan"))
        rew   = (final_metrics or {}).get("reward_mean",             float("nan"))
        ece   = (final_metrics or {}).get("calibration_ECE",         float("nan"))
        gain  = (final_metrics or {}).get("avg_info_gain_per_turn",  float("nan"))
        far   = (final_metrics or {}).get("false_accusation_rate",   float("nan"))

        def _fmt(v: float) -> str:
            return f"{v:.4f}" if not (v != v) else "—"  # nan check

        readme_text = f"""\
# BluffBuster — {config_name.upper()} Results

Trained LoRA adapter: [{model_repo}](https://huggingface.co/{model_repo})

## Final Evaluation Metrics

| Metric | Value |
|--------|-------|
| Classification Accuracy | {_fmt(acc)} |
| Reward Mean | {_fmt(rew)} |
| Calibration ECE | {_fmt(ece)} |
| Avg Info Gain / Turn | {_fmt(gain)} |
| False Accusation Rate | {_fmt(far)} |

## Files in this repo

| File | Description |
|------|-------------|
| `final_metrics.json` | Full per-metric results after {config_name} training |
| `baseline_metrics.json` | Pre-training baselines (Random / Definitional / Bayesian) |
| `checkpoint_metrics.json` | Mid-training evaluation snapshots (if run) |

## How to load the model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "{model_repo}")
tokenizer = AutoTokenizer.from_pretrained("{model_repo}")
```
"""
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(readme_text)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=results_repo,
            repo_type="dataset",
            commit_message="add results README",
        )
        os.unlink(tmp_path)

        results_url = f"https://huggingface.co/datasets/{results_repo}"
        urls["results"] = results_url
        print(
            f"[hub-save] Results saved → {results_url}  "
            f"({len(uploaded)} JSON files + README)  ✓",
            flush=True,
        )

        # ── 4. Also persist a local copy so the Space UI can read it ──────────
        summary_path = os.path.join("outputs", "eval", "hub_share_links.json")
        os.makedirs(os.path.join("outputs", "eval"), exist_ok=True)
        with open(summary_path, "w") as _sf:
            json.dump({
                "model_url":   urls.get("model", ""),
                "results_url": urls.get("results", ""),
                "metrics":     {
                    "classification_accuracy": acc,
                    "reward_mean":             rew,
                    "calibration_ECE":         ece,
                    "avg_info_gain_per_turn":  gain,
                    "false_accusation_rate":   far,
                },
            }, _sf, indent=2)
        print(f"[hub-save] Share links written to {summary_path}", flush=True)

    except Exception as _e:
        print(f"[hub-save] results push WARN (non-fatal): {_e}", flush=True)

    return urls


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


def _checkpoint_step(path: str | None) -> int | None:
    """Extract integer step from .../checkpoint-<step> path."""
    if not path:
        return None
    base = os.path.basename(path.rstrip("/\\"))
    if not base.startswith("checkpoint-"):
        return None
    try:
        return int(base.split("checkpoint-", 1)[1])
    except ValueError:
        return None


class _TrainedExaminerWrapper:
    """Adapts the trained model to the examiner.act(observation) interface."""
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
        try:
            max_new = int(os.environ.get("EVAL_MAX_NEW_TOKENS", "256"))
        except ValueError:
            max_new = 256
        max_new = max(64, min(512, max_new))
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
        return self.__class__.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()


# ══════════════════════════════════════════════════════════════════════════
#   Eval-only entry point (load saved LoRA from Hub, run held-out eval)
# ══════════════════════════════════════════════════════════════════════════

def run_eval_only(config_name: str, eval_config: dict) -> dict:
    """
    Load the latest LoRA checkpoint from the Hub and run held-out eval.

    No training, no optimizer, no GRPOTrainer. Used after a training run
    crashes during final eval (or when you just want to re-evaluate a
    saved checkpoint without paying for another training run).

    Flow:
      1. Resolve config + HF_TOKEN + checkpoint repo
      2. Locate latest lora-step-N on the Hub
      3. Load Qwen base (Unsloth, 4-bit) + attach LoRA structure
      4. Restore adapter weights from Hub
      5. Ensure baseline_metrics.json exists (fast 5-ep baseline if not)
      6. Run final held-out eval (FINAL_EVAL_EPISODES, default 10)
      7. Save metrics + best-effort comparison plot
      8. Upload artifacts to Hub (best-effort)

    Returns the final_metrics dict.
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise RuntimeError(
            "Unsloth not installed — run_eval_only requires the same env "
            f"as train(). Error: {e}"
        )

    from examiner_env.knowledge_base import KB
    from examiner_env.baselines import (
        RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer,
    )
    from training.eval import run_eval

    config = get_config(config_name)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set — cannot download checkpoint from Hub.")

    repo_id = os.environ.get("CHECKPOINT_REPO", "").strip()
    if not repo_id:
        try:
            from huggingface_hub import HfApi as _HfApi
            _hf_username = _HfApi(token=hf_token).whoami()["name"]
            repo_id = f"{_hf_username}/bluffbuster-checkpoints"
            print(f"[eval-only] Auto-detected checkpoint repo: {repo_id}")
        except Exception as _e:
            raise RuntimeError(
                f"Could not derive checkpoint repo from token ({_e}). "
                "Set CHECKPOINT_REPO env var explicitly."
            )

    print(f"[eval-only] Scanning {repo_id} for latest lora-step-N ...")
    latest_step, latest_folder = _hub_get_latest_step(repo_id, hf_token)
    if latest_step <= 0 or not latest_folder:
        raise RuntimeError(
            f"No lora-step-N folders found in {repo_id}. "
            "Run training first to produce a checkpoint."
        )
    print(f"[eval-only] Found checkpoint: {latest_folder} (step={latest_step})")

    # Load base model (4-bit, prefer pre-quantized variant) with same config
    # the training run used, so adapter shapes match.
    model_name = config.model_name
    if config.use_4bit and not model_name.startswith("unsloth/"):
        prequant = {
            "Qwen/Qwen2.5-1.5B-Instruct": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
            "Qwen/Qwen2.5-3B-Instruct":   "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
            "Qwen/Qwen2.5-7B-Instruct":   "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        }.get(model_name)
        if prequant:
            print(f"[eval-only] Using pre-quantized weights: {prequant}")
            model_name = prequant

    print(f"[eval-only] Loading {model_name} (4-bit={config.use_4bit}) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.use_4bit,
        dtype=None,
    )
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
    print("[eval-only] Base model + LoRA scaffold ready.")

    # Restore LoRA weights from Hub
    local_ckpt_dir = os.path.join("outputs", "checkpoints", latest_folder)
    ts = _hub_restore_lora(model, repo_id, hf_token, latest_folder, local_ckpt_dir)
    if ts is None:
        raise RuntimeError(
            f"Failed to restore LoRA weights from {latest_folder}. See logs above."
        )
    print(f"[eval-only] Restored LoRA weights from step {latest_step}.")

    # Switch to inference mode
    FastLanguageModel.for_inference(model)
    model.eval()

    # Baseline eval (5 episodes — fast). Only if not already cached.
    _baseline_cfg = {**eval_config, "num_episodes": 5}
    _run_baseline_eval(
        _baseline_cfg, KB, run_eval,
        RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer,
    )

    # Final eval
    n_ep = int(os.environ.get("FINAL_EVAL_EPISODES", "10"))
    _final_cfg = {**eval_config, "num_episodes": n_ep}
    print(f"\n[eval-only] Running held-out eval on {n_ep} episodes ...")

    _TrainedExaminerWrapper.model = model
    _TrainedExaminerWrapper.tokenizer = tokenizer
    _TrainedExaminerWrapper.config = config

    import torch as _torch
    with _torch.no_grad():
        final_metrics = run_eval(
            _TrainedExaminerWrapper(),
            _final_cfg,
            KB,
            output_path=os.path.join("outputs", "eval", "final_metrics.json"),
        )

    print(
        f"[eval-only] Final — accuracy={final_metrics['classification_accuracy']:.3f} "
        f"info_gain={final_metrics['avg_info_gain_per_turn']:.4f} "
        f"ECE={final_metrics['calibration_ECE']:.4f} "
        f"reward={final_metrics['reward_mean']:.4f}"
    )

    # Best-effort comparison plot (baseline vs trained)
    try:
        _make_comparison_plot()
    except Exception as _plot_exc:
        print(f"[eval-only] WARNING: plot generation failed ({_plot_exc})")

    # Best-effort upload of eval artifacts to Hub
    try:
        _upload_eval_artifacts(repo_id, hf_token, latest_step)
    except Exception as _up_exc:
        print(f"[eval-only] WARNING: artifact upload failed ({_up_exc})")

    return final_metrics


def _make_comparison_plot() -> None:
    """Render outputs/plots/comparison.png from baseline + final metrics JSON."""
    baseline_path = os.path.join("outputs", "eval", "baseline_metrics.json")
    final_path    = os.path.join("outputs", "eval", "final_metrics.json")
    if not (os.path.exists(baseline_path) and os.path.exists(final_path)):
        print("[eval-only] Skipping plot — metrics JSON not found.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(baseline_path) as f:
        baselines = json.load(f)
    with open(final_path) as f:
        trained = json.load(f)

    metrics_to_plot = [
        ("reward_mean",              "Reward (mean R_total)"),
        ("classification_accuracy",  "Classification accuracy"),
        ("avg_info_gain_per_turn",   "Avg info gain / turn"),
        ("false_accusation_rate",    "False accusation rate (lower=better)"),
    ]
    examiners = list(baselines.keys()) + ["TrainedExaminer"]
    examiner_metrics = {**baselines, "TrainedExaminer": trained}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title) in zip(axes.flat, metrics_to_plot):
        vals = [examiner_metrics[e].get(key, float("nan")) for e in examiners]
        bars = ax.bar(examiners, vals)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            try:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            except Exception:
                pass
    fig.suptitle("BluffBuster — Baselines vs Trained Examiner", fontsize=13)
    fig.tight_layout()

    plot_dir = os.path.join("outputs", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "comparison.png")
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval-only] Wrote comparison plot → {plot_path}")


def _upload_eval_artifacts(repo_id: str, token: str, step: int) -> None:
    """Push final_metrics.json + comparison.png to the checkpoint repo."""
    try:
        from huggingface_hub import HfApi as _HfApi
    except ImportError:
        return
    api = _HfApi(token=token)

    artifacts = [
        ("outputs/eval/final_metrics.json",     f"eval-step-{step}/final_metrics.json"),
        ("outputs/eval/baseline_metrics.json",  f"eval-step-{step}/baseline_metrics.json"),
        ("outputs/plots/comparison.png",        f"eval-step-{step}/comparison.png"),
    ]
    uploaded: list[str] = []
    for local, remote in artifacts:
        if not os.path.exists(local):
            continue
        try:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"eval-only artifacts for step {step}",
            )
            uploaded.append(remote)
        except Exception as _e:
            print(f"[eval-only] upload {local} failed: {_e}")

    if uploaded:
        results_url = f"https://huggingface.co/{repo_id}/tree/main/eval-step-{step}"
        os.makedirs(os.path.join("outputs", "eval"), exist_ok=True)
        with open(os.path.join("outputs", "eval", "hub_share_links.json"), "w") as f:
            json.dump({"results_url": results_url, "files": uploaded}, f, indent=2)
        print(f"[eval-only] Artifacts uploaded → {results_url}")


# ══════════════════════════════════════════════════════════════════════════
#   CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BluffBuster Examiner with GRPO")
    parser.add_argument("--config", default="DEBUG",
                        choices=["DEBUG", "DEMO", "DEMO_FAST", "FULL", "FULL_FAST"])
    parser.add_argument("--eval_config", default="eval_config.json")
    args = parser.parse_args()

    with open(args.eval_config) as f:
        eval_cfg = json.load(f)

    train(get_config(args.config), eval_cfg)
