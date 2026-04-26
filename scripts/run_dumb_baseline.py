"""
scripts/run_dumb_baseline.py — Evaluate the UNTRAINED Qwen2.5-1.5B-Instruct
on the frozen eval suite. This is the LLM "before" baseline.

Why this script exists
----------------------
The three baselines in `examiner_env/baselines.py` are non-LLM (Random,
Definitional, BayesianHeuristic). To make the "trained model is meaningfully
smarter" claim airtight, we also need a baseline that is the **same LLM,
same prompt, same env, just no GRPO training applied**. That is what this
script produces.

Side-by-side comparison after a FAST run:
    outputs/eval/dumb_baseline_metrics.json   <- this script (no LoRA)
    outputs/eval/final_metrics.json           <- training/train_grpo.py
                                                 (GRPO-trained LoRA on
                                                 the same base model)

Both share the schema produced by training.eval.run_eval, so any of the 16
metrics is directly diffable.

Speed defaults (under 1 hr on A10G)
-----------------------------------
    --limit 15           # 15 seeds (matches FAST_CONFIG.eval_episodes)
    --max_new_tokens 64  # 1 valid ASK/CLASSIFY JSON fits in 64 tokens
    --max_seq_length 1024

Estimated wall-clock on A10G:  model load ~1.5 min  +  ~1.0 s/turn  *
                               15 seeds * ~4 turns = ~3 min eval
                               total ~5 min.

Usage
-----
    # default — fast 15-seed run
    python scripts/run_dumb_baseline.py

    # full 50-seed run (still ~10 min on A10G)
    python scripts/run_dumb_baseline.py --limit 50

    # different base model (must equal model_name from training config)
    python scripts/run_dumb_baseline.py --model Qwen/Qwen2.5-3B-Instruct

HF Hub upload (auto)
--------------------
If `HF_TOKEN` env var is set, the metrics JSON is also pushed to the same
checkpoint repo used by training (`CHECKPOINT_REPO`, default
`Samarth1401/bluffbuster-checkpoints`) under the path
`dumb_baseline/dumb_baseline_metrics.json`. This means the result survives
HF Spaces ephemeral filesystem restarts.

Owner: C2. Reads `eval_config.json` and `examiner_env/` (read-only) — does
NOT modify any C1-owned file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load_eval_config(path: str, limit: int | None) -> dict:
    """Load eval_config.json and optionally truncate the seed list.

    NEVER reorder or modify the seeds — only take a prefix when --limit is
    set, so the result is a strict subset of the trained model's eval set
    and metrics remain seed-matched.
    """
    with open(path, "r") as f:
        cfg = json.load(f)
    if limit is not None and limit > 0:
        original = cfg["seeds"]
        if limit > len(original):
            print(
                f"[dumb_baseline] --limit={limit} exceeds available seeds "
                f"({len(original)}). Using all seeds."
            )
        else:
            cfg = {**cfg, "seeds": original[:limit]}
            print(
                f"[dumb_baseline] Using first {limit} seeds (subset of full "
                f"eval suite, identical seed values for fair comparison)."
            )
    return cfg


def _print_summary(metrics: dict, output_path: str) -> None:
    print("\n" + "=" * 60)
    print("UNTRAINED 1.5B BASELINE — frozen eval suite")
    print("=" * 60)
    headline = [
        ("classification_accuracy", "{:.4f}"),
        ("avg_info_gain_per_turn",  "{:.4f}"),
        ("false_accusation_rate",   "{:.4f}"),
        ("false_exoneration_rate",  "{:.4f}"),
        ("reward_mean",             "{:+.4f}"),
        ("reward_std",              "{:.4f}"),
        ("calibration_ECE",         "{:.4f}"),
        ("calibration_brier",       "{:.4f}"),
        ("parse_failure_rate",      "{:.4f}"),
    ]
    width = max(len(k) for k, _ in headline)
    for k, fmt in headline:
        print(f"  {k:<{width}} : {fmt.format(metrics[k])}")
    print("=" * 60)
    print(f"Full metrics -> {output_path}")


def _maybe_upload_to_hub(local_path: str) -> None:
    """Best-effort upload of the metrics JSON to the checkpoint Hub repo.

    Survives Space restarts (ephemeral filesystem). Silent no-op if HF_TOKEN
    is not set — local file is the source of truth.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[dumb_baseline] HF_TOKEN not set — skipping Hub upload "
              "(local file is canonical).")
        return
    repo_id = os.environ.get(
        "CHECKPOINT_REPO", "Samarth1401/bluffbuster-checkpoints"
    )
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=token)
        try:
            create_repo(repo_id=repo_id, token=token, private=True,
                        repo_type="model", exist_ok=True)
        except Exception:
            pass  # repo likely already exists
        path_in_repo = "dumb_baseline/dumb_baseline_metrics.json"
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="dumb baseline metrics (untrained 1.5B)",
        )
        print(f"[dumb_baseline] Uploaded -> "
              f"https://huggingface.co/{repo_id}/blob/main/{path_in_repo}")
    except Exception as e:
        print(f"[dumb_baseline] Hub upload failed ({e!r}); "
              "local file still saved.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run untrained 1.5B baseline on the frozen eval suite."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base HF model id (must match the training config's model_name).",
    )
    parser.add_argument(
        "--eval_config",
        default="eval_config.json",
        help="Path to eval_config.json. Seeds are NEVER reordered.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Same default as FAST_CONFIG.max_seq_length.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Generation cap per examiner turn. 64 fits one valid JSON.",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit loading (use fp16/bf16). Default: 4-bit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help=(
            "Use only the first N seeds. Default 15 matches "
            "FAST_CONFIG.eval_episodes. Set 50 for the full eval suite."
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/eval/dumb_baseline_metrics.json",
        help="Where to write the metrics JSON.",
    )
    parser.add_argument(
        "--no_compare",
        action="store_true",
        help="Skip auto-generating the comparison plot/table at the end.",
    )
    args = parser.parse_args()

    from training.dumb_examiner import load_dumb_examiner
    from training.eval import run_eval
    from examiner_env.knowledge_base import KB

    eval_cfg = _load_eval_config(args.eval_config, args.limit)
    print(
        f"[dumb_baseline] eval suite: {len(eval_cfg['seeds'])} seeds "
        f"(first 5 = {eval_cfg['seeds'][:5]})"
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    t0 = time.time()
    examiner = load_dumb_examiner(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        use_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"[dumb_baseline] Model ready in {time.time() - t0:.1f}s. "
          "Running eval (no LoRA, no training)...")

    t1 = time.time()
    metrics = run_eval(examiner, eval_cfg, KB, output_path=args.output)
    print(f"[dumb_baseline] Eval finished in {time.time() - t1:.1f}s.")

    _print_summary(metrics, args.output)
    _maybe_upload_to_hub(args.output)

    if not args.no_compare:
        try:
            from scripts.compare_baselines import build_comparison
            print("\n[dumb_baseline] Generating 4-way comparison artifacts...")
            build_comparison()
        except Exception as e:
            print(f"[dumb_baseline] compare_baselines skipped ({e!r}). "
                  "Run `python scripts/compare_baselines.py` manually.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
