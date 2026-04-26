"""
scripts/compare_baselines.py — Side-by-side comparison of all examiners.

Reads whatever metric JSONs are present and produces:

    outputs/plots/four_way_comparison.png   — bar chart, 4 metrics x 4-5 examiners
    outputs/eval/comparison_table.md        — markdown table, all numeric metrics
    outputs/eval/comparison_summary.json    — machine-readable for downstream

Inputs (any subset; missing files are silently skipped):

    outputs/eval/baseline_metrics.json
        {"RandomExaminer": {...metrics...}, "DefinitionalExaminer": {...},
         "BayesianHeuristicExaminer": {...}}

    outputs/eval/dumb_baseline_metrics.json
        Single metrics dict (untrained 1.5B, no LoRA).

    outputs/eval/final_metrics.json
        Single metrics dict (GRPO-trained 1.5B, with LoRA).

Run standalone:

    python scripts/compare_baselines.py

The script never modifies any C1-owned file.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

_EVAL_DIR = os.path.join(_ROOT, "outputs", "eval")
_PLOT_DIR = os.path.join(_ROOT, "outputs", "plots")

# Order matters — left-to-right on bar chart goes from "weakest" to "strongest".
_DISPLAY_ORDER = [
    "RandomExaminer",
    "BayesianHeuristicExaminer",
    "DumbLLM (1.5B, no training)",
    "TrainedLLM (1.5B + GRPO LoRA)",
]

# Metrics shown in the bar chart. "higher_is_better=True" means up-bar is good.
_HEADLINE_METRICS = [
    ("avg_info_gain_per_turn", "Avg Info Gain / Turn",   True),
    ("classification_accuracy", "Classification Accuracy", True),
    ("false_accusation_rate",   "False Accusation Rate",   False),
    ("reward_mean",             "Reward Mean",             True),
]

# Full metrics list shown in the markdown table.
_TABLE_METRICS = [
    "classification_accuracy",
    "avg_info_gain_per_turn",
    "false_accusation_rate",
    "false_exoneration_rate",
    "terminal_posterior_correctness",
    "calibration_ECE",
    "calibration_brier",
    "reward_mean",
    "reward_std",
    "parse_failure_rate",
    "avg_turns_to_classify",
]


def _safe_load(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[compare] Failed to read {path}: {e!r}")
        return None


def _collect() -> dict[str, dict]:
    """Return {display_name: metrics_dict} for every available source."""
    out: dict[str, dict] = {}

    bm = _safe_load(os.path.join(_EVAL_DIR, "baseline_metrics.json"))
    if bm:
        if "RandomExaminer" in bm:
            out["RandomExaminer"] = bm["RandomExaminer"]
        if "BayesianHeuristicExaminer" in bm:
            out["BayesianHeuristicExaminer"] = bm["BayesianHeuristicExaminer"]
        # Definitional intentionally not shown — overlaps with RandomExaminer
        # in storytelling and clutters the chart.

    dumb = _safe_load(os.path.join(_EVAL_DIR, "dumb_baseline_metrics.json"))
    if dumb:
        out["DumbLLM (1.5B, no training)"] = dumb

    trained = _safe_load(os.path.join(_EVAL_DIR, "final_metrics.json"))
    if trained:
        out["TrainedLLM (1.5B + GRPO LoRA)"] = trained

    return out


def _format_value(v: Any, fmt: str = "{:+.4f}") -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "nan"
    try:
        return fmt.format(v)
    except (TypeError, ValueError):
        return str(v)


def _build_table(collected: dict[str, dict]) -> str:
    cols = [c for c in _DISPLAY_ORDER if c in collected]
    if not cols:
        return "*(No metric files found yet.)*"

    lines = []
    header = "| Metric | " + " | ".join(cols) + " |"
    sep    = "|" + "---|" * (len(cols) + 1)
    lines.append(header)
    lines.append(sep)

    for m in _TABLE_METRICS:
        row = [m]
        for c in cols:
            v = collected[c].get(m)
            fmt = "{:+.4f}" if m == "reward_mean" else "{:.4f}"
            row.append(_format_value(v, fmt))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _build_bar_chart(collected: dict[str, dict], output_path: str) -> str:
    cols = [c for c in _DISPLAY_ORDER if c in collected]
    if not cols:
        return ""

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    short_names = {
        "RandomExaminer":                    "Random",
        "BayesianHeuristicExaminer":         "Bayesian\nHeuristic",
        "DumbLLM (1.5B, no training)":       "Dumb LLM\n(1.5B)",
        "TrainedLLM (1.5B + GRPO LoRA)":     "Trained\n(GRPO)",
    }

    # Color: trained = green, dumb-LLM = orange, classical = gray-blue.
    color_map = {
        "RandomExaminer":                    "#9aa5b1",
        "BayesianHeuristicExaminer":         "#5e7dd0",
        "DumbLLM (1.5B, no training)":       "#f0a868",
        "TrainedLLM (1.5B + GRPO LoRA)":     "#3aa755",
    }

    for ax, (key, label, higher_is_better) in zip(axes, _HEADLINE_METRICS):
        values = [collected[c].get(key, np.nan) for c in cols]
        labels = [short_names.get(c, c) for c in cols]
        colors = [color_map.get(c, "#888") for c in cols]

        bars = ax.bar(range(len(cols)), values, color=colors, edgecolor="black")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        arrow = "↑ better" if higher_is_better else "↓ better"
        ax.set_ylabel(arrow, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for bar, v in zip(bars, values):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            ax.annotate(
                f"{v:+.3f}" if key == "reward_mean" else f"{v:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3 if v >= 0 else -12),
                textcoords="offset points",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, fontweight="bold",
            )

    fig.suptitle("BluffBuster — Examiner Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_comparison() -> dict:
    """
    End-to-end: read every available metric JSON, write the markdown table,
    write the bar chart PNG, write a summary JSON. Returns {"table": str,
    "plot_path": str, "n_sources": int, "sources": list[str]}.
    """
    os.makedirs(_PLOT_DIR, exist_ok=True)
    os.makedirs(_EVAL_DIR, exist_ok=True)

    collected = _collect()

    table_md = _build_table(collected)
    table_path = os.path.join(_EVAL_DIR, "comparison_table.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# BluffBuster — Examiner Comparison\n\n")
        f.write(table_md)
        f.write("\n")

    plot_path = _build_bar_chart(
        collected, os.path.join(_PLOT_DIR, "four_way_comparison.png")
    )

    summary_path = os.path.join(_EVAL_DIR, "comparison_summary.json")
    summary = {
        "sources_present": list(collected.keys()),
        "metrics": {
            name: {m: collected[name].get(m) for m in _TABLE_METRICS}
            for name in collected
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("EXAMINER COMPARISON")
    print("=" * 60)
    print(f"Sources found: {len(collected)} / 4")
    for name in _DISPLAY_ORDER:
        marker = "OK" if name in collected else "..MISSING.."
        print(f"  [{marker}] {name}")
    print("\n" + table_md)
    print("\nArtifacts:")
    print(f"  table : {table_path}")
    print(f"  plot  : {plot_path}")
    print(f"  json  : {summary_path}")

    return {
        "table": table_md,
        "plot_path": plot_path,
        "table_path": table_path,
        "summary_path": summary_path,
        "n_sources": len(collected),
        "sources": list(collected.keys()),
    }


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    build_comparison()
