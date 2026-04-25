"""
scripts/generate_plots.py — Evidence plot generation.
C2 owns.

⚠️ ALL plots are derived from real W&B / training output data.
   No synthetic data. No hardcoded values. MSR-3 blocker if violated.

Required plots (9 total):
1. reward_curve.png        — R_total mean ± std per checkpoint step
2. reward_components.png   — Small-multiples: R_acc, R_info, R_cal, R_qual, R_asym
3. accuracy_curve.png      — Classification accuracy over training steps
4. false_rates_curve.png   — False accusation + false exoneration overlaid
5. info_gain_curve.png     — avg_info_gain_per_turn over training steps
6. calibration_ece_curve.png — Calibration ECE over training steps
7. comparison_bar.png      — [4 examiners] × [accuracy + info_gain + ECE]
8. per_style_heatmap.png   — 7 styles × 10 sections accuracy heatmap
9. posterior_trace_example.png — Per-section p_t over turns (best AFTER transcript)
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

PLOT_DIR = os.path.join("outputs", "plots")
DPI = 150
STYLES_ORDER = ["K1", "K2", "K3", "F1", "F2", "F3", "F4"]
SECTIONS_ORDER = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"]
EXAMINER_ORDER = ["RandomExaminer", "DefinitionalExaminer", "BayesianHeuristicExaminer", "TrainedExaminer"]
EXAMINER_LABELS = ["Random", "Definitional", "BayesianHeuristic", "Trained"]
COLORS = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]


def _save(fig: plt.Figure, name: str) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")
    return path


# ── 1. R_total reward curve ────────────────────────────────────────────────

def plot_reward_curve(checkpoint_metrics: dict, config_name: str = "DEMO") -> str:
    steps = sorted(int(k) for k in checkpoint_metrics.keys())
    means = [checkpoint_metrics[str(s)]["reward_mean"] for s in steps]
    stds = [checkpoint_metrics[str(s)]["reward_std"] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, means, color="#2ecc71", linewidth=2, label="R_total mean")
    ax.fill_between(steps,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.25, color="#2ecc71", label="±1 std")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("R_total")
    ax.set_title(f"{config_name} Config — R_total over Training")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "reward_curve.png")


# ── 2. Per-component small-multiples ──────────────────────────────────────

def plot_reward_components(checkpoint_metrics: dict, config_name: str = "DEMO") -> str:
    steps = sorted(int(k) for k in checkpoint_metrics.keys())
    components = ["mean_R_acc", "mean_R_info", "mean_R_cal", "mean_R_qual", "mean_R_asym"]
    labels = ["R_acc", "R_info", "R_cal", "R_qual", "R_asym"]

    # R_asym is not directly stored — derive as reward_mean minus sum of others
    # For now, just plot available components (R_asym may need to be added to run_eval)
    available = [c for c in components if str(steps[0]) in checkpoint_metrics
                 and c in checkpoint_metrics[str(steps[0])]]

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 3.5), sharey=False)
    if len(available) == 1:
        axes = [axes]

    for ax, comp, label in zip(axes, available, labels):
        vals = [checkpoint_metrics[str(s)].get(comp, float("nan")) for s in steps]
        ax.plot(steps, vals, linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)

    fig.suptitle(f"{config_name} Config — Per-Component Rewards", y=1.02)
    fig.tight_layout()
    return _save(fig, "reward_components.png")


# ── 3. Accuracy curve ─────────────────────────────────────────────────────

def plot_accuracy_curve(checkpoint_metrics: dict, config_name: str = "DEMO") -> str:
    steps = sorted(int(k) for k in checkpoint_metrics.keys())
    accs = [checkpoint_metrics[str(s)]["classification_accuracy"] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, accs, color="#3498db", linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title(f"{config_name} Config — Classification Accuracy over Training")
    ax.grid(alpha=0.3)
    return _save(fig, "accuracy_curve.png")


# ── 4. False rates curve ──────────────────────────────────────────────────

def plot_false_rates_curve(checkpoint_metrics: dict, config_name: str = "DEMO") -> str:
    steps = sorted(int(k) for k in checkpoint_metrics.keys())
    fa_rates = [checkpoint_metrics[str(s)]["false_accusation_rate"] for s in steps]
    fe_rates = [checkpoint_metrics[str(s)]["false_exoneration_rate"] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, fa_rates, color="#e74c3c", linewidth=2, label="False Accusation Rate (KNOWS→FAKING)")
    ax.plot(steps, fe_rates, color="#f39c12", linewidth=2, label="False Exoneration Rate (FAKING→KNOWS)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Rate")
    ax.set_title(f"{config_name} Config — False Rates over Training")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "false_rates_curve.png")


# ── 5. Info-gain curve ────────────────────────────────────────────────────

def plot_info_gain_curve(checkpoint_metrics: dict, config_name: str = "DEMO") -> str:
    steps = sorted(int(k) for k in checkpoint_metrics.keys())
    gains = [checkpoint_metrics[str(s)]["avg_info_gain_per_turn"] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, gains, color="#9b59b6", linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("avg ΔH_t (bits)")
    ax.set_title(f"{config_name} Config — Average Information Gain per Turn over Training")
    ax.grid(alpha=0.3)
    return _save(fig, "info_gain_curve.png")


# ── 6. ECE curve ──────────────────────────────────────────────────────────

def plot_ece_curve(checkpoint_metrics: dict, config_name: str = "DEMO") -> str:
    steps = sorted(int(k) for k in checkpoint_metrics.keys())
    eces = [checkpoint_metrics[str(s)]["calibration_ECE"] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, eces, color="#1abc9c", linewidth=2)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title(f"{config_name} Config — Calibration ECE over Training")
    ax.grid(alpha=0.3)
    return _save(fig, "calibration_ece_curve.png")


# ── 7. Comparison bar chart ───────────────────────────────────────────────

def plot_comparison_bar(baseline_metrics: dict, final_metrics: dict) -> str:
    metrics_to_plot = [
        ("classification_accuracy", "Accuracy", [0, 1]),
        ("avg_info_gain_per_turn", "Avg Info Gain/Turn", None),
        ("calibration_ECE", "ECE (lower=better)", [0, 0.5]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (metric, label, ylim) in zip(axes, metrics_to_plot):
        values = []
        for name in EXAMINER_ORDER:
            if name == "TrainedExaminer":
                v = final_metrics.get("TrainedExaminer", {}).get(metric, float("nan"))
            else:
                v = baseline_metrics.get(name, {}).get(metric, float("nan"))
            values.append(v)

        bars = ax.bar(EXAMINER_LABELS, values, color=COLORS, edgecolor="white", linewidth=0.5)
        ax.set_title(label)
        ax.set_ylabel(label)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)

        # Annotate bar values
        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9,
                )

        ax.set_xticklabels(EXAMINER_LABELS, rotation=15, ha="right")

    fig.suptitle(
        "Held-Out Eval Suite — 4 Examiner Comparison\n"
        "(Trained on F1+F2, evaluated on held-out F3 + S05)",
        fontsize=12,
    )
    fig.tight_layout()
    return _save(fig, "comparison_bar.png")


# ── 8. Per-style accuracy heatmap ─────────────────────────────────────────

def plot_per_style_heatmap(final_metrics: dict) -> str:
    trained = final_metrics.get("TrainedExaminer", {})
    per_style_raw = trained.get("per_style_accuracy", {})

    matrix = np.full((len(STYLES_ORDER), len(SECTIONS_ORDER)), np.nan)
    for i, style in enumerate(STYLES_ORDER):
        for j, section in enumerate(SECTIONS_ORDER):
            key = f"{style}_{section}"
            if key in per_style_raw:
                matrix[i, j] = per_style_raw[key]
            elif style in per_style_raw:
                # Fallback: style-level average if section breakdown unavailable
                matrix[i, j] = per_style_raw[style]

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(
        matrix,
        xticklabels=SECTIONS_ORDER,
        yticklabels=STYLES_ORDER,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        mask=np.isnan(matrix),
        ax=ax,
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_title(
        "Per-Style Accuracy Heatmap — TrainedExaminer on Held-Out Eval Suite\n"
        "(NaN = style/section combo not in eval suite)"
    )
    fig.tight_layout()
    return _save(fig, "per_style_heatmap.png")


# ── 9. Posterior trace example ────────────────────────────────────────────

def plot_posterior_trace(after_transcript_path: str = "outputs/transcripts/after_transcript.json") -> str:
    with open(after_transcript_path) as f:
        transcript = json.load(f)

    trace: list[dict] = transcript.get("posterior_trace", [])
    if not trace:
        raise ValueError("after_transcript.json has no posterior_trace. Re-run select_transcripts.py.")

    sections = list(trace[0].keys())
    turns = list(range(1, len(trace) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    for s_id in sections:
        vals = [step.get(s_id, 0.5) for step in trace]
        ax.plot(turns, vals, marker="o", label=s_id, linewidth=1.5)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Prior (0.5)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Turn")
    ax.set_ylabel("p_t(s) — Posterior Belief (KNOWS)")
    ax.set_title(
        f"Posterior Belief Trace — TrainedExaminer (seed={transcript.get('episode_seed', '?')})\n"
        "Each line = one section's belief that student KNOWS the material"
    )
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, "posterior_trace_example.png")


# ── Main ──────────────────────────────────────────────────────────────────

def generate_all_plots(
    checkpoint_metrics_path: str = "outputs/eval/checkpoint_metrics.json",
    baseline_metrics_path: str = "outputs/eval/baseline_metrics.json",
    final_metrics_path: str = "outputs/eval/final_metrics.json",
    after_transcript_path: str = "outputs/transcripts/after_transcript.json",
    config_name: str = "DEMO",
) -> list[str]:
    """Generate all 9 required plots. Raises if any data file is missing."""
    print("Generating evidence plots...")

    with open(checkpoint_metrics_path) as f:
        checkpoint_metrics = json.load(f)
    with open(baseline_metrics_path) as f:
        baseline_metrics = json.load(f)
    with open(final_metrics_path) as f:
        final_metrics = json.load(f)

    paths = []
    paths.append(plot_reward_curve(checkpoint_metrics, config_name))
    paths.append(plot_reward_components(checkpoint_metrics, config_name))
    paths.append(plot_accuracy_curve(checkpoint_metrics, config_name))
    paths.append(plot_false_rates_curve(checkpoint_metrics, config_name))
    paths.append(plot_info_gain_curve(checkpoint_metrics, config_name))
    paths.append(plot_ece_curve(checkpoint_metrics, config_name))
    paths.append(plot_comparison_bar(baseline_metrics, final_metrics))
    paths.append(plot_per_style_heatmap(final_metrics))
    paths.append(plot_posterior_trace(after_transcript_path))

    print(f"\n✓ All {len(paths)} plots saved to {PLOT_DIR}/")
    return paths


if __name__ == "__main__":
    generate_all_plots()
