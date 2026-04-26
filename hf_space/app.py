"""
hf_space/app.py — Gradio 4-tab demo for The Examiner (BluffBuster).
C2 owns.

Tabs:
  1. Live Episode     — user triggers episode; live posterior trace; full reward breakdown
  2. Baseline vs Trained — same seed, 4 examiners side by side
  3. Training Evidence — real plots from W&B training run
  4. Environment Details — reward spec, style table, action schema

⚠️ BEFORE DEPLOYMENT:
  - Confirm plots exist in outputs/plots/
  - Test all 4 tabs in Gradio 4 (not 3)
  - Test in incognito before marking MSR-5 done
  - Check Space logs for silent startup failures
"""

from __future__ import annotations

# Set before torch/unsloth load so @torch.compile becomes a no-op globally.
# Prevents Unsloth compiled GRPOTrainer shape-trace errors with environment_factory.
import os as _env_os
_env_os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import json
import os
import sys

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Narrative (verbatim from PROJECT IDENTITY) ──
THREE_SENTENCE_NARRATIVE = (
    "Most AI benchmarks reward getting the right answer — but almost none reward asking "
    "the right question. The Examiner is an adversarial RL environment where an examiner "
    "agent learns, through information-gain reward shaping and calibrated terminal scoring, "
    "to design questions that expose confident bluffing across multiple deceptive student "
    "styles. We train a language model examiner using GRPO and demonstrate measurable "
    "improvement over definitional and random baselines on held-out student styles and "
    "unseen topic sections, with reward decomposition that judges can audit live."
)

ENV_DETAILS_MARKDOWN = """
## Student Simulator Family (7 styles)

| Style | Knowledge | Mech. Cue Rate | Misc. Rate | Special Behavior |
|-------|-----------|---------------|------------|-----------------|
| K1    | KNOWS     | 0.85          | 0.05       | Mechanistic, causal, concrete |
| K2    | KNOWS     | 0.55          | 0.05       | Concise and correct |
| K3    | KNOWS     | 0.65          | 0.08       | Hedges appropriately |
| F1    | FAKING    | 0.15          | 0.30       | Collapses under "why/how" probes |
| F2    | FAKING    | 0.20          | 0.25       | Mirrors jargon from question |
| F3    | FAKING    | 0.10          | 0.20       | Pivots to adjacent topics |
| F4    | FAKING    | 0.05          | 0.40       | Very confident, zero specificity |

**Probe modulation:** A "why/how/mechanism" probe to a FAKING student halves their mechanism cue rate and increases misconception rate by 50%.

## Action Schema

**Ask:**
```json
{"action_type": "ask", "section_id": "S01", "question_text": "Why does momentum help convergence?"}
```

**Classify (terminates episode):**
```json
{"action_type": "classify", "classifications": {"S01": "KNOWS", "S02": "FAKING", ...}}
```

## Reward Function

| Component | Range | Purpose |
|-----------|-------|---------|
| R_acc | [−1, +1] | Classification accuracy |
| R_asym | [−0.5, 0] | Asymmetric error costs (false accusations cost more) |
| R_cal | [−0.4, +0.4] | Calibration: rewards confident-correct |
| R_eff | [0, +0.20] | Efficiency: fast correct classification |
| R_cov | [−0.35, 0] | Coverage: all 10 sections classified |
| R_info | [0, +0.40] | Information gain: potential-based ΔH_t |
| R_qual | [0, +0.10] | Question quality (mechanism/specificity/edge-case) |
| R_div | [0, +0.05] | Diversity across sections |
| Penalties | ≤ 0 | Malformed (−0.20), repetition (−0.10), invalid section (−0.10) |
| **R_total** | **[−2.05, +1.95]** | **Theoretical bounds asserted at runtime** |

## Honest Caveats

- Our simulator uses a controlled family of seven scripted styles, not real human experts.
- We use a KB-grounded posterior oracle for reward shaping only — not as ground truth.
- Results on held-out styles are reported honestly; worst per-style cells disclosed.
"""

REWARD_COMPONENT_KEYS = [
    "R_acc", "R_asym", "R_cal", "R_eff", "R_cov",
    "R_info", "R_qual", "R_div", "P_malformed", "R_total",
]

# Resolve PLOT_DIR for both local dev and HF Space deployment.
# When deployed to HF Space the Space root contains app.py directly, so
# plots live at <space_root>/outputs/plots/.
# When running from the repo root, plots live at outputs/plots/ (one level up
# from hf_space/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_PLOT_DIRS = [
    os.path.join(_THIS_DIR, "outputs", "plots"),          # HF Space at root
    os.path.join(_THIS_DIR, "..", "outputs", "plots"),    # local dev from repo root
]
PLOT_DIR = next(
    (d for d in _CANDIDATE_PLOT_DIRS if os.path.isdir(d)),
    _CANDIDATE_PLOT_DIRS[0],  # default to Space-style even if not yet created
)


# ──────────────────────────────────────────────────────────
# Helper: load env + examiners (lazy, cached at module level)
# ──────────────────────────────────────────────────────────

_env = None
_kb = None
_baselines = {}
_trained = None


def _get_env_and_baselines():
    global _env, _kb, _baselines, _trained
    if _env is not None:
        return _env, _kb, _baselines, _trained

    try:
        from examiner_env.environment import ExaminerEnv
        from examiner_env.knowledge_base import KB
        from examiner_env.baselines import (
            RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer
        )
        _kb = KB
        _env = ExaminerEnv(kb=KB, config=None)
        _baselines = {
            "Random": RandomExaminer(),
            "Definitional": DefinitionalExaminer(),
            "BayesianHeuristic": BayesianHeuristicExaminer(KB),
        }
        # Trained model: loaded from HF Hub after training
        # Placeholder until model is pushed to Hub
        _trained = None
    except ImportError as e:
        print(f"WARNING: examiner_env not available: {e}")
    return _env, _kb, _baselines, _trained


# ──────────────────────────────────────────────────────────
# Tab 1: Live Episode
# ──────────────────────────────────────────────────────────

def _make_posterior_figure(posterior_trace: list) -> plt.Figure:
    """Build a matplotlib figure from posterior_trace list[dict[section_id, float]]."""
    fig, ax = plt.subplots(figsize=(9, 4))
    if not posterior_trace:
        ax.text(0.5, 0.5, "No posterior trace available", ha="center", va="center")
        ax.set_axis_off()
        return fig

    sections = list(posterior_trace[0].keys())
    turns = list(range(1, len(posterior_trace) + 1))
    for s_id in sections:
        vals = [step.get(s_id, 0.5) for step in posterior_trace]
        ax.plot(turns, vals, marker="o", linewidth=1.8, label=s_id)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Prior (0.5)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Turn")
    ax.set_ylabel("p_t(s) — P(KNOWS)")
    ax.set_title("Per-Section Posterior Belief Trace")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def run_live_episode(seed: int, examiner_name: str):
    env, kb, baselines, trained = _get_env_and_baselines()

    _empty_fig = _make_posterior_figure([])

    if env is None:
        return [], _empty_fig, {}, "Environment not available."

    examiner = baselines.get(examiner_name) or trained
    if examiner is None:
        return [], _empty_fig, {}, f"Examiner '{examiner_name}' not available."

    obs, _ = env.reset(seed=int(seed))
    done = False
    dialogue_rows = []
    step = 0

    if hasattr(examiner, "reset"):
        examiner.reset()

    while not done:
        action_text = examiner.act(obs)
        obs, reward, terminated, truncated, info = env.step(action_text)
        done = terminated or truncated

        hist = obs.get("dialogue_history", [])
        if hist:
            last = hist[-1]
            dialogue_rows.append([
                step + 1,
                last.get("section_id", ""),
                last.get("question", ""),
                last.get("response", ""),
            ])
        step += 1

    bd = info.get("reward_breakdown")
    true_labels = info.get("true_labels", {})

    posterior_fig = _make_posterior_figure(bd.posterior_trace if bd and bd.posterior_trace else [])

    reward_display = {}
    if bd:
        reward_display = {
            "R_acc": round(bd.R_acc, 4),
            "R_asym": round(bd.R_asym, 4),
            "R_cal": round(bd.R_cal, 4),
            "R_eff": round(bd.R_eff, 4),
            "R_cov": round(bd.R_cov, 4),
            "R_info": round(bd.R_info, 4),
            "R_qual": round(bd.R_qual, 4),
            "R_div": round(bd.R_div, 4),
            "P_malformed": round(bd.P_malformed, 4),
            "P_repetition": round(bd.P_repetition, 4),
            "P_invalid_sec": round(bd.P_invalid_sec, 4),
            "R_total": round(bd.R_total, 4),
        }

    ground_truth_md = "### Ground Truth (revealed after classify)\n" + "\n".join(
        f"- **{s}**: {label}" for s, label in sorted(true_labels.items())
    )

    return dialogue_rows, posterior_fig, reward_display, ground_truth_md


# ──────────────────────────────────────────────────────────
# Tab 2: Baseline vs Trained Comparison
# ──────────────────────────────────────────────────────────

def run_comparison(seed: int):
    env, kb, baselines, trained = _get_env_and_baselines()
    if env is None:
        return [["Environment not available", "", "", "", ""]]

    examiner_map = {**baselines}
    if trained:
        examiner_map["Trained"] = trained

    rows = []
    for name, examiner in examiner_map.items():
        obs, _ = env.reset(seed=int(seed))
        done = False
        questions_asked = []
        if hasattr(examiner, "reset"):
            examiner.reset()

        while not done:
            action_text = examiner.act(obs)
            obs, reward, terminated, truncated, info = env.step(action_text)
            done = terminated or truncated
            hist = obs.get("dialogue_history", [])
            if hist:
                last = hist[-1]
                questions_asked.append(f"[{last.get('section_id','')}] {last.get('question','')}")

        bd = info.get("reward_breakdown")
        true_labels = info.get("true_labels", {})
        classifications = info.get("classifications", {})
        correct = classifications == true_labels

        rows.append([
            name,
            "\n".join(questions_asked[:3]) + ("..." if len(questions_asked) > 3 else ""),
            "✓ CORRECT" if correct else "✗ WRONG",
            f"{bd.R_total:.3f}" if bd else "N/A",
            f"{bd.R_info:.3f}" if bd else "N/A",
        ])

    return rows


# ──────────────────────────────────────────────────────────
# Build Gradio app
# ──────────────────────────────────────────────────────────

def create_app():
    with gr.Blocks(title="The Examiner — BluffBuster") as app:

        gr.Markdown(f"# 🧐 The Examiner — BluffBuster\n\n> {THREE_SENTENCE_NARRATIVE}")

        # ── Tab 1: Live Episode ──
        with gr.Tab("🔴 Live Episode"):
            gr.Markdown(
                "Run a single episode. Watch the **posterior belief trace** — "
                "each line shows how confident the examiner is about each section.\n\n"
                "Good questions move the needle; bad ones don't."
            )
            with gr.Row():
                seed_input_1 = gr.Number(value=1000, label="Episode Seed", precision=0)
                examiner_dropdown = gr.Dropdown(
                    choices=["Definitional", "BayesianHeuristic", "Trained"],
                    value="Definitional",
                    label="Examiner",
                )
                run_btn = gr.Button("▶ Run Episode", variant="primary")

            dialogue_table = gr.Dataframe(
                headers=["Turn", "Section", "Question Asked", "Student Response"],
                label="Dialogue",
                wrap=True,
            )
            posterior_plot = gr.Plot(label="Per-Section Belief p_t(s) — 0.5=uncertain, 1.0=confident KNOWS")
            info_gain_display = gr.JSON(label="Reward Breakdown")
            ground_truth_display = gr.Markdown(label="Ground Truth (revealed after classify)")

            run_btn.click(
                fn=run_live_episode,
                inputs=[seed_input_1, examiner_dropdown],
                outputs=[dialogue_table, posterior_plot, info_gain_display, ground_truth_display],
            )

        # ── Tab 2: Baseline vs Trained ──
        with gr.Tab("⚖️ Baseline vs Trained"):
            gr.Markdown(
                "Same episode seed through **4 examiners** side by side.\n"
                "Observe: the Trained examiner asks *why/how* probes earlier; "
                "Definitional asks surface questions."
            )
            with gr.Row():
                seed_input_2 = gr.Number(value=1000, label="Episode Seed", precision=0)
                compare_btn = gr.Button("▶ Run Comparison", variant="primary")

            comparison_table = gr.Dataframe(
                headers=["Examiner", "Questions Asked (first 3)", "Correct?", "R_total", "R_info"],
                label="4-Examiner Comparison",
                wrap=True,
            )
            compare_btn.click(
                fn=run_comparison,
                inputs=[seed_input_2],
                outputs=[comparison_table],
            )

        # ── Tab 3: Training Evidence ──
        with gr.Tab("📊 Training Evidence"):
            gr.Markdown(
                "**All plots below are generated from real W&B training run data. "
                "Not mocked. Not placeholders. (MSR-3)**"
            )

            with gr.Row():
                gr.Image(
                    value=os.path.join(PLOT_DIR, "reward_curve.png")
                    if os.path.exists(os.path.join(PLOT_DIR, "reward_curve.png")) else None,
                    label="R_total over Training",
                )
                gr.Image(
                    value=os.path.join(PLOT_DIR, "accuracy_curve.png")
                    if os.path.exists(os.path.join(PLOT_DIR, "accuracy_curve.png")) else None,
                    label="Accuracy over Training",
                )

            with gr.Row():
                gr.Image(
                    value=os.path.join(PLOT_DIR, "comparison_bar.png")
                    if os.path.exists(os.path.join(PLOT_DIR, "comparison_bar.png")) else None,
                    label="4-Examiner Comparison (Held-Out Eval)",
                )
                gr.Image(
                    value=os.path.join(PLOT_DIR, "per_style_heatmap.png")
                    if os.path.exists(os.path.join(PLOT_DIR, "per_style_heatmap.png")) else None,
                    label="Per-Style Accuracy Heatmap",
                )

            with gr.Row():
                gr.Image(
                    value=os.path.join(PLOT_DIR, "info_gain_curve.png")
                    if os.path.exists(os.path.join(PLOT_DIR, "info_gain_curve.png")) else None,
                    label="avg Info Gain/Turn over Training",
                )
                gr.Image(
                    value=os.path.join(PLOT_DIR, "calibration_ece_curve.png")
                    if os.path.exists(os.path.join(PLOT_DIR, "calibration_ece_curve.png")) else None,
                    label="Calibration ECE over Training",
                )

            gr.Image(
                value=os.path.join(PLOT_DIR, "posterior_trace_example.png")
                if os.path.exists(os.path.join(PLOT_DIR, "posterior_trace_example.png")) else None,
                label="Posterior Trace Example — Best AFTER Transcript",
            )

        # ── Tab 4: Environment Details ──
        with gr.Tab("🔬 Environment Details"):
            gr.Markdown(ENV_DETAILS_MARKDOWN)

        # ── Tab 5: Training Launcher ──
        with gr.Tab("🏋️ Train (GPU)"):
            gr.Markdown(
                "## GRPO Training Launcher\n"
                "Runs training on this Space's GPU using HF credits.\n\n"
                "- **DEBUG** — 20 episodes, Qwen2.5-1.5B, quick smoke test.\n"
                "- **DEMO** — 100 episodes, Qwen2.5-7B, submission gate / evidence run.\n"
                "- **FULL** — 500 episodes, all 10 sections, main training run "
                "(optimizer steps ≈ `500 / batch_size` → **~250 steps** with `batch_size=2`). "
                "Use a strong GPU (A100) and expect many hours.\n"
                "- **Safety** — checkpoints are saved every 10 steps and relaunch "
                "auto-resumes from the latest `outputs/checkpoints/checkpoint-*`.\n"
                "- To force a fresh run, delete `outputs/checkpoints/` first.\n\n"
                "> Credentials are read from Space secrets (already set). "
                "Pick a config and click Launch."
            )
            train_config_choice = gr.Radio(
                choices=["DEBUG", "DEMO", "FULL", "FAST"],
                value="FAST",
                label="Training preset (FAST = 1.5B, 350 steps, ~70 min on A100 — recommended for tight budgets)",
                elem_id="train_config_preset",
            )
            train_btn = gr.Button("🚀 Launch Training", variant="primary", size="lg")
            eval_only_btn = gr.Button(
                "⚡ Eval Only (load saved model from Hub — no training)",
                variant="secondary",
                size="lg",
            )
            train_output = gr.Textbox(
                label="Training / Eval Log",
                lines=20,
                interactive=False,
                placeholder="Output will appear here...",
            )

            def run_training_on_space(config_name: str) -> str:
                import traceback
                wandb_key = os.environ.get("WANDB_API_KEY", "")
                hf_token = os.environ.get("HF_TOKEN", "")
                if not wandb_key:
                    return "ERROR: WANDB_API_KEY secret not set in this Space."
                if not hf_token:
                    return "ERROR: HF_TOKEN secret not set in this Space."
                try:
                    import json
                    from training.config import get_config
                    from training.train_grpo import train
                    from examiner_env.calibration import run_calibration
                    from examiner_env.knowledge_base import KB

                    os.environ["WANDB_API_KEY"] = wandb_key
                    os.environ["HF_TOKEN"] = hf_token
                    # Keep final eval reliable but bounded for hackathon time budget.
                    # Train code still supports full 50-episode final eval when unset.
                    if config_name == "DEMO":
                        os.environ.setdefault("FINAL_EVAL_EPISODES", "10")
                    elif config_name == "FULL":
                        os.environ.setdefault("FINAL_EVAL_EPISODES", "30")
                    elif config_name == "FAST":
                        # 1.5B is fast at inference -> 15 eval eps is cheap.
                        os.environ.setdefault("FINAL_EVAL_EPISODES", "15")

                    for d in ["outputs/eval", "outputs/plots", "outputs/transcripts", "outputs/checkpoints"]:
                        os.makedirs(d, exist_ok=True)

                    cal_path = "outputs/eval/oracle_calibration.json"
                    if not os.path.exists(cal_path):
                        run_calibration(KB, n_episodes=200, output_path=cal_path)

                    with open(cal_path) as f:
                        cal = json.load(f)
                    brier = cal["calibration_metrics"]["mean_brier"]
                    if brier > 0.18:
                        return f"ERROR: Oracle Brier={brier:.4f} > 0.18. Recalibrate."

                    with open("eval_config.json") as f:
                        eval_config = json.load(f)

                    config = get_config(config_name)
                    final_metrics = train(config, eval_config)

                    acc = final_metrics.get("classification_accuracy", float("nan"))
                    gain = final_metrics.get("avg_info_gain_per_turn", float("nan"))
                    ece = final_metrics.get("calibration_ECE", float("nan"))
                    r_mean = final_metrics.get("reward_mean", float("nan"))
                    far   = final_metrics.get("false_accusation_rate", float("nan"))

                    # Read Hub share links written by _save_all_to_hub
                    links_path = os.path.join("outputs", "eval", "hub_share_links.json")
                    model_url   = ""
                    results_url = ""
                    if os.path.exists(links_path):
                        try:
                            with open(links_path) as _lf:
                                _links = json.load(_lf)
                            model_url   = _links.get("model_url", "")
                            results_url = _links.get("results_url", "")
                        except Exception:
                            pass

                    share_block = (
                        f"\n{'='*55}\n"
                        f"  PERMANENT SHARE LINKS (survive rebuilds):\n"
                        f"  🤗 Model   : {model_url or 'see Space logs'}\n"
                        f"  📊 Results : {results_url or 'see Space logs'}\n"
                        f"{'='*55}\n"
                    ) if (model_url or results_url) else (
                        "\n  (Hub save running in background — check Space logs for links)\n"
                    )

                    return (
                        f"✅ Training complete ({config_name})\n"
                        f"  Classification accuracy (held-out): {acc:.3f}\n"
                        f"  Avg info gain / turn:               {gain:.4f}\n"
                        f"  Calibration ECE:                    {ece:.4f}\n"
                        f"  Mean R_total:                       {r_mean:.4f}\n"
                        f"  False accusation rate:              {far:.4f}\n"
                        f"{share_block}"
                        f"W&B: https://wandb.ai (project: bluffbuster-examiner)\n"
                        f"All JSON metrics → outputs/eval/"
                    )
                except Exception:
                    return f"ERROR:\n{traceback.format_exc()}"

            def run_eval_only_on_space(config_name: str) -> str:
                """Load saved LoRA from Hub and run eval — zero training steps."""
                import traceback
                hf_token = os.environ.get("HF_TOKEN", "")
                if not hf_token:
                    return "ERROR: HF_TOKEN secret not set in this Space."
                try:
                    from training.config import get_config
                    from training.train_grpo import run_eval_only

                    os.environ["HF_TOKEN"] = hf_token
                    # Cap eval at 10 episodes for speed (~20 min on A100)
                    os.environ["FINAL_EVAL_EPISODES"] = "10"

                    for d in ["outputs/eval", "outputs/plots"]:
                        os.makedirs(d, exist_ok=True)

                    with open("eval_config.json") as f:
                        eval_config = json.load(f)

                    print(f"[app] Starting eval-only for {config_name}...", flush=True)
                    final_metrics = run_eval_only(config_name, eval_config)

                    acc   = final_metrics.get("classification_accuracy", float("nan"))
                    gain  = final_metrics.get("avg_info_gain_per_turn", float("nan"))
                    ece   = final_metrics.get("calibration_ECE", float("nan"))
                    r_mean = final_metrics.get("reward_mean", float("nan"))
                    far   = final_metrics.get("false_accusation_rate", float("nan"))

                    links_path = os.path.join("outputs", "eval", "hub_share_links.json")
                    results_url = ""
                    if os.path.exists(links_path):
                        try:
                            with open(links_path) as _lf:
                                _links = json.load(_lf)
                            results_url = _links.get("results_url", "")
                        except Exception:
                            pass

                    return (
                        f"✅ Eval-only complete ({config_name}) — NO retraining\n"
                        f"  Classification accuracy : {acc:.3f}\n"
                        f"  Avg info gain / turn   : {gain:.4f}\n"
                        f"  Calibration ECE        : {ece:.4f}\n"
                        f"  Mean R_total           : {r_mean:.4f}\n"
                        f"  False accusation rate  : {far:.4f}\n\n"
                        f"{'='*55}\n"
                        f"  Results + plot → {results_url or 'outputs/eval/'}\n"
                        f"{'='*55}"
                    )
                except Exception:
                    return f"ERROR (eval-only):\n{traceback.format_exc()}"

            train_btn.click(
                fn=run_training_on_space,
                inputs=[train_config_choice],
                outputs=[train_output],
            )

            eval_only_btn.click(
                fn=run_eval_only_on_space,
                inputs=[train_config_choice],
                outputs=[train_output],
            )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(theme=gr.themes.Soft())
