"""
hf_space/train_space.py — ZeroGPU / persistent GPU training launcher.

Deploy to a HuggingFace Space with GPU hardware (A10G / A100) to use
HF credits for training the BluffBuster Examiner.

Space setup:
  SDK: Gradio
  Hardware: A10G or A100 (ZeroGPU for zero-cost burst, or persistent GPU for long runs)
  requirements.txt: unsloth[colab-new] trl>=0.9.0 wandb spaces pydantic>=2.0 datasets

Pattern from TRL OpenEnv + ZeroGPU:
  https://huggingface.co/docs/trl/en/openenv
  https://huggingface.co/docs/hub/spaces-zerogpu
"""

from __future__ import annotations

import json
import os
import sys

import gradio as gr

# Add repo root to path so training/ and examiner_env/ are importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── ZeroGPU / GPU allocation ──────────────────────────────────────────────────
# @spaces.GPU allocates the GPU for the duration of this function call.
# Use duration=7200 for DEMO config (~2 hrs), 3600 for DEBUG.
try:
    import spaces  # type: ignore
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    # Fallback: define a no-op decorator for local dev
    class spaces:  # type: ignore
        @staticmethod
        def GPU(duration=3600):
            def decorator(fn):
                return fn
            return decorator


# ── Training function ─────────────────────────────────────────────────────────

@spaces.GPU(duration=7200)
def launch_training(
    config_name: str,
    wandb_key: str,
    hf_token: str,
    progress=gr.Progress(track_tqdm=True),
) -> str:
    """
    Launch GRPO training on GPU. Triggered from the Gradio UI.

    Uses TRL environment_factory=ExaminerToolEnv (multi-turn GRPO).
    Per TRL OpenEnv docs: https://huggingface.co/docs/trl/en/openenv

    Args:
        config_name: "DEBUG" (20 eps, ~30 min) or "DEMO" (200 eps, ~2 hrs)
        wandb_key: W&B API key (from huggingface.co/settings/tokens)
        hf_token: HF write token (for pushing artifacts after training)

    Returns:
        Training summary string.
    """
    import subprocess

    # Set credentials
    if not wandb_key:
        return "ERROR: W&B API key is required."
    if not hf_token:
        return "ERROR: HF token is required."

    os.environ["WANDB_API_KEY"] = wandb_key.strip()
    os.environ["HF_TOKEN"] = hf_token.strip()

    # Ensure output dirs exist
    for d in ["outputs/eval", "outputs/plots", "outputs/transcripts", "outputs/checkpoints"]:
        os.makedirs(os.path.join(_ROOT, d), exist_ok=True)

    progress(0.05, desc="Checking oracle calibration...")

    # Check oracle calibration
    cal_path = os.path.join(_ROOT, "outputs", "eval", "oracle_calibration.json")
    if not os.path.exists(cal_path):
        try:
            from examiner_env.calibration import run_calibration
            from examiner_env.knowledge_base import KB
            progress(0.10, desc="Running oracle calibration (200 episodes)...")
            run_calibration(KB, n_episodes=200, output_path=cal_path)
        except Exception as e:
            return f"ERROR: Oracle calibration failed: {e}"

    with open(cal_path) as f:
        cal = json.load(f)
    brier = cal["calibration_metrics"]["mean_brier"]
    if brier > 0.18:
        return f"ERROR: Oracle Brier={brier:.4f} > 0.18. Recalibrate before training."

    progress(0.15, desc=f"Calibration OK (Brier={brier:.4f}). Loading model...")

    # Run training
    try:
        from training.config import get_config
        from training.train_grpo import train

        eval_config_path = os.path.join(_ROOT, "eval_config.json")
        with open(eval_config_path) as f:
            eval_config = json.load(f)

        config = get_config(config_name)
        progress(0.20, desc=f"Starting {config_name} training ({config.num_episodes} episodes)...")

        final_metrics = train(config, eval_config)

        acc = final_metrics.get("classification_accuracy", float("nan"))
        gain = final_metrics.get("avg_info_gain_per_turn", float("nan"))
        ece = final_metrics.get("calibration_ECE", float("nan"))
        r_mean = final_metrics.get("reward_mean", float("nan"))

        progress(1.0, desc="Training complete!")
        return (
            f"Training complete ({config_name} config)\n"
            f"  Classification accuracy (held-out): {acc:.3f}\n"
            f"  Avg info gain/turn:                 {gain:.4f}\n"
            f"  Calibration ECE:                    {ece:.4f}\n"
            f"  Mean R_total:                       {r_mean:.4f}\n\n"
            f"Artifacts saved to outputs/eval/ and outputs/plots/\n"
            f"W&B dashboard: https://wandb.ai (project: bluffbuster-examiner)"
        )
    except Exception as e:
        import traceback
        return f"ERROR during training:\n{traceback.format_exc()}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def create_train_app():
    with gr.Blocks(title="BluffBuster — Training Launcher", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# BluffBuster Training Launcher\n"
            "Runs GRPO training on this Space's GPU using HF credits.\n\n"
            "> **Pattern:** TRL `environment_factory=ExaminerToolEnv` — "
            "[OpenEnv docs](https://huggingface.co/docs/trl/en/openenv)"
        )

        with gr.Row():
            config_dd = gr.Dropdown(
                choices=["DEBUG", "DEMO"],
                value="DEBUG",
                label="Training Config",
                info="DEBUG: 20 episodes, ~30 min. DEMO: 200 episodes, ~2 hrs.",
            )
        with gr.Row():
            wandb_tb = gr.Textbox(
                label="W&B API Key",
                type="password",
                placeholder="Find at wandb.ai/authorize",
            )
            hf_tb = gr.Textbox(
                label="HF Write Token",
                type="password",
                placeholder="Find at huggingface.co/settings/tokens",
            )

        run_btn = gr.Button("Launch Training", variant="primary", size="lg")
        result_box = gr.Textbox(
            label="Training Result",
            lines=12,
            interactive=False,
        )

        gr.Markdown(
            "**Notes:**\n"
            "- Training uses W&B for metric logging — you must provide an API key.\n"
            "- DEBUG config is for pipeline verification (no improvement expected).\n"
            "- DEMO config trains on F1+F2 styles, evaluates on held-out F3+S05.\n"
            "- GPU time is billed against your HF credits at Space hardware rates."
        )

        run_btn.click(
            fn=launch_training,
            inputs=[config_dd, wandb_tb, hf_tb],
            outputs=[result_box],
        )

    return app


if __name__ == "__main__":
    app = create_train_app()
    app.launch()
