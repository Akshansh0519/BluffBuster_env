"""
training/train_grpo.py — Unsloth + TRL GRPOTrainer training loop.
C2 owns.

⚠️  TRL GRPOTrainer API changes between releases.
    Before running: pip show trl | grep Version
    Then verify GRPOTrainer kwargs against that version's docs.
    Current target: trl>=0.8.0

ALL hyperparameters come from training/config.py. None are hardcoded here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

from training.config import TrainingConfig, get_config, DEBUG_CONFIG
from training.reward_fn import reward_fn, init_wandb, log_reward_breakdown

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ──────────────────────────────────────────────────────────
# Reward variance monitor
# ──────────────────────────────────────────────────────────

def _check_reward_variance(
    reward_buffer: list[float],
    config: TrainingConfig,
    step: int,
    current_beta_kl: float,
) -> float:
    """
    Checks reward variance. Warns if collapsed or exploding.
    Auto-increases KL penalty if variance is too high.
    Returns (possibly updated) beta_kl.
    """
    if len(reward_buffer) < 10:
        return current_beta_kl

    recent = reward_buffer[-50:]
    variance = float(np.std(recent))

    if _WANDB_AVAILABLE:
        wandb.log({"reward/variance_monitor": variance}, step=step)

    if variance < config.reward_variance_floor:
        msg = (
            f"WARNING [step {step}]: Reward variance {variance:.4f} < floor "
            f"{config.reward_variance_floor}. Signal may have collapsed."
        )
        print(msg)
        if _WANDB_AVAILABLE:
            wandb.log({"warning/reward_variance_collapsed": 1}, step=step)

    if variance > config.reward_variance_ceiling:
        new_beta = min(0.10, current_beta_kl * 1.5)
        msg = (
            f"WARNING [step {step}]: Reward variance {variance:.4f} > ceiling "
            f"{config.reward_variance_ceiling}. Auto-increasing beta_kl: "
            f"{current_beta_kl:.4f} → {new_beta:.4f}"
        )
        print(msg)
        if _WANDB_AVAILABLE:
            wandb.log({"adaptive/beta_kl": new_beta}, step=step)
        return new_beta

    return current_beta_kl


# ──────────────────────────────────────────────────────────
# Main training entry point
# ──────────────────────────────────────────────────────────

def train(config: TrainingConfig, eval_config: dict) -> None:
    """
    Full GRPO training loop.

    Steps:
    1. Init W&B
    2. Load model with Unsloth 4-bit quantization
    3. Apply LoRA
    4. Create GRPOTrainer with reward_fn bridge
    5. Register eval hook every N steps → run_eval() → log to W&B
    6. Train
    7. Final eval on held-out suite
    8. Save final metrics
    """
    # ── Deferred imports (require C1 env + Colab packages) ──
    try:
        from unsloth import FastLanguageModel
        from trl import GRPOTrainer, GRPOConfig
        from transformers import TrainerCallback
    except ImportError as e:
        raise RuntimeError(
            "Unsloth / TRL not installed. Run: pip install unsloth[colab] trl>=0.8.0\n"
            f"Original error: {e}"
        )

    try:
        from examiner_env.environment import ExaminerEnv
        from examiner_env.knowledge_base import KB
        from examiner_env.baselines import RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer
        from training.eval import run_eval
    except ImportError as e:
        raise RuntimeError(
            "examiner_env not available — C1 Phase 1 gate must clear first.\n"
            f"Original error: {e}"
        )

    # ── W&B init ──
    init_wandb(config)

    # ── Oracle calibration check ──
    cal_path = os.path.join("outputs", "eval", "oracle_calibration.json")
    if not os.path.exists(cal_path):
        raise FileNotFoundError(
            "outputs/eval/oracle_calibration.json not found. "
            "Run examiner_env.calibration.run_calibration() before training."
        )
    with open(cal_path) as f:
        cal = json.load(f)
    assert cal["calibration_metrics"]["mean_brier"] <= 0.18, (
        f"Oracle Brier {cal['calibration_metrics']['mean_brier']:.4f} > 0.18. "
        "Recalibrate before training."
    )
    print(f"✓ Oracle calibration OK — Brier={cal['calibration_metrics']['mean_brier']:.4f}")

    # ── Pre-training baseline eval ──
    print("Running pre-training baseline evaluation...")
    env = ExaminerEnv(kb=KB, config=config)
    baseline_metrics = {}
    for name, examiner in [
        ("RandomExaminer", RandomExaminer()),
        ("DefinitionalExaminer", DefinitionalExaminer()),
        ("BayesianHeuristicExaminer", BayesianHeuristicExaminer(KB)),
    ]:
        metrics = run_eval(examiner, eval_config, KB)
        baseline_metrics[name] = metrics
        print(f"  {name}: accuracy={metrics['classification_accuracy']:.3f}, "
              f"info_gain={metrics['avg_info_gain_per_turn']:.4f}")

    os.makedirs(os.path.join("outputs", "eval"), exist_ok=True)
    with open(os.path.join("outputs", "eval", "baseline_metrics.json"), "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    print("✓ baseline_metrics.json saved.")

    # ── Load model with Unsloth ──
    print(f"Loading model: {config.model_name} (4-bit={config.use_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.use_4bit,
        dtype=None,
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

    # ── Build prompt dataset for GRPOTrainer ──
    from datasets import Dataset
    from training.prompt_builder import build_prompt

    def generate_episode_prompts(n: int) -> list[dict]:
        """Generate n training prompts by running environment resets."""
        prompts = []
        for i in range(n):
            obs, _ = env.reset(seed=i)
            prompt = build_prompt(obs)
            prompts.append({"prompt": prompt, "episode_seed": i})
        return prompts

    dataset = Dataset.from_list(generate_episode_prompts(config.num_episodes))

    # ── GRPOConfig ──
    grpo_config = GRPOConfig(
        output_dir=os.path.join("outputs", "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        bf16=config.bf16,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        num_generations=config.num_generations,
        beta=config.beta_kl,
        save_steps=config.checkpoint_every_n_steps,
        logging_steps=1,
        report_to="wandb" if _WANDB_AVAILABLE else "none",
    )

    # ── Checkpoint + eval callback ──
    reward_buffer: list[float] = []
    current_beta_kl = config.beta_kl
    checkpoint_metrics_log: dict[int, dict] = {}

    class EvalAndMonitorCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            nonlocal current_beta_kl
            step = state.global_step

            # Reward variance monitor every 50 steps
            if step % 50 == 0 and reward_buffer:
                current_beta_kl = _check_reward_variance(
                    reward_buffer, config, step, current_beta_kl
                )

            # Periodic eval on frozen suite
            if step % config.eval_every_n_steps == 0:
                print(f"[step {step}] Running checkpoint eval...")
                chk_metrics = run_eval(
                    _TrainedExaminerWrapper(model, tokenizer, config),
                    eval_config,
                    KB,
                )
                checkpoint_metrics_log[step] = chk_metrics

                os.makedirs(os.path.join("outputs", "eval"), exist_ok=True)
                with open(os.path.join("outputs", "eval", "checkpoint_metrics.json"), "w") as f:
                    json.dump(checkpoint_metrics_log, f, indent=2)

                if _WANDB_AVAILABLE:
                    wandb.log(
                        {
                            "eval/reward_mean": chk_metrics["reward_mean"],
                            "eval/classification_accuracy": chk_metrics["classification_accuracy"],
                            "eval/avg_info_gain": chk_metrics["avg_info_gain_per_turn"],
                            "eval/calibration_ECE": chk_metrics["calibration_ECE"],
                            "eval/false_accusation_rate": chk_metrics["false_accusation_rate"],
                        },
                        step=step,
                    )
                print(f"  accuracy={chk_metrics['classification_accuracy']:.3f}, "
                      f"info_gain={chk_metrics['avg_info_gain_per_turn']:.4f}, "
                      f"ECE={chk_metrics['calibration_ECE']:.4f}")

    # ── Trainer ──
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=[EvalAndMonitorCallback()],
    )

    print(f"Starting {config.config_name} training ({config.num_episodes} episodes)...")
    trainer.train()

    # ── Final eval on held-out suite ──
    print("Running final held-out eval...")
    final_metrics = run_eval(
        _TrainedExaminerWrapper(model, tokenizer, config),
        eval_config,
        KB,
        output_path=os.path.join("outputs", "eval", "final_metrics.json"),
    )
    print(
        f"Final held-out eval — accuracy={final_metrics['classification_accuracy']:.3f}, "
        f"info_gain={final_metrics['avg_info_gain_per_turn']:.4f}, "
        f"ECE={final_metrics['calibration_ECE']:.4f}"
    )

    if _WANDB_AVAILABLE:
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


# ──────────────────────────────────────────────────────────
# Wrapper: turns a fine-tuned model into an examiner-compatible interface
# ──────────────────────────────────────────────────────────

class _TrainedExaminerWrapper:
    """Wraps the trained model to satisfy the examiner.act(observation) interface."""

    def __init__(self, model: Any, tokenizer: Any, config: TrainingConfig):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config

    def reset(self) -> None:
        pass

    def act(self, observation: dict) -> str:
        from training.prompt_builder import build_prompt
        prompt = build_prompt(observation)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BluffBuster Examiner with GRPO")
    parser.add_argument(
        "--config",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "DEMO", "FULL"],
        help="Training config tier",
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        default="eval_config.json",
        help="Path to eval_config.json",
    )
    args = parser.parse_args()

    with open(args.eval_config) as f:
        eval_cfg = json.load(f)

    cfg = get_config(args.config)
    train(cfg, eval_cfg)
