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
from typing import Any

import numpy as np

from training.config import TrainingConfig, get_config, DEBUG_CONFIG

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


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
    # ── Deferred imports (require Colab packages) ──────────────────────────
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
    assert brier <= 0.18, f"Oracle Brier={brier:.4f} > 0.18 — recalibrate."
    print(f"Oracle calibration OK (Brier={brier:.4f})")

    # ── Pre-training baseline eval ─────────────────────────────────────────
    _run_baseline_eval(eval_config, KB, run_eval,
                       RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer)

    # ── Load model with Unsloth ────────────────────────────────────────────
    print(f"Loading {config.model_name} (4-bit={config.use_4bit})...")
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
    print("Model loaded with LoRA adapters.")

    # ── Episode seed dataset ───────────────────────────────────────────────
    # TRL forwards each dataset row's columns to env.reset(**kwargs).
    # episode_seed → reset(episode_seed=...) for deterministic episodes.
    n = config.num_episodes
    dataset = Dataset.from_dict({
        "prompt": [[{
            "role": "user",
            "content": (
                "You are an expert examiner testing a student on machine learning theory. "
                "Use the ask() tool to probe each section, then classify() to end. "
                "Prefer WHY/HOW/edge-case questions — they expose bluffing better than "
                "surface definitions. You MUST call classify() by the final turn."
            ),
        }]] * n,
        "episode_seed": list(range(n)),
    })

    # ── Eval + checkpoint callback ─────────────────────────────────────────
    reward_buffer: list[float] = []
    current_beta_kl = config.beta_kl
    checkpoint_metrics_log: dict = {}

    class EvalAndMonitorCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                r = logs.get("train/reward", logs.get("reward/R_total_batch_mean"))
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
                from examiner_env.environment import ExaminerEnv
                _TrainedExaminerWrapper.model = model
                _TrainedExaminerWrapper.tokenizer = tokenizer
                _TrainedExaminerWrapper.config = config
                chk = run_eval(_TrainedExaminerWrapper(), eval_config, KB)
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
                    }, step=step)
                print(f"  accuracy={chk['classification_accuracy']:.3f} "
                      f"info_gain={chk['avg_info_gain_per_turn']:.4f} "
                      f"ECE={chk['calibration_ECE']:.4f}")

    # ── GRPOConfig ────────────────────────────────────────────────────────
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
        # Multi-turn budget — enough for up to 6 turns of Q&A + classify
        max_completion_length=config.max_seq_length,
        # Tool-calling mode: environment_factory handles the loop
        log_completions=True,
    )

    # ── GRPOTrainer with environment_factory ──────────────────────────────
    # Per TRL OpenEnv docs: pass the CLASS (not an instance).
    # TRL creates one ExaminerToolEnv per generation slot.
    # It calls reset() → multi-turn (ask/classify tool calls) → reward_func().
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        environment_factory=ExaminerToolEnv,
        callbacks=[EvalAndMonitorCallback()],
    )

    print(f"\nStarting {config.config_name} training ({n} episodes)...")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA: r={config.lora_rank} alpha={config.lora_alpha}")
    print(f"  Generations/step: {config.num_generations}")
    print(f"  Eval every {config.eval_every_n_steps} steps")
    trainer.train()

    # ── Final held-out eval ───────────────────────────────────────────────
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


class _TrainedExaminerWrapper:
    """Wraps trained model into examiner.act(observation) interface for run_eval()."""
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
        return self.__class__.tokenizer.decode(generated, skip_special_tokens=True).strip()


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
