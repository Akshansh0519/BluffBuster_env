"""
training/config.py — All hyperparameters for DEBUG / DEMO / FULL training configs.
C2 owns. No hyperparameters may be hardcoded anywhere else in the codebase.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    config_name: str
    sections: List[str]
    max_turns: int
    num_episodes: int
    fake_styles_train: List[str]
    eval_styles_held_out: List[str]
    held_out_sections: List[str]
    eval_episodes: int
    model_name: str
    lora_rank: int
    lora_alpha: int
    max_seq_length: int
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    num_generations: int
    bf16: bool
    use_4bit: bool
    beta_kl: float
    advantage_clip: float
    reward_variance_floor: float
    reward_variance_ceiling: float
    max_grad_norm: float
    warmup_ratio: float
    checkpoint_every_n_steps: int
    eval_every_n_steps: int

    # Theoretical reward bounds (asserted at runtime; never override inside env)
    reward_min: float = -2.05
    reward_max: float = 1.95


# ──────────────────────────────────────────────────────────
# DEBUG  — Smoke test config: pipeline health verification only
# Purpose: verify finite rewards, non-constant variance, W&B logging
# NOT for measuring improvement; 20 episodes is too few.
# ──────────────────────────────────────────────────────────
DEBUG_CONFIG = TrainingConfig(
    config_name="DEBUG",
    sections=["S01", "S02", "S03"],
    max_turns=3,
    num_episodes=20,
    fake_styles_train=["F1"],
    eval_styles_held_out=[],
    held_out_sections=[],
    eval_episodes=10,
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    lora_rank=8,
    lora_alpha=16,
    # 2048 avoids Unsloth truncating prompt+completion when system prompt is long
    # (logs showed seq 1061 > 1024 on T4 DEBUG runs).
    max_seq_length=2048,
    batch_size=1,
    gradient_accumulation=4,
    learning_rate=5e-6,
    num_generations=4,
    bf16=True,
    use_4bit=True,
    beta_kl=0.04,
    advantage_clip=5.0,
    reward_variance_floor=0.05,
    reward_variance_ceiling=1.5,
    max_grad_norm=1.0,
    warmup_ratio=0.05,
    checkpoint_every_n_steps=10,
    eval_every_n_steps=10,
)

# ──────────────────────────────────────────────────────────
# DEMO  — Evidence generation: submit this config
# Trains on F1+F2, evaluates on held-out F3 style and S05 section
# ──────────────────────────────────────────────────────────
DEMO_CONFIG = TrainingConfig(
    config_name="DEMO",
    sections=["S01", "S02", "S03", "S04", "S05"],
    max_turns=4,
    # Time-constrained submission DEMO: 100 episodes (~50 optimizer steps with
    # batch_size=2). Mid-training run_eval() was freezing Spaces for tens of
    # minutes (50 eval episodes × generate); checkpoint curve is optional.
    # Final held-out eval still runs once at the end of train().
    num_episodes=100,
    fake_styles_train=["F1", "F2"],
    eval_styles_held_out=["F3"],
    held_out_sections=["S05"],
    eval_episodes=50,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    lora_rank=16,
    lora_alpha=32,
    max_seq_length=2048,
    batch_size=2,
    gradient_accumulation=8,
    learning_rate=5e-6,
    num_generations=8,
    bf16=True,
    use_4bit=True,
    beta_kl=0.04,
    advantage_clip=5.0,
    reward_variance_floor=0.05,
    reward_variance_ceiling=1.5,
    max_grad_norm=1.0,
    warmup_ratio=0.05,
    checkpoint_every_n_steps=25,
    # Skip mid-training eval on HF (set >> num_episodes so callback never fires).
    eval_every_n_steps=10_000,
)

# ──────────────────────────────────────────────────────────
# FULL  — All 10 sections, 3 training styles, held-out F4 + S09/S10
# Run only if hardware and time allow after DEMO is complete.
# ──────────────────────────────────────────────────────────
FULL_CONFIG = TrainingConfig(
    config_name="FULL",
    sections=["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"],
    max_turns=4,
    num_episodes=200,
    fake_styles_train=["F1", "F2", "F3"],
    eval_styles_held_out=["F4"],
    held_out_sections=["S09", "S10"],
    eval_episodes=10,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    lora_rank=16,
    lora_alpha=32,
    max_seq_length=1280,
    batch_size=2,
    gradient_accumulation=8,
    learning_rate=5e-6,
    num_generations=8,
    bf16=True,
    use_4bit=True,
    beta_kl=0.04,
    advantage_clip=5.0,
    reward_variance_floor=0.05,
    reward_variance_ceiling=1.5,
    max_grad_norm=1.0,
    warmup_ratio=0.05,
    checkpoint_every_n_steps=25,
    eval_every_n_steps=9999,
)

# ──────────────────────────────────────────────────────────
# DEMO_FAST  — Time-constrained submission run (7B-Instruct, ~1.5 hrs on A100)
# Same held-out split as DEMO; 5 sections, max_turns=4, seq=1024 for speed.
# VALIDATOR NOTE: 1.5B was removed — it caused parse_failure_rate ~1.0.
# ──────────────────────────────────────────────────────────
DEMO_FAST_CONFIG = TrainingConfig(
    config_name="DEMO_FAST",
    sections=["S01", "S02", "S03", "S04", "S05"],
    max_turns=4,
    num_episodes=200,
    fake_styles_train=["F1", "F2"],
    eval_styles_held_out=["F3"],
    held_out_sections=["S05"],
    eval_episodes=50,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    lora_rank=16,
    lora_alpha=32,
    max_seq_length=1024,
    batch_size=2,
    gradient_accumulation=8,
    learning_rate=5e-6,
    num_generations=8,
    bf16=True,
    use_4bit=True,
    beta_kl=0.04,
    advantage_clip=5.0,
    reward_variance_floor=0.05,
    reward_variance_ceiling=1.5,
    max_grad_norm=1.0,
    warmup_ratio=0.05,
    checkpoint_every_n_steps=25,
    eval_every_n_steps=50,
)

# ──────────────────────────────────────────────────────────
# FULL_FAST  — All 10 sections, 7B-Instruct, deadline-safe
# Same split as FULL; max_turns=4, seq=1024 to fit in ~3 hrs on A100.
# VALIDATOR NOTE: 1.5B removed — use 7B with reduced seq+turns for speed.
# ──────────────────────────────────────────────────────────
FULL_FAST_CONFIG = TrainingConfig(
    config_name="FULL_FAST",
    sections=["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"],
    max_turns=4,
    num_episodes=200,
    fake_styles_train=["F1", "F2", "F3"],
    eval_styles_held_out=["F4"],
    held_out_sections=["S09", "S10"],
    eval_episodes=60,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    lora_rank=16,
    lora_alpha=32,
    max_seq_length=1024,
    batch_size=2,
    gradient_accumulation=8,
    learning_rate=5e-6,
    num_generations=8,
    bf16=True,
    use_4bit=True,
    beta_kl=0.04,
    advantage_clip=5.0,
    reward_variance_floor=0.05,
    reward_variance_ceiling=1.5,
    max_grad_norm=1.0,
    warmup_ratio=0.05,
    checkpoint_every_n_steps=25,
    eval_every_n_steps=50,
)

CONFIGS = {
    "DEBUG":     DEBUG_CONFIG,
    "DEMO":      DEMO_CONFIG,
    "DEMO_FAST": DEMO_FAST_CONFIG,
    "FULL":      FULL_CONFIG,
    "FULL_FAST": FULL_FAST_CONFIG,
    "FAST":      FULL_FAST_CONFIG,
}


def get_config(name: str) -> TrainingConfig:
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Choose from: {list(CONFIGS)}")
    return CONFIGS[name]
