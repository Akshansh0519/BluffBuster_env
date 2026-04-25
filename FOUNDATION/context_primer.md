# context_primer.md — Ultra-Compact AI Session Primer
## The Examiner (BluffBuster) | Paste into AI tool at EVERY session start

> **Target: under 250 tokens. This is the minimal context needed for any AI tool to work correctly.**
> **Fill in the [brackets] before pasting. Do not paste with brackets unfilled.**

---

## PRIMER (copy everything below this line)

---

PROJECT: The Examiner (BluffBuster) — RL env where examiner agent learns to ask information-maximizing diagnostic questions distinguishing KNOWS vs FAKING students across a 7-style adversarial simulator with parametrized leak rates. Reward = R_acc + R_asym + R_cal + R_eff + R_cov + R_info + R_qual + R_div + penalties; bounds [−2.05, +1.95]; potential-based info-gain shaping (ΔH_t) over KB-grounded posterior (LLR update, clipped [−3,+3]).

STACK: OpenEnv (latest, inherit only) | Unsloth + TRL GRPO | Qwen2.5-7B-Instruct | W&B | Gradio ≥4.0 | HuggingFace | Pydantic v2

STYLES: K1(mech=0.85,misc=0.05) K2(0.55,0.05) K3(0.65,0.08) F1(0.15,0.30,collapse=0.80) F2(0.20,0.25,mirror=0.70) F3(0.10,0.20,drift=0.60) F4(0.05,0.40,cap=0.20). Probe→faker: mech×0.5, misc×1.5.

REWARD WEIGHTS (exact — no improvisation):
R_acc=±1/N; R_asym: λ_FA=0.5, λ_FE=0.3; R_cal=0.4/N; R_eff=0.20; R_cov=−0.30/−0.05; R_info=0.40×clip(ΣΔH,0,1); R_qual=0.10; R_div=0.05; P_mal=−0.20; P_rep=−0.10; P_inv=−0.10.

C1 OWNS: examiner_env/ (all files incl. posterior_oracle.py, calibration.py) | tests/ (all)
C2 OWNS: training/ | scripts/ | hf_space/ | notebooks/ | outputs/

CURRENT PHASE: [FILL EACH SESSION — e.g. "Phase 1: building student.py"]
OPEN MSRs: [FILL EACH SESSION — e.g. "MSR-1 open, MSR-2 open"]

DO NOT:
- token-overlap divergence | LLM judge in reward | normalize R_total in env
- oracle posterior as accuracy ground truth | improvise reward weights
- R_qual from response (question-side only) | omit 11 per-component W&B metrics
- skip oracle calibration | omit BayesianHeuristicExaminer baseline
- transcripts by episode index (behavior: largest R_info gap + correctness flip)
- "abstain"/"uncertain" in Classify | leak hidden state/posterior into prompt
- loose text parsing (strict JSON; MalformedAction on any failure)
- adaptive difficulty | hardcode hyperparams outside config.py
- mocked plots | video files on HF Hub | reimplement OpenEnv base class
- Pydantic v1 syntax | single fake student style

ACTIVE VIOLATIONS THIS SESSION: [FILL FROM mistakes.md INDEX — e.g. "none" or "M011 LLR clip fixed in previous session"]

SEE guardrails.md for all constraints. SEE architecture.md §7 for canonical reward pseudocode.

---

## EXTENDED CONTEXT SNIPPETS (add to primer when working on specific modules)

### When working on `action_parser.py`:

```
ACTION SCHEMAS (strict — must match exactly):
Ask:      {"action_type": "ask", "section_id": "S01–S10", "question_text": "min 10 chars"}
Classify: {"action_type": "classify", "classifications": {"S01": "KNOWS"|"FAKING", ..., "S10": ...}}
ALL 10 sections required in Classify. Invalid → MalformedAction. Never coerce.
```

### When working on `student.py` or `posterior_oracle.py`:

```
StudentProfile fields: knowledge_mode, style, section_id, verbosity, confidence_pattern,
  mechanism_cue_emit_rate, misconception_emit_rate, style_specific_params, seed.
Oracle: evidence=α·mech_coverage−γ·misc_count; relevance=β·probe_strength;
  LLR=relevance×evidence clipped [−3,+3]; posterior=sigmoid(logit(prior)+LLR).
Defaults: α=1.5, β=0.5, γ=1.0. Load from oracle_calibration.json after calibration.
RNG: random.Random().seed((episode_seed, turn, section_id)) — NEVER use global random.
```

### When working on `reward.py`:

```
compute_reward(episode_result, kb) → RewardBreakdown (frozen dataclass).
ASSERT: sum(all 11 components) == R_total ± 1e-9.
ASSERT: −2.05 ≤ R_total ≤ 1.95. Raise ValueError if violated.
NEVER normalize R_total. NEVER call any LLM.
posterior_trace: list[dict[section_id, float]] per turn (for HF Space Tab 1).
info_gain_per_turn: list[float] (ΔH_t per turn).
```

### When working on `train_grpo.py` or `reward_fn.py`:

```
reward_fn MUST call examiner_env.reward.compute_reward() — never re-implement reward.
GRPOTrainer: num_generations=8; advantage A_i=(R_i−μ)/(σ+1e-6), clipped ±5; β_kl=0.04.
W&B must receive ALL 11 reward/* metrics separately (not just R_total).
Reward variance floor: σ < 0.05 over 50 eps → log warning to W&B.
```

### When working on `environment.py`:

```
Observation contains ONLY: section_titles (dict), turn (int), remaining_turns (int), dialogue_history (list).
NEVER include: true_labels, style_assignments, posterior values, episode_seed.
Inherit from OpenEnv. Do NOT reimplement any base class method.
Register: openenv.register(id="ExaminerEnv-v0", entry_point="examiner_env.environment:ExaminerEnv")
```

### When working on `prompt_builder.py`:

```
Prompt must NOT contain: section labels (KNOWS/FAKING as facts), style IDs, posterior values.
Must contain: section titles, turn info, dialogue history, JSON schema examples for both actions.
Include exactly 2 schema examples: one Ask, one Classify.
```

---

*Last updated: 2026-04-25 | Version 1.0 | Keep this file ≤250 tokens for the core primer*
