# implementation_coder2.md — C2 Training, Eval & Deployment Playbook
## The Examiner (BluffBuster) | Coder 2 Self-Contained Reference

> **Read this file + `context_primer.md` at every session start. Read `mistakes.md` to check active errors.**
> **C2 owns `training/`, `scripts/`, `hf_space/`, `notebooks/`, `outputs/`. Do NOT touch `examiner_env/` or `tests/`.**

---

## C2 OWNERSHIP MAP

| File | Phase | Status | Gate |
|------|-------|--------|------|
| `requirements.txt` | 0 | [ ] | Gate 0 |
| `training/config.py` | 0 | [ ] | Gate 0 |
| `eval_config.json` | 0 | [ ] | Gate 0 |
| `notebooks/train_examiner.ipynb` (skeleton) | 0 | [ ] | Gate 0 |
| `training/eval.py` | 1 | [ ] | Gate 1 |
| `outputs/eval/baseline_metrics.json` | 1 | [ ] | Gate 1 |
| `training/prompt_builder.py` | 2 | [ ] | Gate 2 |
| `training/reward_fn.py` | 2 | [ ] | Gate 2 |
| `training/train_grpo.py` | 2 | [ ] | Gate 2 |
| DEBUG smoke test | 2 | [ ] | Gate 2 |
| `outputs/eval/final_metrics.json` | 3 | [ ] | Gate 3 |
| `outputs/eval/checkpoint_metrics.json` | 3 | [ ] | Gate 3 |
| `scripts/select_transcripts.py` | 3 | [ ] | Gate 3 |
| `scripts/generate_plots.py` | 3 | [ ] | Gate 3 |
| `outputs/plots/` | 3 | [ ] | Gate 3 |
| `outputs/transcripts/` | 3 | [ ] | Gate 3 |
| `hf_space/app.py` | 4 | [ ] | Gate 4 |
| `notebooks/train_examiner.ipynb` (complete) | 4 | [ ] | Gate 4 |
| README.md | 4–5 | [ ] | Gate 4–5 |
| Writeup/mini-blog | 5 | [ ] | Gate 5 |

---

## C2 MSR OWNERSHIP

| MSR | C2 Responsibility |
|-----|-------------------|
| MSR-2 | Training script (Unsloth + TRL GRPO) as runnable Colab notebook — runs top-to-bottom on clean session |
| MSR-3 | Real plots from actual training run — derived from W&B artifacts, not mocked |
| MSR-5 | HF Space live, all 4 tabs functional, accessible in incognito |
| MSR-6 | README complete (all 6 sections, comparison table) |
| MSR-7 | README links to HF Space |
| MSR-8 | README links to all materials (blog, Colab, W&B) |
| MSR-9 | No video files in HF Hub |

**C2's work drives REWARD_EVIDENCE (20%), PIPELINE (10%), and co-owns STORYTELLING (30%).**

---

## PARALLEL WORK MAP

```
Phase 0 (parallel with C1):
  C2: repo structure, Colab skeleton, eval_config.json, W&B setup, config.py
  → While C1 builds models.py and action_parser.py

Phase 1 (parallel with C1):
  C2: training/eval.py (frozen eval runner) — can start with baselines once C1 finishes env skeleton
  → Unblocked as soon as C1 merges environment.py + baselines.py

Phase 2:
  C2 primary: prompt_builder.py → reward_fn.py → train_grpo.py → DEBUG smoke test
  → C1 reviews reward_fn.py to confirm no reward re-implementation

Phase 3:
  C2 primary: DEMO training run (Colab A100), plot generation
  → Parallel: C1 can do transcript selection if C2 is running the training

Phase 4:
  C2: HF Space, complete Colab notebook, README
  → All 3 complete together

Phase 5:
  C2 leads writeup; all 3 review
```

---

## WHAT C2 MUST NOT DO

- Touch `examiner_env/` or `tests/` (C1 owns)
- Use token-overlap divergence for diagnostic score (removed — see `guardrails.md` §3)
- Select transcripts by episode index (removed — behavior-quality selection only)
- Generate mocked or placeholder plots (MSR-3 blocker — immediate disqualification)
- Add video files to HF Hub (MSR-9 blocker)
- Hardcode hyperparameters outside `training/config.py`
- Narrate "3 visible training phases" unless phases genuinely appear in real data
- Re-implement reward logic in `reward_fn.py` (must call `examiner_env.reward.compute_reward()`)
- Commit `mistakes.md` to GitHub (local only, in `.gitignore`)

---

## DETAILED TASK SPECS

### TASK C2-0.1: Repo Structure + Colab Skeleton

**Colab notebook cell structure (must be exactly this order):**

```
Cell 1: !pip install
Cell 2: Authentication (W&B, HF tokens)
Cell 3: Clone/install repo
Cell 4: Oracle calibration (run calibration.py)
Cell 5: Baseline evaluation (run all 3 baselines on frozen eval suite)
Cell 6: Training (DEBUG config first)
Cell 7: DEMO training run
Cell 8: Final evaluation
Cell 9: Plot generation
Cell 10: Export/push to HF Hub
```

**Requirements.txt minimum:**
```
openenv
unsloth[colab]
trl>=0.8.0
wandb
gradio>=4.0
pydantic>=2.0
numpy
torch
transformers
accelerate
bitsandbytes
matplotlib
seaborn
sentence-transformers
```

**⚠️ No absolute paths in ANY cell:**
```python
# WRONG (will fail on clean Colab):
kb = load_kb("/home/user/Desktop/BluffBuster/examiner_env/knowledge_base.py")

# CORRECT:
import os
kb = load_kb(os.path.join(os.getcwd(), "examiner_env", "knowledge_base.py"))
```

---

### TASK C2-0.2: W&B Setup + Eval Config

**W&B initialization pattern for Colab:**
```python
import wandb
import os

# In Colab: set this before importing wandb
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")  # Colab secrets
wandb.init(
    project="bluffbuster-examiner",
    name=f"run-{config.config_name}-{datetime.now().strftime('%m%d-%H%M')}",
    config=vars(config),
    tags=[config.config_name, "grpo", "bluffbuster"]
)
```

**eval_config.json must be committed to repo** (it's not a secret):
```json
{
  "n_episodes": 50,
  "seeds": [1000, 1001, 1002, ..., 1049],
  "demo_config": {
    "training_styles": ["F1", "F2"],
    "held_out_styles": ["F3"],
    "training_sections": ["S01","S02","S03","S04"],
    "held_out_sections": ["S05"]
  }
}
```

---

### TASK C2-1.1: Frozen Eval Suite Runner (`training/eval.py`)

**ECE computation (10-bin, standard implementation):**
```python
def compute_ece(posteriors: list[float], labels: list[int], n_bins: int = 10) -> float:
    """
    posteriors: list of p_T(s) values (one per section per episode)
    labels: 1 if KNOWS (positive class), 0 if FAKING
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (np.array(posteriors) >= bins[i]) & (np.array(posteriors) < bins[i+1])
        if bin_mask.sum() == 0:
            continue
        bin_confidence = np.mean(np.array(posteriors)[bin_mask])
        bin_accuracy = np.mean(np.array(labels)[bin_mask])
        ece += bin_mask.sum() * abs(bin_confidence - bin_accuracy)
    return ece / len(posteriors)
```

**per_style_accuracy computation:**
```python
# from RewardBreakdown.posterior_trace and episode metadata
per_style_acc = {}
for episode in eval_results:
    for section_id, style in episode["style_assignments"].items():
        is_correct = (episode["classifications"][section_id] == episode["true_labels"][section_id])
        per_style_acc.setdefault(style, []).append(int(is_correct))
per_style_accuracy = {style: np.mean(vals) for style, vals in per_style_acc.items()}
```

**Baseline eval MUST run BEFORE any training:**
```python
# In Colab notebook, this cell runs BEFORE training:
from examiner_env.baselines import RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer
from training.eval import run_eval
import json

baseline_metrics = {}
for examiner_name, examiner in [
    ("RandomExaminer", RandomExaminer()),
    ("DefinitionalExaminer", DefinitionalExaminer()),
    ("BayesianHeuristicExaminer", BayesianHeuristicExaminer(kb)),
]:
    metrics = run_eval(examiner, eval_config, kb)
    baseline_metrics[examiner_name] = metrics

json.dump(baseline_metrics, open("outputs/eval/baseline_metrics.json", "w"), indent=2)
print("Baseline eval complete.")
```

---

### TASK C2-2.1: Prompt Builder (`training/prompt_builder.py`)

**Prompt leakage check — Validator will run this test:**
```python
obs = env.reset(seed=0)[0]  # real observation
prompt = build_prompt(obs)
# These strings must NOT appear in the prompt:
forbidden = ["KNOWS", "FAKING", "K1", "K2", "K3", "F1", "F2", "F3", "F4",
             "true_label", "posterior", "p_t", "p_0", "0.5", "0.7"]
# Exception: "KNOWS" and "FAKING" MAY appear as schema example values only
# Check that no section has a label attached to it in the prompt
import re
for s_id in ["S01","S02","S03","S04","S05","S06","S07","S08","S09","S10"]:
    # Should not appear as: "S01: Gradient Descent (KNOWS)" or similar
    assert not re.search(rf"{s_id}.*KNOWS|{s_id}.*FAKING", prompt), \
        f"Hidden label leaked for {s_id}!"
print("Prompt leakage check: PASSED")
```

---

### TASK C2-2.2: Reward Function Bridge (`training/reward_fn.py`)

**CRITICAL: The reward function must DELEGATE to `examiner_env.reward`, never re-implement.**

**Correct pattern:**
```python
from examiner_env.reward import compute_reward
from examiner_env.action_parser import parse

def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    rewards = []
    n_malformed = 0

    for completion in completions:
        action = parse(completion)
        if isinstance(action, MalformedAction):
            rewards.append(-0.20)  # P_malformed for 1 malformed
            n_malformed += 1
            continue

        # Get episode result from environment state (passed via kwargs or thread-local)
        episode_result = kwargs.get("episode_results", {}).get(id(completion))
        if episode_result is None:
            rewards.append(-0.20)  # fallback
            continue

        breakdown = compute_reward(episode_result, kb)
        rewards.append(breakdown.R_total)

    # W&B logging
    wandb.log({
        "reward/R_total_batch_mean": np.mean(rewards),
        "reward/R_total_batch_std": np.std(rewards),
        "training/parse_failure_rate": n_malformed / len(completions),
    })

    return rewards
```

**Wrong pattern (AI will try this — prevent it):**
```python
# WRONG — re-implementing reward logic in reward_fn.py:
def reward_fn(completions, prompts, **kwargs):
    rewards = []
    for completion in completions:
        if "KNOWS" in completion:
            reward = 1.0  # WRONG: this is not the reward function
        ...
    return rewards
```

---

### TASK C2-2.3: Training Script (`training/train_grpo.py`)

**⚠️ TRL GRPOTrainer API changes frequently. Before writing this prompt, check current TRL version:**
```bash
pip show trl | grep Version
# Then look up GRPOTrainer in TRL docs for that version
```

**W&B logging hooks (must be actual W&B calls, not print statements):**
```python
class RewardLoggingCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            # Log per-component rewards to W&B
            wandb.log({
                "eval/reward_mean": metrics.get("eval_reward_mean"),
                "eval/classification_accuracy": metrics.get("eval_classification_accuracy"),
                "eval/avg_info_gain": metrics.get("eval_avg_info_gain_per_turn"),
                "eval/calibration_ECE": metrics.get("eval_calibration_ECE"),
            }, step=state.global_step)
```

**Reward variance monitor:**
```python
# After every 50 episodes, check reward variance
if episode_count % 50 == 0:
    recent_rewards = reward_buffer[-50:]
    variance = np.std(recent_rewards)
    if variance < config.reward_variance_floor:  # 0.05
        wandb.log({"warning/reward_variance_collapsed": 1}, step=step)
        print(f"WARNING: Reward variance {variance:.4f} < floor {config.reward_variance_floor}")
    if variance > config.reward_variance_ceiling:  # 1.5
        # Auto-increase KL penalty
        current_beta_kl = min(0.10, current_beta_kl * 1.5)
        wandb.log({"adaptive/beta_kl": current_beta_kl}, step=step)
```

---

### TASK C2-2.4: DEBUG Smoke Test Verification

**Run this verification script after DEBUG training:**
```python
# smoke_test_verify.py
import wandb
import json
import numpy as np

# Load W&B run data
api = wandb.Api()
run = api.run(f"bluffbuster-examiner/{RUN_ID}")

# Check: finite rewards within bounds
history = run.history(keys=["reward/R_total"])
rewards = history["reward/R_total"].dropna().values
assert all(np.isfinite(rewards)), "Non-finite rewards found!"
assert all(-2.05 <= r <= 1.95 for r in rewards), "Rewards out of bounds!"
print(f"✓ Rewards: finite, in [-2.05, +1.95], n={len(rewards)}")

# Check: variance not collapsed
reward_std = np.std(rewards)
assert 0.05 <= reward_std <= 1.5, f"Reward variance {reward_std:.4f} outside [0.05, 1.5]"
print(f"✓ Reward variance: {reward_std:.4f}")

# Check: all 11 per-component rewards logged
required_metrics = ["reward/R_acc", "reward/R_asym", "reward/R_cal", "reward/R_eff",
                    "reward/R_cov", "reward/R_info", "reward/R_qual", "reward/R_div",
                    "reward/P_malformed", "reward/P_repetition", "reward/P_invalid_sec"]
for metric in required_metrics:
    hist = run.history(keys=[metric])[metric].dropna()
    assert len(hist) > 0, f"Missing W&B metric: {metric}"
    print(f"✓ {metric}: logged ({len(hist)} values)")

# Check: oracle calibration exists
cal = json.load(open("outputs/eval/oracle_calibration.json"))
assert cal["calibration_metrics"]["mean_brier"] <= 0.18
print(f"✓ Oracle calibration: Brier={cal['calibration_metrics']['mean_brier']:.4f}")

print("\n=== ALL SMOKE TEST CHECKS PASSED ===")
```

---

### TASK C2-3.1: DEMO Training Run

**Pre-run checklist:**
```
[ ] Oracle calibration complete (outputs/eval/oracle_calibration.json exists, Brier ≤ 0.18)
[ ] All Phase 1 tests passing (run pytest tests/ -v remotely or locally)
[ ] eval_config.json has 50 seeds
[ ] W&B API key set as Colab secret
[ ] HF token set as Colab secret (for push at end)
[ ] DEMO_CONFIG loaded in training script
[ ] baseline_metrics.json generated before starting training
```

**Post-run checklist:**
```
[ ] W&B run is accessible at wandb.ai/{your_team}/bluffbuster-examiner
[ ] checkpoint_metrics.json has entries at steps 0, 50, 100, 150, 200
[ ] final_metrics.json exists with all required fields
[ ] W&B run ID logged to outputs/plots/README.md
```

**If Colab disconnects mid-run:**
- Resume from last checkpoint: `FastLanguageModel.from_pretrained(checkpoint_path)`
- TRL GRPOTrainer supports `resume_from_checkpoint` — use it
- Document the gap in W&B notes (don't hide it)

---

### TASK C2-3.3: Plot Generation (`scripts/generate_plots.py`)

**All plots must have these properties:**
1. Title that includes config name (e.g., "DEMO Config — 200 Episodes")
2. X-axis labeled with actual step numbers (not episode indices)
3. Data sourced from `checkpoint_metrics.json` or W&B API (not hardcoded)
4. Saved to `outputs/plots/{plot_name}.png` at 150+ DPI

**Per-style accuracy heatmap:**
```python
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np

final_metrics = json.load(open("outputs/eval/final_metrics.json"))
per_style = final_metrics["TrainedExaminer"]["per_style_accuracy"]
styles = ["K1", "K2", "K3", "F1", "F2", "F3", "F4"]
sections = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"]

# Build matrix (rows=styles, cols=sections)
# NaN for (style, section) combos not in eval suite
matrix = np.full((len(styles), len(sections)), np.nan)
for i, style in enumerate(styles):
    for j, section in enumerate(sections):
        key = f"{style}_{section}"
        if key in per_style:
            matrix[i, j] = per_style[key]

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(matrix, xticklabels=sections, yticklabels=styles,
            annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
            mask=np.isnan(matrix), ax=ax)
ax.set_title("Per-Style Accuracy Heatmap — TrainedExaminer on Held-Out Eval Suite\n(NaN = not in eval suite)")
plt.tight_layout()
plt.savefig("outputs/plots/per_style_heatmap.png", dpi=150)
```

**⚠️ NaN cells in the heatmap are OK for (style, section) combos not in the eval suite. Do NOT fill with 0 — that would be misleading.**

---

### TASK C2-4.1: HF Space (`hf_space/app.py`)

**Gradio 4 multi-tab pattern:**
```python
import gradio as gr

def create_app(kb, env, baselines, trained_model):
    with gr.Blocks(title="The Examiner — BluffBuster") as app:
        gr.Markdown("# The Examiner\n> " + THREE_SENTENCE_NARRATIVE)

        with gr.Tab("Live Episode"):
            # Tab 1 content
            run_btn = gr.Button("Run Episode")
            section_titles_display = gr.Markdown()
            dialogue_display = gr.Dataframe(headers=["Turn", "Section", "Question", "Response"])
            posterior_chart = gr.LinePlot(
                x="Turn", y="Posterior", color="Section",
                title="Per-Section Belief Trace (p_t)"
            )
            info_gain_chart = gr.BarPlot(x="Turn", y="Info Gain (ΔH_t)")
            reward_breakdown_display = gr.JSON(label="Reward Breakdown")
            ground_truth_display = gr.Markdown(visible=False)

        with gr.Tab("Baseline vs Trained"):
            # Tab 2 content (4 examiners side by side)
            seed_input = gr.Number(value=1000, label="Episode Seed")
            run_comparison_btn = gr.Button("Run Comparison")
            comparison_display = gr.DataFrame()

        with gr.Tab("Training Evidence"):
            # Tab 3 content (real plots)
            gr.Image("outputs/plots/reward_curve.png", label="R_total over Training")
            gr.Image("outputs/plots/accuracy_curve.png", label="Accuracy over Training")
            gr.Image("outputs/plots/comparison_bar.png", label="Baseline Comparison")
            gr.Image("outputs/plots/per_style_heatmap.png", label="Per-Style Accuracy")

        with gr.Tab("Environment Details"):
            # Tab 4 content (technical details)
            gr.Markdown(ENV_DETAILS_MARKDOWN)

    return app
```

**HF Space config (`hf_space/README.md` — required YAML front matter):**
```yaml
---
title: The Examiner - BluffBuster
emoji: 🧐
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
tags:
  - reinforcement-learning
  - education
  - grpo
  - openenv
---
```

**Common HF Space failures (check Space logs):**
- Missing `requirements.txt` in `hf_space/` folder
- App imports `examiner_env` but path isn't set up
- Plot files not included in Space (need to be in `hf_space/outputs/plots/`)
- `gr.LinePlot` API version mismatch (check Gradio 4 docs)

**Test in incognito before marking done:**
1. Open Space URL in incognito
2. Tab 1: Click "Run Episode" — dialogue appears, posterior chart updates
3. Tab 2: Click "Run Comparison" — all 4 examiners show side by side
4. Tab 3: All 4 plot images visible (not broken image icons)
5. Tab 4: Markdown renders correctly

---

### TASK C2-4.2: Colab Notebook End-to-End Test

**Pre-submission test protocol:**
```
1. Open a NEW Colab session (Runtime → Disconnect and delete runtime)
2. Upload the notebook or open from GitHub
3. Run ALL cells top to bottom WITHOUT modifying anything
4. Verify:
   - Cell 1 (install) completes without error
   - Cell 4 (oracle calibration) produces oracle_calibration.json
   - Cell 5 (baseline eval) produces baseline_metrics.json
   - Cell 6 (DEBUG smoke test) shows R_total finite and non-constant
   - W&B dashboard shows a new run with metrics
   - Final cell produces at least one plot
```

**If any cell fails on clean Colab:** That is a MSR-2 blocker. Fix before marking done.

---

### TASK C2-5.1: Writeup / Mini-Blog

**Required content checklist:**
```
[ ] 3-sentence narrative (verbatim from PROJECT IDENTITY — do not paraphrase)
[ ] Screenshot from Tab 2 (4 examiners side-by-side)
[ ] Comparison table: [Random|Definitional|BayesianHeuristic|Trained] × [accuracy|info_gain|ECE]
[ ] Reward curve image (R_total over training steps, from real run)
[ ] Honest caveat 1: "Our simulator uses scripted styles, not real human experts..."
[ ] Honest caveat 2: "We do not claim guaranteed improvement..."
[ ] Link to HF Space (MSR-7)
[ ] Link to Colab notebook (MSR-8)
[ ] Link to W&B training run (MSR-8)
```

**HuggingFace blog post format:**
- Create at huggingface.co/blog/new
- Include all required content above
- Must be publicly accessible without login (MSR-4)
- Target length: 800–1200 words (under 5 min read)

---

## SESSION START CHECKLIST (C2)

```
[ ] git pull origin main
[ ] Read mistakes.md — check active errors list
[ ] Check which phase/task is in progress
[ ] Paste context_primer.md into AI tool
[ ] Confirm: am I working on a file in training/ or scripts/ or hf_space/ or notebooks/?
[ ] Check if C1's gate has been cleared (required before Phase 2 starts)
[ ] Confirm WANDB_API_KEY and HF_TOKEN are set (for training/deployment tasks)
```

## SESSION END CHECKLIST (C2)

```
[ ] All 3 sanity conditions for completed task are TRUE
[ ] W&B dashboard shows expected metrics (if training/eval task)
[ ] HF Space accessible in incognito (if deployment task)
[ ] New AI mistakes logged to mistakes.md (do NOT commit mistakes.md)
[ ] Feature branch committed with structured commit message
[ ] If at gate: notify Validator for gate review
[ ] Update ownership map table above (mark status)
```

---

## C2 VALIDATOR HANDOFF TEMPLATES

### Handoff after Phase 2 Gate:
```
HANDOFF: C2 → Validator | Phase 2 Gate
Date: [date]
MSR Status: MSR-2 partial (Colab imports without error)

Smoke test checklist:
  [ ] 20 episodes, no crash
  [ ] All R_total finite and in [-2.05, +1.95]
  [ ] σ(R_total) ∈ [0.05, 1.5]: [VALUE]
  [ ] 11 per-component rewards in W&B: [LINK]
  [ ] parse_failure_rate < 0.5: [VALUE]
  [ ] Posterior trace logged: [Y/N]
  [ ] oracle_calibration.json Brier: [VALUE]
  [ ] Advantage mean ≈ 0: [VALUE], std ≈ 1: [VALUE]

W&B run: [URL]
Blockers: [any issues]
```

### Handoff after Phase 4 Gate:
```
HANDOFF: C2 → Validator | Phase 4 Gate
MSR-2: Colab tested on clean runtime [PASS/FAIL]
MSR-3: Plots derived from W&B run [RUN_ID] [PASS/FAIL]
MSR-5: HF Space [URL] tested in incognito [PASS/FAIL]
MSR-6: README all 6 sections complete [PASS/FAIL]
MSR-7: HF Space link in README [PASS/FAIL]
MSR-9: No video files [PASS/FAIL — git ls-files output]
```

---

*Last updated: 2026-04-25 | Version 1.0 | C2 Playbook for BluffBuster / The Examiner*
