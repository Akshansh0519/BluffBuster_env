# implementation_validator.md — Validator QC Playbook
## The Examiner (BluffBuster) | Validator Self-Contained Reference

> **Validator is never idle. Your role: prevent regressions, catch scope drift, clear gate conditions, maintain scientific honesty.**
> **Authority: `guardrails.md`. You enforce it. No merge to main without your approval.**

---

## VALIDATOR ROLE SUMMARY

| Phase | Validator's Primary Work |
|-------|--------------------------|
| Phase 0 | Prepare review prompts, pre-check architecture vs MSRs, set up W&B project, create frozen eval suite seeds |
| Phase 1 | Verify parser tests (8 malformed cases), confirm OpenEnv inheritance, check no hidden state leaks, verify 7 styles with behavioral differentiation |
| Phase 2 | Verify reward function matches `architecture.md` §7 pseudocode exactly, confirm W&B logging is live (not just print), verify `reward_fn.py` delegates to `reward.py` |
| Phase 3 | Verify training run is real (W&B artifacts exist), validate plots from actual logs, confirm transcript selection is behavior-based |
| Phase 4 | Verify HF Space accessible in incognito (all 4 tabs), run Colab top-to-bottom on clean runtime |
| Phase 5 | Review writeup from non-technical judge perspective, confirm honest caveats present, confirm 3-sentence narrative is verbatim |

---

## VALIDATOR PARALLEL WORK BY PHASE

### Phase 0 (while C1 and C2 build foundations)

```
[ ] Read architecture.md and verify it is internally consistent with guardrails.md
[ ] Create pre-filled AI review prompt for action_parser.py (for Phase 1)
[ ] Verify eval_config.json once created: 50 seeds, no overlap between training/held-out styles
[ ] Create W&B project "bluffbuster-examiner" — share with C1 and C2
[ ] Pre-populate Gate 0 checklist (see below)
[ ] Read OpenEnv documentation — know the exact base class interface BEFORE C1 needs it
```

### Phase 1 (while C1 builds environment core)

```
[ ] Prepare reward audit prompt (paste §6.2 from architecture.md)
[ ] Prepare OpenEnv inheritance check prompt
[ ] Prepare hidden state leak scan (for observation dict)
[ ] Run parser tests yourself: pytest tests/test_parser.py after C1 pushes
[ ] Run 7-style behavioral check: verify K1 != F1 responses statistically
[ ] Verify oracle calibration: check oracle_calibration.json values are realistic
```

### Phase 2 (while C2 builds training pipeline)

```
[ ] Scan reward_fn.py for any inline reward logic — must only call compute_reward()
[ ] Verify W&B logs by loading the run and checking metric keys
[ ] Run prompt_builder.py output through the hidden state scan
[ ] Verify DEBUG smoke test checklist (all 9 items)
```

### Phase 3 (while C2 runs DEMO training)

```
[ ] Open W&B dashboard — verify run is REAL (not a test run, has 200+ episodes)
[ ] Verify plot generation script reads from outputs/eval/final_metrics.json (not hardcoded)
[ ] Verify transcript selection is behavior-based (check select_transcripts.py code)
[ ] Verify comparison table includes BayesianHeuristicExaminer (not just random)
[ ] Check worst (style, section) cell in per_style_accuracy heatmap — document honestly
```

### Phase 4 (deployment)

```
[ ] Open HF Space in incognito — ALL 4 tabs must function
[ ] Run Colab notebook top-to-bottom on CLEAN runtime (not just importing)
[ ] Check: does Tab 1 show posterior trace? (Line chart updates per turn)
[ ] Check: does Tab 3 show real plots? (Not placeholder "results loading..." text)
[ ] Run full MSR checklist
```

### Phase 5 (storytelling)

```
[ ] Read writeup from non-technical judge perspective:
    - Is 3-sentence narrative verbatim? (Compare with guardrails.md §1)
    - Are both honest caveats present?
    - Does non-technical reader understand what the examiner does?
    - Is comparison table in writeup?
[ ] Final submission checklist (see submission_checklist.md)
```

---

## GATE REVIEW PROCEDURES

### 🔀 GATE 0 REVIEW

**Time budget: 15 minutes**

Run this review prompt:
```
Review the following code for "The Examiner" RL project.

File: examiner_env/action_parser.py
Check:
1. Does parse() return MalformedAction for ANY invalid input?
2. Does parse() ever try to coerce/fix malformed JSON? (FAIL if yes)
3. Does parse() use only json.loads() — not eval(), ast.literal_eval(), or regex extraction?
4. Does validate() return ValidationResult with penalties list?
5. Does validate() use Jaccard similarity for near-duplicate detection?

File: examiner_env/models.py  
Check:
1. Does AskAction use Pydantic v2 syntax? (model_validator, not v1 @validator)
2. Does ClassifyAction validate that ALL 10 sections are present?
3. Does StudentProfile have all required fields from architecture.md §5?

Return structured findings: PASS/FAIL for each check, with line numbers for failures.
```

**MSR gate check:**
- [ ] MSR-1 partial: environment skeleton exists with OpenEnv import
- [ ] MSR-2 partial: Colab skeleton exists, first cell is `!pip install`

**Manual checks:**
- Run `pytest tests/test_parser.py -v` → all 10 pass
- Open `eval_config.json` → 50 seeds, held-out styles not in training_styles

---

### 🔀 GATE 1 REVIEW — CRITICAL GATE (BLOCKS ALL TRAINING)

**Time budget: 45 minutes**

**Step 1: Run all unit tests**
```bash
pytest tests/ -v --tb=short
# Must exit 0. All tests pass.
```

**Step 2: Run the reward audit prompt**
```
Review examiner_env/reward.py against these exact specifications:

REQUIRED COMPONENTS (weights must be exact):
R_acc   = (1/N)*sum_s [+1 if correct else -1]                      ∈ [-1, +1]
R_asym  = -(0.5*FA + 0.3*FE)/N   where FA=KNOWS→FAKING, FE=FAKING→KNOWS  ∈ [-0.5, 0]
R_cal   = (0.4/N)*sum_s sign(correct-0.5)*|2*p_T(s)-1|             ∈ [-0.4, +0.4]
R_eff   = 0.20*max(0,(MAX_T-t)/MAX_T)*(1 if R_acc>0 else 0)        ∈ [0, +0.20]
R_cov   = -0.30*(1 if missing) - 0.05*(n_missing/10)               ∈ [-0.35, 0]
R_info  = 0.40*clip(sum_t ΔH_t, 0, 1)                              ∈ [0, +0.40]
R_qual  = 0.10*mean_asks(question_features)                         ∈ [0, +0.10]
R_div   = 0.05*(unique_sections/min(turns,N))                       ∈ [0, +0.05]
P_mal   = -0.20*n_malformed
P_rep   = -0.10*n_near_duplicates
P_inv   = -0.10*n_invalid_sections
R_total = sum of above, bounds [-2.05, +1.95], ASSERT finite

FAIL conditions:
- Any weight different from above
- R_total normalized inside this function
- LLR not clipped to [-3, +3] in posterior_oracle.py
- R_qual uses student_response as input
- compute_reward() calls any LLM
- Decomposition assertion missing or tolerance > 1e-9

Return PASS/FAIL per check with line numbers.
```

**Step 3: OpenEnv inheritance check**
```bash
python -c "
from examiner_env.environment import ExaminerEnv
import inspect
mro = [c.__name__ for c in ExaminerEnv.__mro__]
print('MRO:', mro)
assert 'OpenEnv' in str(ExaminerEnv.__bases__) or any('OpenEnv' in c.__name__ for c in ExaminerEnv.__mro__[1:]), \
  'ExaminerEnv does not inherit from OpenEnv!'
print('OpenEnv inheritance: PASS')
"
```

**Step 4: Hidden state leak scan**
```python
from examiner_env.environment import ExaminerEnv
from examiner_env.knowledge_base import KB
from training.config import DEBUG_CONFIG

env = ExaminerEnv(config=DEBUG_CONFIG, kb=KB)
obs, info = env.reset(seed=42)
obs_str = str(obs)

forbidden_patterns = ["KNOWS", "FAKING", "K1", "K2", "K3", "F1", "F2", "F3", "F4",
                      "true_labels", "style_assignment", "posterior"]
for pattern in forbidden_patterns:
    if pattern in obs_str:
        print(f"FAIL: '{pattern}' found in observation!")
        break
else:
    print("Hidden state leak scan: PASS")
```

**Step 5: Oracle calibration check**
```python
import json
cal = json.load(open("outputs/eval/oracle_calibration.json"))
brier = cal["calibration_metrics"]["mean_brier"]
acc = cal["calibration_metrics"]["terminal_accuracy"]
assert brier <= 0.18, f"FAIL: Oracle Brier {brier:.4f} > 0.18"
assert acc >= 0.75, f"FAIL: Oracle accuracy {acc:.4f} < 0.75"
print(f"Oracle calibration: PASS (Brier={brier:.4f}, Acc={acc:.4f})")
```

**Step 6: Reward variance check (non-constant)**
```python
from examiner_env.environment import ExaminerEnv
from examiner_env.baselines import RandomExaminer
from examiner_env.knowledge_base import KB
from training.config import DEBUG_CONFIG
import numpy as np

env = ExaminerEnv(config=DEBUG_CONFIG, kb=KB)
examiner = RandomExaminer()
rewards = []
for seed in range(20):
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = examiner.act(obs)
        import json
        obs, reward, term, trunc, info = env.step(json.dumps(action))
        done = term or trunc
    rewards.append(reward)

std_r = np.std(rewards)
assert std_r >= 0.05, f"FAIL: Reward variance {std_r:.4f} < 0.05 (reward is constant)"
assert std_r <= 1.5, f"FAIL: Reward variance {std_r:.4f} > 1.5 (reward is exploding)"
print(f"Reward variance: PASS (σ={std_r:.4f})")
```

**Step 7: Baseline sanity check (this is the hard credibility check)**
```python
from training.eval import run_eval
from examiner_env.baselines import RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer
from examiner_env.knowledge_base import KB
import json

eval_config = json.load(open("eval_config.json"))

r_metrics = run_eval(RandomExaminer(), eval_config, KB)
d_metrics = run_eval(DefinitionalExaminer(), eval_config, KB)
b_metrics = run_eval(BayesianHeuristicExaminer(KB), eval_config, KB)

# Sanity 1: Definitional > Random in accuracy
assert d_metrics["classification_accuracy"] > r_metrics["classification_accuracy"], \
    f"FAIL: DefinitionalExaminer {d_metrics['classification_accuracy']:.3f} <= RandomExaminer {r_metrics['classification_accuracy']:.3f}"
print(f"Baseline sanity 1: PASS (Def {d_metrics['classification_accuracy']:.3f} > Rand {r_metrics['classification_accuracy']:.3f})")

# Sanity 2: BayesianHeuristic > Definitional in avg_info_gain
assert b_metrics["avg_info_gain_per_turn"] > d_metrics["avg_info_gain_per_turn"], \
    f"FAIL: BayesHeuristic info_gain {b_metrics['avg_info_gain_per_turn']:.4f} <= Definitional {d_metrics['avg_info_gain_per_turn']:.4f}"
print(f"Baseline sanity 2: PASS (Bayesian info_gain > Definitional)")
```

**Gate 1 verdict form:**
```
Gate 1 Review | Date: [date]
All unit tests: [ PASS / FAIL — n failed ]
Reward audit:   [ PASS / FAIL — list failures ]
OpenEnv inheritance: [ PASS / FAIL ]
Hidden state leak:   [ PASS / FAIL ]
Oracle calibration:  [ PASS / FAIL — Brier=X, Acc=X ]
Reward variance:     [ PASS / FAIL — σ=X ]
Baseline sanity 1:   [ PASS / FAIL ]
Baseline sanity 2:   [ PASS / FAIL ]
MSR-1:               [ SATISFIED / FAIL ]

GATE 1 DECISION: [ CLEAR / BLOCKED ]
Blockers: [list any 🔴 issues]
Notes: [any 🟡 issues to log]
```

---

### 🔀 GATE 2 REVIEW

**Time budget: 20 minutes**

**Step 1: reward_fn.py delegation check**
```bash
grep -n "R_acc\|R_asym\|R_cal\|R_eff\|compute\|lambda_FA\|lambda_FE" training/reward_fn.py
# Should find ONLY: from examiner_env.reward import compute_reward
# Should NOT find: any reward formulas, any weights, any R_xxx computations
```

**Step 2: W&B per-component metrics check**
```python
import wandb
api = wandb.Api()
run = api.run("bluffbuster-examiner/[DEBUG_RUN_ID]")
metrics = list(run.history(keys=None).columns)
required = ["reward/R_acc", "reward/R_asym", "reward/R_cal", "reward/R_eff",
            "reward/R_cov", "reward/R_info", "reward/R_qual", "reward/R_div",
            "reward/P_malformed", "reward/P_repetition", "reward/P_invalid_sec"]
missing = [m for m in required if m not in metrics]
if missing:
    print(f"FAIL: Missing W&B metrics: {missing}")
else:
    print("W&B per-component metrics: PASS")
```

**Step 3: Prompt leakage check**
```python
from examiner_env.environment import ExaminerEnv
from training.prompt_builder import build_prompt
from examiner_env.knowledge_base import KB
from training.config import DEBUG_CONFIG
import re

env = ExaminerEnv(config=DEBUG_CONFIG, kb=KB)
obs, _ = env.reset(seed=42)
prompt = build_prompt(obs)

# Check for hidden state in prompt
for s_id in ["S01","S02","S03","S04","S05","S06","S07","S08","S09","S10"]:
    # Pattern: section ID followed by KNOWS or FAKING label
    if re.search(rf"{s_id}.*?: ?(KNOWS|FAKING)", prompt):
        print(f"FAIL: Hidden label for {s_id} in prompt!")
        break
for style in ["K1","K2","K3","F1","F2","F3","F4"]:
    if style in prompt:
        print(f"FAIL: Style ID '{style}' leaked into prompt!")
        break
else:
    print("Prompt leakage check: PASS")
```

**Step 4: DEBUG smoke test verification**
```
[ ] 20 episodes completed
[ ] R_total finite and in [-2.05, +1.95]: Check W&B run
[ ] σ(R_total) ∈ [0.05, 1.5]: Check W&B run
[ ] All 11 per-component metrics in W&B: verified above
[ ] parse_failure_rate < 0.5: Check W&B metric
[ ] Posterior trace logged (check env info dict): Y/N
[ ] Advantage mean ≈ 0, std ≈ 1: Check W&B advantage metrics
```

---

### 🔀 GATE 3 REVIEW

**Time budget: 30 minutes**

**Step 1: Verify training run is real**
```python
import wandb
api = wandb.Api()
run = api.run("bluffbuster-examiner/[DEMO_RUN_ID]")
history = run.history(keys=["reward/R_total_batch_mean"])
n_steps = len(history.dropna())
print(f"Training run has {n_steps} logged steps")
assert n_steps >= 150, f"FAIL: Only {n_steps} steps logged (expected ~200)"
print("Training run is real: PASS")
```

**Step 2: Verify plots are NOT hardcoded**
```bash
grep -n "0.8\|0.7\|0.65\|0.42" scripts/generate_plots.py
# If these numbers appear as literal floats (not in comments), investigate
# Plots must load data from JSON/W&B, not hardcode values
```

**Step 3: Transcript selection is behavior-based (not episode-index)**
```bash
grep -n "episode_index\|ep_idx\|episode_number\|sorted.*episode" scripts/select_transcripts.py
# Should NOT find: selection by episode index
grep -n "R_info\|diagnostic_score\|correctness\|correct.*False.*True" scripts/select_transcripts.py
# SHOULD find: selection by R_info gap and correctness flip
```

**Step 4: Comparison table completeness check**
```python
import json
final = json.load(open("outputs/eval/final_metrics.json"))
required_examiners = ["RandomExaminer", "DefinitionalExaminer", "BayesianHeuristicExaminer", "TrainedExaminer"]
for ex in required_examiners:
    assert ex in final, f"FAIL: {ex} missing from final_metrics.json"
    required_metrics = ["classification_accuracy", "avg_info_gain_per_turn", "calibration_ECE"]
    for m in required_metrics:
        assert m in final[ex], f"FAIL: {m} missing for {ex}"
print("Comparison table: PASS — all 4 examiners, all required metrics")
```

**Step 5: Phase 3 scientific honesty check**
```
[ ] Trained > Definitional in classification_accuracy on held-out eval?
    If YES: document as evidence
    If NO: document honestly — do NOT suppress. Note pipeline health was confirmed.
[ ] Worst (style, section) cell in per_style_accuracy identified and documented?
[ ] Plots derived from real W&B run (run ID matches outputs/plots/README.md)?
[ ] BayesianHeuristicExaminer included in comparison table?
```

---

### 🔀 GATE 4 REVIEW

**Time budget: 30 minutes**

**Step 1: HF Space incognito test (run this yourself)**
```
1. Open [HF_SPACE_URL] in a private/incognito browser window
2. Tab 1: Click "Run Episode"
   - [ ] Dialogue appears with section titles, questions, responses
   - [ ] Posterior trace chart updates after each turn
   - [ ] Per-turn info gain bar chart appears
   - [ ] Reward breakdown JSON appears after classify
3. Tab 2: Enter seed 1000, click "Run Comparison"
   - [ ] All 4 examiners shown (Random, Definitional, Bayesian, Trained)
   - [ ] Questions are visibly different per examiner
4. Tab 3: Training Evidence
   - [ ] R_total curve visible (not broken image)
   - [ ] Comparison bar chart visible
   - [ ] Per-style heatmap visible
   - [ ] Tab heading says "from real training run" or similar
5. Tab 4: Environment Details
   - [ ] Style family table visible with leak rates
   - [ ] Reward formula visible
```

**Step 2: Clean Colab test (run this yourself)**
```
1. Open new Colab session (Runtime → Disconnect and delete runtime)
2. Open notebook from GitHub URL (do not upload)
3. Run ALL cells top to bottom
4. [ ] Completes without error
5. [ ] W&B run initialized (new run visible in W&B project)
6. [ ] Produces at least 1 real plot
```

**Step 3: Video files check**
```bash
git ls-files | grep -E "\.mp4|\.mov|\.avi|\.mkv"
# Must return EMPTY
```

**Step 4: MSR checklist run (gates 1–4)**
```
MSR-1: [√ / ✗]
MSR-2: [√ / ✗]
MSR-3: [√ / ✗]
MSR-5: [√ / ✗]
MSR-6: [√ / ✗]
MSR-7: [√ / ✗]
MSR-9: [√ / ✗]
```

---

## RL-SPECIFIC MANUAL CHECK LIST (run at every gate)

This checklist is copied from `guardrails.md` §6 for convenience. Run ALL at every gate:

```
[ ] OpenEnv base class properly inherited? (MSR-1)
    Verify: ExaminerEnv.__bases__ contains OpenEnv class
[ ] Reward function matches §7 architecture.md pseudocode exactly?
    Verify: code review against exact weights and bounds
[ ] All 11 RewardBreakdown components present and individually logged to W&B?
    Verify: W&B dashboard shows 11 separate reward/* metrics
[ ] R_total within theoretical bounds [-2.05, +1.95] for every episode in DEBUG run?
    Verify: W&B min/max for reward/R_total
[ ] Reward variance σ(R_total) ∈ [0.05, 1.5] in DEBUG run?
    Verify: W&B std for reward/R_total over 20 episodes
[ ] Posterior oracle calibration JSON exists with mean Brier ≤ 0.18?
    Verify: oracle_calibration.json
[ ] Reward NOT normalized inside environment?
    Verify: grep -n "normalize\|z_score\|/ std\|/ sigma" examiner_env/reward.py → should be empty
[ ] Posterior oracle never used as accuracy ground truth?
    Verify: grep -n "true_label.*posterior\|p_T.*accuracy" examiner_env/reward.py → should be empty
[ ] LLR clipped to [-3, +3] per turn?
    Verify: grep -n "clip\|max.*min\|-3.*3\|3.*-3" examiner_env/posterior_oracle.py
[ ] Action parser has unit tests passing?
    Verify: pytest tests/test_parser.py exits 0
[ ] Student simulator leak rates match specification from guardrails.md?
    Verify: behavioral test (F1 under probe → ≤1 mechanism cue in 80%+)
[ ] Prompt builder has no hidden state leakage?
    Verify: prompt leakage check script above
[ ] W&B logging writing to W&B (not just print)?
    Verify: W&B dashboard shows new metrics after training step
[ ] Transcript selection is behavior-based, not episode-index?
    Verify: grep for episode_index in select_transcripts.py → empty
[ ] BayesianHeuristicExaminer present in baseline comparison?
    Verify: final_metrics.json has "BayesianHeuristicExaminer" key
[ ] Plots derived from real run data?
    Verify: plot generation script reads from JSON/W&B (not hardcoded)
[ ] Colab notebook cells have no local path assumptions?
    Verify: grep for "/home/\|/Users/\|C:\\\|D:\" notebooks/ → should be empty
[ ] HF Space Tab 1 shows live posterior trace?
    Verify: run Tab 1 manually in incognito
[ ] HF Space tested in incognito?
    Verify: Validator ran it personally
```

---

## BUG SEVERITY GUIDE

| Severity | Definition | Action |
|----------|------------|--------|
| 🔴 Blocker | Breaks an MSR or a judging criterion | **No merge**. Fix required. New review after fix. |
| 🟡 Degraded | Works but sub-optimal (missing component, weak baseline, cosmetic metric bug) | **Merge with note**. Log in `mistakes.md`. Follow-up task created. |
| 🟢 Minor | Cosmetic (typo, log label, unused variable) | **Merge**. No follow-up needed. |

**Examples:**
- 🔴 `reward_fn.py` re-implements reward formula inline → blocks MSR-3, PIPELINE criterion
- 🔴 Hidden partition in observation → breaks ENV_INNOV (information leak defeats the RL task)
- 🔴 `normalize(R_total)` inside `compute_reward()` → corrupts GRPO advantage computation
- 🔴 Mocked training plots → blocks MSR-3
- 🟡 Missing BayesianHeuristicExaminer in comparison table → weakens REWARD_EVIDENCE
- 🟡 Adaptive difficulty logic added (removed feature) → SCOPE_DRIFT
- 🟡 W&B metric labeled "reward" instead of "reward/R_total" → DEBUGGABILITY
- 🟢 Typo in section title → cosmetic

---

## SECOND AI OPINION PROTOCOL

**Use this when:**
- Reward logic is disputed (C1 and C2 disagree on a formula)
- OpenEnv integration is uncertain (base class interface unclear)
- Training results look anomalous (rewards all the same, all exploding, etc.)
- A gate condition is borderline (e.g., Brier = 0.17 vs 0.18)

**Protocol:**
1. Document the dispute/anomaly in 2 sentences
2. Run the disputed code through a second AI model (different from the one that wrote it)
3. Reconcile the two outputs — if they agree, proceed; if they disagree, default to `architecture.md` §7 (reward spec)
4. Log the reconciliation decision in `mistakes.md`

---

## FINAL SUBMISSION VALIDATION SEQUENCE (10 minutes max)

Run this sequence in ORDER before submitting:

```
1. [ ] Full MSR checklist (all 9) — every checkbox green
2. [ ] ENV_INNOV evidence: architecture.md documents 7-style simulator; HF Space Tab 4 shows reward decomposition
3. [ ] STORYTELLING evidence: 3-sentence narrative verbatim in README, HF Space, writeup; Tab 1+2 functional
4. [ ] REWARD_EVIDENCE: comparison table [4 examiners] exists; real plots; behavior-selected transcripts with posteriors
5. [ ] PIPELINE: reward pseudocode in architecture.md matches code; 11 components in W&B; Colab runs
6. [ ] Open HF Space incognito — live? Tab 1 runs a full episode? Tab 3 shows real plots?
7. [ ] Run Colab top-to-bottom on clean runtime — completes without error? W&B run started?
8. [ ] Open README — all MSR links present and working? Comparison table populated?
9. [ ] Run: git ls-files | grep -E ".mp4|.mov" → empty (no video files)
10. [ ] Read/skim writeup — honest caveats present? 3-sentence narrative present? Under 2 min read?
11. [ ] View reward curves in W&B — real run? Non-trivial variance?
12. [ ] Verify comparison table has DefinitionalExaminer (not just RandomExaminer) as primary baseline
13. [ ] Per-style accuracy heatmap published — worst cell identified and disclosed?
14. [ ] All 3 team members verbally confirm → SUBMIT
```

---

*Last updated: 2026-04-25 | Version 1.0 | Validator Playbook for BluffBuster / The Examiner*
