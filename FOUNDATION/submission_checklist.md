# submission_checklist.md — Final Pre-Submission Checklist
## The Examiner (BluffBuster) | Run in under 10 minutes | Last thing before submit

> **This is the Validator's final authority document. Every checkbox must be GREEN before submission.**
> **Any unchecked box = submission blocked. No exceptions.**

---

## MSR CHECKLIST (ALL 9 — EVERY ONE A SUBMISSION BLOCKER)

| # | MSR | Verification Command / Action | Owner | Status |
|---|-----|-------------------------------|-------|--------|
| MSR-1 | OpenEnv base class inherited (not reimplemented) | `python -c "from examiner_env.environment import ExaminerEnv; print([c.__name__ for c in ExaminerEnv.__mro__])"` → "OpenEnv" appears in MRO | C1 | [ ] |
| MSR-2 | Colab notebook runs top-to-bottom on clean runtime | Validator runs notebook on clean Colab session — all cells complete without modification or error | C2 | [ ] |
| MSR-3 | Real loss/reward plots from actual training run | `cat outputs/plots/README.md` shows W&B run ID → open that run → reward curve matches plot files | C2 | [ ] |
| MSR-4 | Writeup/blog/slides published and linked | Open writeup URL in incognito — accessible without login — reads under 2 min | ALL | [ ] |
| MSR-5 | HF Space live and runnable in incognito | Open HF Space URL in incognito — all 4 tabs load and run without error | C2 | [ ] |
| MSR-6 | README complete (all 6 sections present) | Open README.md — check: Problem, How It Works, What Agent Learns, Honest Caveats, Tech Stack, Links | ALL | [ ] |
| MSR-7 | README links to HF Space | `grep "huggingface.co/spaces" README.md` → non-empty; click link in incognito → Space loads | ALL | [ ] |
| MSR-8 | README links to all materials | Verify in README: blog/writeup link, Colab link, W&B link — all clickable and accessible | ALL | [ ] |
| MSR-9 | No video files in HF Hub submission | `git ls-files \| grep -E "\.mp4\|\.mov\|\.avi\|\.mkv"` → returns empty | C2 | [ ] |

---

## JUDGING CRITERIA EVIDENCE CHECKLIST

### ENV_INNOV (40%) — Primary Score Driver
- [ ] `architecture.md` documents 7-style simulator family with exact parametrized leak rates table
- [ ] `architecture.md` documents KB-grounded posterior oracle (LLR update equations, §7)
- [ ] `architecture.md` documents potential-based R_info shaping (Ng et al. 1999 citation)
- [ ] HF Space **Tab 4** shows: style family table with leak rates, reward component breakdown with formulas
- [ ] The novel claim is specific and defensible: *"information-gain reward shaping over KB-grounded posterior oracle to distinguish KNOWS/FAKING via diagnostic questioning"* — not generic

### STORYTELLING (30%) — Second Score Driver
- [ ] 3-sentence narrative appears **verbatim** in README header (compare with `guardrails.md` §1)
- [ ] 3-sentence narrative appears **verbatim** in HF Space description
- [ ] 3-sentence narrative appears **verbatim** in writeup
- [ ] HF Space **Tab 1** shows live posterior trace line chart updating per turn
- [ ] HF Space **Tab 2** shows all 4 examiners side-by-side (Random, Definitional, Bayesian, Trained)
- [ ] Demo sequence works: Tab 1 runs live episode → posterior trace visible → reward breakdown shown
- [ ] Writeup/blog has screenshot from Tab 2 (Baseline vs Trained comparison)
- [ ] Non-technical judge can understand the demo in under 2 minutes

### REWARD_EVIDENCE (20%)
- [ ] Comparison table exists with **all 4 examiners** [Random|Definitional|BayesianHeuristic|Trained]
- [ ] Table includes: accuracy, false_accusation_rate, avg_info_gain_per_turn, ECE
- [ ] Primary baseline is **DefinitionalExaminer** (not just random)
- [ ] **BayesianHeuristicExaminer** is included (credibility baseline)
- [ ] Per-component reward curves from real W&B run (R_acc, R_info, R_cal, R_qual, R_asym)
- [ ] Behavior-selected transcripts: `before_transcript.json` has wrong classification, `after_transcript.json` has correct classification, SAME episode seed
- [ ] Both transcripts include `posterior_trace` field populated
- [ ] Per-style accuracy heatmap published (7 styles × 10 sections, NaN for untested combos)
- [ ] Worst (style, section) cell identified and honestly disclosed in writeup

### PIPELINE (10%)
- [ ] Reward pseudocode in `architecture.md` matches `examiner_env/reward.py` exactly (spot-check 3 components)
- [ ] All 11 RewardBreakdown components visible as separate metrics in W&B
- [ ] R_total within theoretical bounds [−2.05, +1.95] — verified in W&B run
- [ ] σ(R_total) between floor (0.05) and ceiling (1.5) — verified in W&B run
- [ ] Oracle calibration exists: `outputs/eval/oracle_calibration.json` Brier ≤ 0.18
- [ ] Colab notebook runs without error (MSR-2 already checked above)

---

## SCIENTIFIC HONESTY FINAL CHECK

- [ ] Both honest caveats from PROJECT IDENTITY appear in README under "Honest Caveats" section
- [ ] Both honest caveats appear in writeup/blog
- [ ] Training improvement claim is on **HELD-OUT** styles/sections (not training distribution)
- [ ] Plots are derived from real W&B run data (run ID documented in `outputs/plots/README.md`)
- [ ] Transcripts selected by behavioral quality (not episode index) — verify `scripts/select_transcripts.py`
- [ ] Comparison uses DefinitionalExaminer as primary baseline (not just random)
- [ ] Calibration ECE reported alongside accuracy (no accuracy-only narrative)
- [ ] Worst per_style_accuracy cell disclosed honestly in writeup (not hidden)
- [ ] "World's first" or "traditional benchmarks are dead" language does NOT appear anywhere
- [ ] "3 visible training phases" language only used if phases genuinely appeared in real data

---

## TECHNICAL INTEGRITY CHECKS

Run these commands and verify results:

```bash
# Check 1: No video files
git ls-files | grep -E "\.mp4|\.mov|\.avi|\.mkv"
# Expected: empty output

# Check 2: mistakes.md not committed
git ls-files | grep "mistakes.md"
# Expected: empty output

# Check 3: secrets not committed
git ls-files | grep -E "\.env|credentials|api_key"
# Expected: empty output

# Check 4: Oracle calibration exists and is valid
python -c "
import json
cal = json.load(open('outputs/eval/oracle_calibration.json'))
brier = cal['calibration_metrics']['mean_brier']
acc = cal['calibration_metrics']['terminal_accuracy']
print(f'Brier: {brier:.4f} (need ≤0.18): {\"PASS\" if brier <= 0.18 else \"FAIL\"}')
print(f'Acc: {acc:.4f} (need ≥0.75): {\"PASS\" if acc >= 0.75 else \"FAIL\"}')
"

# Check 5: Final metrics exist with all required keys
python -c "
import json
fm = json.load(open('outputs/eval/final_metrics.json'))
required_examiners = ['RandomExaminer','DefinitionalExaminer','BayesianHeuristicExaminer','TrainedExaminer']
for ex in required_examiners:
    status = 'FOUND' if ex in fm else 'MISSING'
    print(f'{ex}: {status}')
"

# Check 6: All plots exist
ls outputs/plots/*.png
# Expected: 9 plot files listed

# Check 7: Transcripts have posterior_trace
python -c "
import json
before = json.load(open('outputs/transcripts/before_transcript.json'))
after = json.load(open('outputs/transcripts/after_transcript.json'))
print('before posterior_trace:', 'PRESENT' if 'posterior_trace' in before else 'MISSING')
print('after posterior_trace:', 'PRESENT' if 'posterior_trace' in after else 'MISSING')
print('Same seed:', before['episode_seed'] == after['episode_seed'])
print('before correct:', before.get('correct'))  # should be False
print('after correct:', after.get('correct'))    # should be True
"

# Check 8: 3-sentence narrative in README (verbatim check)
python -c "
narrative = 'Most AI benchmarks reward getting the right answer'
with open('README.md') as f:
    content = f.read()
print('Narrative in README:', 'FOUND' if narrative in content else 'MISSING')
"
```

---

## HF SPACE LIVE CHECK (Validator runs personally)

Open `[HF_SPACE_URL]` in **incognito/private browser**:

```
Tab 1 — Live Episode:
  [ ] Page loads without error
  [ ] "Run Episode" button is clickable
  [ ] After clicking: dialogue appears (section titles visible, questions/responses shown)
  [ ] Posterior trace line chart appears and updates per turn
  [ ] Per-turn info gain bar chart appears
  [ ] After classify: reward breakdown JSON appears
  [ ] Ground truth revealed at end

Tab 2 — Baseline vs Trained:
  [ ] All 4 examiners shown (Random, Definitional, BayesianHeuristic, Trained)
  [ ] Questions asked by Trained are visibly more targeted than Definitional
  [ ] Posterior traces visible for each examiner

Tab 3 — Training Evidence:
  [ ] R_total reward curve image loads (not broken)
  [ ] Comparison bar chart loads
  [ ] Per-style heatmap loads
  [ ] No placeholder text "Results loading..." or "Coming soon"
  [ ] Data matches values in final_metrics.json (spot check 1 value)

Tab 4 — Environment Details:
  [ ] Style family table with leak rates visible
  [ ] Action schema JSON examples visible
  [ ] Reward formula visible with bounds
```

---

## ALL-TEAM SIGN-OFF

Before submitting, all 3 team members must verbally confirm:

```
C1 confirms:
  [ ] All unit tests pass: pytest tests/ -v → exit 0
  [ ] Environment and simulator implemented exactly per architecture.md
  [ ] Reward function matches architecture.md §7 pseudocode exactly
  [ ] Oracle calibration completed with Brier ≤ 0.18

C2 confirms:
  [ ] Training run is real (W&B run ID documented)
  [ ] Plots generated from actual training data
  [ ] HF Space tested in incognito — all 4 tabs work
  [ ] Colab runs top-to-bottom on clean session
  [ ] No video files committed to HF Hub

Validator confirms:
  [ ] All 9 MSRs verified personally
  [ ] Gate 5 cleared
  [ ] Scientific honesty checks passed
  [ ] All 3 team members signed off above

SUBMIT: [ ] All three confirmed → submission completed
```

---

*This checklist must be completed in ≤10 minutes. If any item takes longer, escalate immediately.*
*Last updated: 2026-04-25 | Version 1.0*
