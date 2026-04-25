# guardrails.md — Supreme Authority Document
## The Examiner (BluffBuster) | Hackathon Project

> **This file supersedes all other documents in any conflict. Read this FIRST at every session start, every PR review, and before any merge.**

---

## SECTION 1: JUDGING CRITERIA ALIGNMENT MATRIX

### ENV_INNOV — 40% of Score (C1's primary ownership)

**Defensible Novel Claim:**
> "The Examiner is the first RL environment that frames diagnostic questioning as an information-theoretic task: the agent is rewarded for questions whose answers measurably shift a KB-grounded Bayesian posterior over a hidden KNOWS/FAKING partition, using potential-based information-gain shaping that is theoretically guaranteed not to bias the optimal policy."

**What makes this novel — specific and defensible:**
1. **7-style adversarial simulator family** with parametrized leak rates — not a single scripted student, not a hand-coded fake. Each style has deterministic but statistically heterogeneous response generation driven by `mechanism_cue_emit_rate` and `misconception_emit_rate`.
2. **Hidden-state diagnostic RL** — the agent never sees KNOWS/FAKING labels; it must infer the partition from question-answer evidence, creating a genuine information-gathering incentive.
3. **KB-grounded posterior oracle** (LLR update over KB cues/misconceptions) used exclusively for reward shaping — not as ground truth, not as an LLM judge. Fully deterministic and auditable.
4. **Potential-based R_info** via entropy reduction ΔH_t — theoretically grounded (Ng et al. 1999): total shaping = φ(p_T) − φ(p_0) where φ = −H, guaranteeing the optimal policy is unchanged versus terminal-only reward.
5. **Asymmetric error costs** in R_asym (λ_FA=0.5 > λ_FE=0.3) — encodes the real-world judgment that false accusations are worse than false exonerations.
6. **Calibration reward** R_cal — rewards confident-correct and punishes confident-wrong, incentivizing calibrated beliefs without requiring the model to emit explicit probabilities.

**Design decisions that exist solely to maximize ENV_INNOV:**
- 7 styles (K1/K2/K3/F1/F2/F3/F4) with independent per-section sampling — prevents style memorization and forces general diagnostic reasoning.
- Probe sensitivity: "why/how/edge-case" probes modulate faker leak rates by ×0.5 / ×1.5 — creates the diagnostic gap R_info can detect.
- Held-out style on eval suite — generalization to unseen deceptive styles is the primary novelty claim.

**What "innovative enough" looks like:**
- ✅ Environment where a question's VALUE is measured by information gain, not correctness of the question itself.
- ✅ Simulator family with parametrized statistical differentiation (not hand-crafted one-offs).
- ✅ Calibration component in reward (R_cal) — not just accuracy.
- ✅ Potential-based shaping with theoretical guarantee (cite Ng et al. 1999).

**What would be considered DERIVATIVE (trigger review):**
- ❌ Any environment mechanic that maps directly to CartPole/MuJoCo/Atari dynamics.
- ❌ Reward that rewards token-overlap with ground truth (removed — see Section 3).
- ❌ Single scripted student without style heterogeneity.
- ❌ Reward computed by an LLM judge.

**Innovation re-validation checkpoints:**
- Phase 0 Gate: confirm OpenEnv registration with novel env class name and description.
- Phase 1 Gate: Validator verifies 7 distinct behavioral signatures in `test_student_styles.py`.
- Phase 3 Gate: per-style accuracy heatmap confirms style differentiation is learnable.

---

### STORYTELLING — 30% of Score (All three, C2 leads deployment)

**Core 3-Sentence Narrative (verbatim — must appear in README, HF Space, writeup):**
> "Most AI benchmarks reward getting the right answer — but almost none reward asking the right question. The Examiner is an adversarial RL environment where an examiner agent learns, through information-gain reward shaping and calibrated terminal scoring, to design questions that expose confident bluffing across multiple deceptive student styles. We train a language model examiner using GRPO and demonstrate measurable improvement over definitional and random baselines on held-out student styles and unseen topic sections, with reward decomposition that judges can audit live."

**Demo sequence (ordered, non-negotiable):**
1. Environment running → show episode with visible hidden state
2. DefinitionalExaminer episode (weak) — "What is gradient descent?" style questions, wrong classification
3. TrainedExaminer episode (strong, same seed) — mechanism probes, faster confidence convergence, correct classification
4. Held-out eval results table — [Random|Definitional|BayesianHeuristic|Trained] × [accuracy|avg_info_gain|ECE]
5. Reward curve — real training run from W&B

**What non-technical judges must understand:**
1. Student may or may not know the material — examiner cannot see which.
2. Examiner asks questions; answers shift the examiner's belief (the "confidence dial").
3. Good questions move the needle; bad ones don't. Like a doctor's diagnostic test.
4. After training: sharper questions, faster confidence, more correct — visible in the per-turn posterior trace.

**Storytelling non-negotiables:**
- 3-sentence narrative is verbatim in README, HF Space description, and writeup. No paraphrase.
- Tab 2 (Baseline vs Trained) is the primary demo artifact — 4 examiners side-by-side on same seed.
- Tab 1 must show LIVE posterior trace ("doctor's confidence dial") — this is the visual hook.
- NO "3 visible training phases" unless phases genuinely emerge from real data.
- Writeup produced incrementally — not last-minute.
- Honest caveats from PROJECT IDENTITY appear in writeup (see below).

**Honest caveats (required verbatim in README and writeup):**
> "Our current simulator uses a controlled family of seven scripted styles, not real human experts — this is the primary generalization risk, which we mitigate with multi-style fakers, held-out style evaluation, and held-out topic sections."
> "We do not make claims about guaranteed short-run improvement — we demonstrate pipeline health, calibrated belief tracking, and behavioral shift on the held-out eval suite."
> "Our KB-grounded posterior is a deterministic surrogate, not a true Bayesian oracle — we use it for reward shaping only, never as ground truth for accuracy."

---

### REWARD_EVIDENCE — 20% of Score (C2 primary, C1 wires reward)

**Evidence we will produce (exact artifacts):**
1. `outputs/eval/baseline_metrics.json` — all 4 baselines on frozen eval suite BEFORE training.
2. `outputs/eval/checkpoint_metrics.json` — trained examiner every 50 steps.
3. `outputs/eval/final_metrics.json` — trained examiner after DEMO run on held-out style (F3) and held-out section (S05).
4. Comparison table: [Random|Definitional|BayesianHeuristic|Trained] × [accuracy|false_accusation_rate|false_exoneration_rate|avg_turns|avg_info_gain_per_turn|ECE|R_total_mean].
5. Reward curves (real W&B): R_total, per-component (R_acc, R_info, R_cal, R_qual, R_asym), accuracy, ECE.
6. Behavior-selected transcripts: DefinitionalExaminer wrong + TrainedExaminer correct on same seed, largest R_info gap.
7. Per-style accuracy heatmap (K1/K2/K3/F1/F2/F3/F4 × S01–S10) — worst cell disclosed honestly.

**Primary comparison:** DefinitionalExaminer (not random) — storytelling primary.
**Credibility comparison:** BayesianHeuristicExaminer — if TrainedExaminer doesn't beat this, RL gain is questionable.

---

### PIPELINE — 10% of Score (C2 owns pipeline, C1 owns reward wiring)

**Reward pseudocode in plain English (must match reward.py exactly):**
```
At episode end:
  R_acc   = mean over sections: +1 if correct, −1 if wrong
  R_asym  = −(0.5×false_accusations + 0.3×false_exonerations) / N
  R_cal   = (0.4/N) × sum: sign(correct−0.5) × |2×p_T(s)−1|
  R_eff   = 0.20 × max(0, (MAX_TURNS−turns_used)/MAX_TURNS) × 1[R_acc>0]
  R_cov   = −0.30×1[any section missing] − 0.05×(#missing/10)
  R_info  = 0.40 × clip(sum_t ΔH_t, 0, 1)
              ΔH_t = H(p_{t-1}) − H(p_t), H = mean per-section binary entropy
  R_qual  = 0.10 × mean_asks(0.40×mechanism_probe + 0.30×specificity + 0.30×edge_case)
  R_div   = 0.05 × (#unique_sections_asked / min(turns, N_sections))
  P_malformed = −0.20 × #malformed
  P_repetition = −0.10 × #near_duplicate_asks
  P_invalid_sec = −0.10 × #invalid_section_ids
  R_total = sum of all above. Bounds: [−2.05, +1.95]. Assert finite. NEVER normalize here.
```

**Smoke test definition (DEBUG config — what pipeline health means):**
- ✅ 20 episodes complete, no crash
- ✅ Every R_total finite and within [−2.05, +1.95]
- ✅ σ(R_total) ∈ [0.05, 1.5] — not collapsed, not exploding
- ✅ All 11 reward components logged separately to W&B
- ✅ `parse_failure_rate` < 0.5
- ✅ Posterior trace logged for ≥1 episode
- ✅ `oracle_calibration.json` exists, mean Brier ≤ 0.18
- ✅ Eval suite produces `baseline_metrics.json` with all 4 baselines
- ✅ GRPO advantage normalization fires (A_i mean ≈ 0, std ≈ 1 per group in W&B)
- ❌ NOT required: reward improvement in smoke test. Improvement measured in DEMO config on held-out eval suite.

---

## SECTION 2: MSR CHECKLIST (ALL 9 — SUBMISSION BLOCKERS)

| # | MSR | Owner | Phase | Verification |
|---|-----|-------|-------|--------------|
| MSR-1 | Use OpenEnv (latest release) — inherit, do not reimplement | C1 | 1 | `grep "class ExaminerEnv" environment.py` shows OpenEnv class in parent; no reimplementation of base methods |
| MSR-2 | Working training script using Unsloth/TRL GRPO as runnable Colab notebook | C2 | 4 | Validator runs Colab top-to-bottom on clean session, completes without modification or crash |
| MSR-3 | Real training evidence — loss and reward plots from actual run (not mocked) | C2 | 3 | Plots match W&B run ID logged in `outputs/plots/README.md`; verified against checkpoint metrics JSON |
| MSR-4 | Short writeup — HF mini-blog OR <2 min YouTube OR short slide deck | ALL | 5 | URL accessible without login; Validator confirms non-technical audience can follow in <2 min |
| MSR-5 | Environment pushed to HF Space — discoverable and runnable | C2 | 4 | HF Space URL opens in incognito; all 4 tabs load and run without error |
| MSR-6 | README complete — motivates problem, explains env, shows results | ALL | 4 | All 6 sections present; comparison table populated with real values |
| MSR-7 | README links to HF Space | ALL | 4 | Link works in incognito |
| MSR-8 | README links to all additional materials (blog, Colab, W&B) | ALL | 5 | All 3 links accessible without auth |
| MSR-9 | No large video files in HF Hub submission | C2 | 4 | `git ls-files \| grep -E ".mp4\|.mov\|.avi\|.mkv"` returns empty |

**Every MSR is a merge blocker AND submission blocker. Partial compliance = non-compliance.**

---

## SECTION 3: SCOPE GUARDRAILS

### IN SCOPE (exhaustive — derive from Base prompt only)

- `ExaminerEnv` class inheriting from OpenEnv base class
- `StudentProfile` with 7 styles (K1/K2/K3/F1/F2/F3/F4), all parametrized leak rates per §2
- 10 ML-theory sections (S01–S10), fully populated KB (key_concepts, mechanism_cues, common_misconceptions, probe_templates, evidence_weights, reference_responses)
- Action space: Ask (JSON) + Classify (JSON) — exactly these two, no others
- Strict JSON action parser with MalformedAction return (no coercion)
- Posterior oracle: KB-grounded LLR update, sigmoid posterior, ΔH_t computation — shaping only
- Oracle calibration: 200-episode held-out split, outputs `oracle_calibration.json`
- All 11 reward components: R_acc, R_asym, R_cal, R_eff, R_cov, R_info, R_qual, R_div, P_malformed, P_repetition, P_invalid_sec
- `RewardBreakdown` frozen dataclass with `posterior_trace` and `info_gain_per_turn`
- 3 baselines: RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer
- `question_features.py` — question-side R_qual only (no response input)
- Frozen eval suite (50 episodes, fixed seeds, held-out F3/F4 styles, held-out S09/S10 sections)
- GRPO training via Unsloth + TRL — DEBUG/DEMO/FULL config tiers
- W&B logging — all 11 reward components, posterior trace, advantage stats
- HF Space — 4 tabs as specified in §11
- Behavior-based transcript selection (largest R_info gap + correctness flip)
- Real evidence plots from W&B/training logs
- Colab notebook (MSR-2)
- `eval_config.json` — frozen seeds
- `context_primer.md` — ultra-compact session primer
- All `.md` planning documents
- README with 3-sentence narrative, honest caveats, comparison table, links

### OUT OF SCOPE (exhaustive — all removed in §12 of Base prompt)

- ❌ Adaptive difficulty / difficulty scaling every N episodes
- ❌ Episode-index transcript selection (episode 10 vs. 400)
- ❌ Hard assertion that N GRPO steps must show reward improvement
- ❌ "3 visible training phases" as a hardcoded demo narrative
- ❌ Token-overlap divergence as diagnostic quality metric
- ❌ Single-style fake student (only the 4-style FAKING family is valid)
- ❌ "World's first" / "traditional benchmarks are dead" overclaiming language
- ❌ Single-tier training config (must use DEBUG/DEMO/FULL)
- ❌ Symmetric ±1 per-section accuracy reward without R_asym + R_cal
- ❌ Step-level dense reward shaping not tied to information theory
- ❌ In-environment reward normalization / z-scoring inside `step()`
- ❌ Posterior oracle used as ground truth for accuracy grading
- ❌ LLM judge in reward computation
- ❌ "abstain" or "uncertain" in Classify action
- ❌ Loose text parsing for actions
- ❌ Custom training loops bypassing Unsloth/TRL
- ❌ Reimplementing OpenEnv base classes
- ❌ Local-only training with no W&B artifacts
- ❌ Video files committed to HF Hub
- ❌ Gradio version < 4.0
- ❌ Pydantic v1

### SCOPE CREEP TRIGGERS — Immediately escalate to Validator

| Trigger | Severity |
|---------|----------|
| AI adds LLM judge to reward computation | 🔴 Blocker |
| AI expands action space with "abstain" or "uncertain" | 🔴 Blocker |
| AI implements adaptive difficulty | 🟡 Degraded |
| AI selects transcripts by episode index | 🔴 Blocker |
| AI reimplements OpenEnv base classes instead of inheriting | 🔴 Blocker |
| AI uses token-overlap divergence for diagnosticity | 🔴 Blocker |
| AI generates single-style fake students | 🔴 Blocker |
| AI writes loose text parsers instead of strict JSON validators | 🔴 Blocker |
| AI normalizes reward inside `step()` | 🔴 Blocker |
| AI uses oracle posterior as accuracy ground truth | 🔴 Blocker |
| AI improvises reward weights (not from §6.2 verbatim) | 🔴 Blocker |
| AI implements R_qual using student_response | 🔴 Blocker |
| AI skips oracle calibration step | 🔴 Blocker |
| AI inlines reward logic in `reward_fn.py` | 🔴 Blocker |
| AI omits BayesianHeuristicExaminer from baselines | 🟡 Degraded |
| AI generates mocked/placeholder plots | 🔴 Blocker |
| AI adds video files to HF Hub | 🔴 Blocker |

---

## SECTION 4: TECH STACK GUARDRAILS (LOCKED — NO DEVIATIONS)

| Component | Technology | Version | Status |
|-----------|-----------|---------|--------|
| RL Environment | OpenEnv | latest release | LOCKED — inherit only, never reimplement |
| Training Framework | Unsloth + TRL (GRPO) | latest | LOCKED — no custom training loops |
| Training Notebook | Google Colab | — | LOCKED — must run top-to-bottom clean |
| Base Model | Qwen2.5-7B-Instruct (DEMO) / Qwen2.5-1.5B-Instruct (DEBUG) | current HF | LOCKED |
| Reward Logging | Weights & Biases | latest | LOCKED — must receive W&B metrics, not just print |
| HF Deployment | HuggingFace Spaces + Hub | — | LOCKED — no local-only deployments |
| Data Validation | Pydantic v2 | ≥2.0 | LOCKED — Pydantic v1 not acceptable |
| Demo UI | Gradio | ≥4.0 | LOCKED — Gradio <4 API is incompatible |
| Action Schema | JSON (strict) | — | LOCKED — no loose text parsing |
| Config | Python dataclasses + config.py | — | LOCKED — no hardcoded hyperparameters |

**BANNED patterns that trigger scope violations:**
- Custom training loops bypassing Unsloth/TRL → MSR-2 violation
- Reimplementing OpenEnv base classes → MSR-1 violation
- Local-only training with no W&B artifacts → MSR-3 violation
- Video files committed to HF Hub → MSR-9 violation
- LLM judge in reward computation → non-determinism violation
- Token-overlap divergence as diagnostic score → removed feature
- Loose text parsing for actions → removed feature

---

## SECTION 5: AI TOOL USAGE RULES

### Mandatory context to paste when prompting for specific modules

| Module | Context to include in prompt |
|--------|------------------------------|
| Action parser code | Paste exact JSON schemas for Ask and Classify (§4 of Base prompt) |
| Simulator code | Paste full StudentProfile structure (§2 of Base prompt) and 7-style leak rate table |
| Reward function code | Paste §6.2 verbatim — reward pseudocode — do not summarize |
| Environment methods | Paste OpenEnv base class interface signature — AI will hallucinate it otherwise |
| Section-referencing code | Always include canonical IDs: S01–S10 |
| Posterior oracle code | Paste §6.1 equations verbatim (LLR formula, sigmoid update) |
| Training script | Paste current TRL GRPOTrainer interface from docs — version-pin |
| Prompt builder | State explicitly: "Do NOT leak hidden partition, style IDs, or posterior values into the prompt" |

### Re-prompt policy
- Maximum **2 re-prompt attempts** before escalating to Validator.
- Re-prompt strategy: paste the failing sanity condition explicitly, add "Do not X" for each violation observed, simplify scope of request.
- If re-prompt 2 fails: document in `mistakes.md`, notify Validator, try alternative approach.

---

## SECTION 6: CODE QUALITY NON-NEGOTIABLES (RL-SPECIFIC)

1. **All hyperparameters in `training/config.py`** — never hardcoded in any other file. This is a 🟡 degraded violation.
2. **Every training run emits to W&B** — not just print statements. W&B must be initialized and receive real metrics. This is a 🔴 blocker for MSR-3.
3. **Reward function must exactly match §6.2 pseudocode** in `architecture.md` — no AI improvisation of weights, bounds, or component formulas.
4. **Action parser must have unit tests passing before any training run begins.** `tests/test_parser.py` is a Phase 1 gate blocker.
5. **Student style family must have behavioral tests passing before training run begins.** `tests/test_student_styles.py` is a Phase 1 gate blocker.
6. **No placeholder reward returns** — `return 0.0` or `return None` anywhere in `reward.py` is a 🔴 blocker.
7. **Colab notebook runs top-to-bottom with no modifications** — every cell must work on a clean Colab runtime without local file assumptions. MSR-2 blocker.
8. **HF Space tested in incognito before final submission** — not just in the Space editor. MSR-5 blocker.
9. **R_total must be asserted finite and within [−2.05, +1.95]** at the end of `compute_reward()` — raise `ValueError` on violation to catch bugs early.
10. **`compute_reward()` must never call any LLM** — all components are programmatic.
11. **Decomposition contract**: `R_acc + R_asym + R_cal + R_eff + R_cov + R_info + R_qual + R_div + P_malformed + P_repetition + P_invalid_sec == R_total ± 1e-9`. Assert in unit tests.

---

## SECTION 7: GIT NON-NEGOTIABLES

1. **No direct commits to `main`** — all changes via feature branches and PR.
2. **No force push** — ever. Escalate to team if history needs correction.
3. **Structured commit messages** (see `merge_procedure.md` for format).
4. **File ownership is absolute** — C1 does not commit to `training/`, C2 does not commit to `examiner_env/`. Cross-ownership requires explicit Validator approval.
5. **`mistakes.md` is NEVER pushed to GitHub** — local only (add to `.gitignore`). It contains session-specific error logs that must not pollute the public repo.
6. **HF Space pushes are separate from GitHub commits** — C2 owns HF Space pushes, only after Validator gate clearance.
7. **No HF Hub commits with video files** — MSR-9.
8. **PR must be assigned to Validator for review** before any merge to `main`.
9. **Squash merge only** — clean history on `main`.

---

## SECTION 8: STORYTELLING NON-NEGOTIABLES

1. 3-sentence narrative is verbatim in README, HF Space description, and writeup. Never paraphrase.
2. Demo sequence is always: environment running → DefinitionalExaminer weak → TrainedExaminer strong → held-out table → reward curve.
3. NO "3 visible training phases" narration unless phases genuinely emerge from real W&B data.
4. Writeup produced **incrementally** (by Phase 3 draft, by Phase 5 final) — not assembled at submission deadline.
5. Honest caveats from PROJECT IDENTITY section must appear in writeup — both of them, verbatim.
6. Comparison baseline for storytelling: **DefinitionalExaminer** (not random — random is too easy).
7. Tab 2 (Baseline vs Trained, same seed) is the **primary visual storytelling artifact** — make it compelling.
8. Never claim "the agent learned X" without pointing to a specific metric (Δaccuracy on held-out style, Δavg_info_gain_per_turn, ΔECE).
9. Scientific honesty section §14 of Base prompt applies to ALL documents and ALL prompts.

---

## SECTION 9: SESSION START RITUAL

**Run through these steps at the start of every AI coding session, in order:**

1. `git pull origin main` — ensure on latest
2. Read `mistakes.md` — note active violations, current error index count
3. Check MSR status (which are open, which are closed)
4. Confirm file ownership for this session (C1 / C2 / Validator)
5. Check current phase in `implementation_plan.md`
6. Paste `context_primer.md` into AI tool at session start
7. State the specific task for this session
8. Confirm "DO NOT" list from `context_primer.md` applies to current task

---

## SECTION 10: SESSION END RITUAL

**Run through these steps before ending every AI coding session:**

1. All sanity checks for completed tasks: 3-condition verification per task
2. Update MSR status if any MSR was closed this session
3. Log any AI mistakes to `mistakes.md` (do NOT commit this file)
4. Commit completed work to feature branch with structured commit message
5. Update progress notes in `implementation_plan.md` (mark tasks done)
6. If at a gate: notify Validator for gate review
7. If handing off to other coder: write 2-sentence handoff note with current state + what they need to know

---

## SECTION 11: GUARDRAILS VERSIONING

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-04-25 | Initial version — aligned to upgraded architecture (Base prompt Opus edition) |

**Rule: `guardrails.md` may only be updated if ALL three team members agree AND a Validator gate has been cleared. No unilateral edits.**

---

*Last updated: 2026-04-25 | Version 1.0 | Source of truth: Base prompt.txt (Opus edition)*
