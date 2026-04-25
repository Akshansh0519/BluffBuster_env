# project_structure.md — Annotated File Tree
## The Examiner (BluffBuster) | Locked After Phase 0

> **This file is locked after Phase 0. No new files may be added without Validator approval.**
> **Every file has 5 annotation fields: purpose | owner | feature | MSR | judging criterion**

---

## COMPLETE FILE TREE

```
BluffBuster/
│
├── examiner_env/                          ← C1 OWNS ALL FILES IN THIS DIRECTORY
│   ├── __init__.py                        # OpenEnv registration; package init
│   ├── action_parser.py                   # Strict JSON parser: parse() → Action|MalformedAction; validate()
│   ├── baselines.py                       # RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer
│   ├── calibration.py                     # Oracle calibration: fits (α,β,γ) on 200-ep split; saves oracle_calibration.json
│   ├── environment.py                     # ExaminerEnv: OpenEnv-inheriting main env class; reset/step
│   ├── knowledge_base.py                  # 10 ML sections × KB (mechanism_cues, misconceptions, probe_templates)
│   ├── models.py                          # Pydantic v2: AskAction, ClassifyAction, StudentProfile, EpisodeState, etc.
│   ├── posterior_oracle.py                # LLR scorer, sigmoid update, ΔH_t; PosteriorTracker class
│   ├── question_features.py               # R_qual computation: question-side ONLY (no response argument)
│   ├── reward.py                          # All 11 reward components; compute_reward(); RewardBreakdown dataclass
│   └── student.py                         # 7-style simulator family with parametrized leak rates; generate_response()
│
├── tests/                                 ← C1 OWNS ALL FILES IN THIS DIRECTORY
│   ├── __init__.py
│   ├── test_parser.py                     # Parser unit tests: 10 test cases (8 malformed + 2 valid + near-duplicate)
│   ├── test_posterior_oracle.py           # Oracle determinism, LLR bounds, calibration, ΔH_t tests
│   ├── test_reward.py                     # Reward component isolation, bounds, decomposition contract
│   └── test_student_styles.py             # 7-style behavioral differentiation, reproducibility, non-empty
│
├── training/                              ← C2 OWNS ALL FILES IN THIS DIRECTORY
│   ├── __init__.py
│   ├── config.py                          # DEBUG_CONFIG, DEMO_CONFIG, FULL_CONFIG dataclasses (all hyperparameters)
│   ├── eval.py                            # Frozen eval suite runner: run_eval() → EvalMetrics; ECE/Brier/per_style
│   ├── prompt_builder.py                  # build_prompt(): section titles + history + JSON schema (NO hidden state)
│   ├── reward_fn.py                       # GRPO reward bridge: calls examiner_env.reward (does NOT re-implement reward)
│   └── train_grpo.py                      # Unsloth + TRL GRPOTrainer; W&B logging; checkpoint/eval hooks
│
├── scripts/                               ← C2 OWNS ALL FILES IN THIS DIRECTORY
│   ├── generate_plots.py                  # Generates all 9 required plots from real W&B/eval data
│   └── select_transcripts.py             # Behavior-based transcript selection: largest R_info gap + correctness flip
│
├── hf_space/                              ← C2 OWNS ALL FILES IN THIS DIRECTORY
│   ├── app.py                             # Gradio ≥4.0 app: 4 tabs (Live Episode, Comparison, Evidence, Details)
│   └── README.md                          # HF Space model card with required YAML front matter
│
├── notebooks/                             ← C2 OWNS ALL FILES IN THIS DIRECTORY
│   └── train_examiner.ipynb               # Colab notebook: install → calibrate → baseline eval → train → eval → plots
│
├── outputs/                               ← C2 OWNS CONTENTS, STRUCTURE LOCKED
│   ├── eval/
│   │   ├── baseline_metrics.json          # All 3 baselines on frozen eval suite (pre-training)
│   │   ├── checkpoint_metrics.json        # Trained examiner at every 50-step checkpoint
│   │   ├── final_metrics.json             # Trained examiner after DEMO run (held-out F3, S05)
│   │   ├── oracle_calibration.json        # Calibrated (α,β,γ) per section; mean Brier; terminal accuracy
│   │   ├── reference_cache.pkl            # Pre-computed KB response fingerprints (deterministic)
│   │   └── reward_bounds.json             # Random-rollout bounds computed at training start
│   ├── plots/
│   │   ├── README.md                      # W&B run ID that generated these plots (required for MSR-3 audit)
│   │   ├── reward_curve.png               # R_total mean ± std over training checkpoints
│   │   ├── reward_components.png          # Small-multiples: R_acc, R_info, R_cal, R_qual, R_asym
│   │   ├── accuracy_curve.png             # Classification accuracy over training steps
│   │   ├── false_rates_curve.png          # False accusation + false exoneration rates overlaid
│   │   ├── info_gain_curve.png            # avg_info_gain_per_turn over training steps
│   │   ├── calibration_ece_curve.png      # Calibration ECE over training steps
│   │   ├── comparison_bar.png             # [4 examiners] × [accuracy + info_gain + ECE] bar chart
│   │   ├── per_style_heatmap.png          # Accuracy heatmap: 7 styles × 10 sections
│   │   └── posterior_trace_example.png   # Per-section p_t over turns for best AFTER transcript
│   └── transcripts/
│       ├── before_transcript.json         # DefinitionalExaminer on selected seed: wrong, low R_info, posterior_trace
│       └── after_transcript.json          # TrainedExaminer on same seed: correct, high R_info, posterior_trace
│
├── README.md                              ← SHARED (all three) — judging entry point; MSR-6/7/8
├── eval_config.json                       ← SHARED — frozen 50 seeds; held-out styles; held-out sections; NEVER modify
├── requirements.txt                       ← SHARED — all Python dependencies with minimum versions
│
├── guardrails.md                          ← VALIDATOR authority document — read first every session
├── architecture.md                        ← Technical reference — reward spec source of truth
├── implementation_plan.md                 ← Master phases/tasks/gates — all three coders
├── implementation_coder1.md               ← C1 self-contained playbook
├── implementation_coder2.md               ← C2 self-contained playbook
├── implementation_validator.md            ← Validator QC playbook
├── merge_procedure.md                     ← Git protocol
├── project_structure.md                   ← This file — LOCKED after Phase 0
├── submission_checklist.md                ← Pre-submission final checklist
├── context_primer.md                      ← Ultra-compact session primer (paste into AI at start)
│
└── mistakes.md                            ← LOCAL ONLY — in .gitignore — NEVER commit
```

---

## FULL ANNOTATION TABLE

| File Path | Purpose | Owner | Feature | MSR | Judging Criterion |
|-----------|---------|-------|---------|-----|-------------------|
| `examiner_env/__init__.py` | OpenEnv registration; package init | C1 | Environment compliance | MSR-1 | ENV_INNOV |
| `examiner_env/action_parser.py` | Strict JSON parser; MalformedAction | C1 | Action validation | MSR-1 | ENV_INNOV / PIPELINE |
| `examiner_env/baselines.py` | Random, Definitional, BayesianHeuristic examiners | C1 | Baseline comparison | MSR-3 partial | REWARD_EVIDENCE |
| `examiner_env/calibration.py` | Oracle (α,β,γ) calibration; outputs oracle_calibration.json | C1 | Oracle correctness | — | PIPELINE |
| `examiner_env/environment.py` | OpenEnv-inheriting ExaminerEnv; reset/step | C1 | Core RL env | MSR-1 | ENV_INNOV |
| `examiner_env/knowledge_base.py` | 10-section KB; cues, misconceptions, probes | C1 | Diagnostic signal | MSR-1 partial | ENV_INNOV |
| `examiner_env/models.py` | Pydantic v2 schemas; all data types | C1 | Data validation | — | PIPELINE |
| `examiner_env/posterior_oracle.py` | LLR + posterior update + ΔH_t | C1 | Info-gain signal | — | ENV_INNOV / PIPELINE |
| `examiner_env/question_features.py` | R_qual question-side scorer | C1 | Reward component | — | PIPELINE |
| `examiner_env/reward.py` | All 11 components; RewardBreakdown | C1 | Reward function | MSR-3 partial | REWARD_EVIDENCE / PIPELINE |
| `examiner_env/student.py` | 7-style simulator; parametrized leak rates | C1 | Adversarial simulator | MSR-1 | ENV_INNOV |
| `tests/test_parser.py` | 10 parser test cases | C1 | Correctness gate | — | PIPELINE |
| `tests/test_posterior_oracle.py` | Oracle determinism + calibration | C1 | Correctness gate | — | PIPELINE |
| `tests/test_reward.py` | Reward component isolation + bounds | C1 | Correctness gate | — | PIPELINE / REWARD_EVIDENCE |
| `tests/test_student_styles.py` | 7-style behavioral differentiation | C1 | ENV_INNOV gate | — | ENV_INNOV |
| `training/config.py` | DEBUG/DEMO/FULL hyperparameter dataclasses | C2 | Training config | MSR-2 | PIPELINE |
| `training/eval.py` | Frozen eval suite runner; all 15 metrics | C2 | Evidence generation | MSR-3 | REWARD_EVIDENCE |
| `training/prompt_builder.py` | Examiner prompt; no hidden state | C2 | Prompt engineering | — | ENV_INNOV / PIPELINE |
| `training/reward_fn.py` | GRPO reward bridge → examiner_env.reward | C2 | Training wiring | MSR-3 | PIPELINE |
| `training/train_grpo.py` | Unsloth + TRL GRPOTrainer; W&B hooks | C2 | Training pipeline | MSR-2, MSR-3 | PIPELINE / REWARD_EVIDENCE |
| `scripts/generate_plots.py` | All 9 evidence plots from real data | C2 | Evidence artifacts | MSR-3 | REWARD_EVIDENCE / STORYTELLING |
| `scripts/select_transcripts.py` | Behavior-based transcript selection | C2 | Before/after demo | — | STORYTELLING / REWARD_EVIDENCE |
| `hf_space/app.py` | Gradio 4-tab demo app | C2 | Live demo | MSR-5 | STORYTELLING / REWARD_EVIDENCE |
| `hf_space/README.md` | HF Space model card with YAML front matter | C2 | Space config | MSR-5 | STORYTELLING |
| `notebooks/train_examiner.ipynb` | End-to-end Colab notebook | C2 | Training artifact | MSR-2 | PIPELINE |
| `outputs/eval/baseline_metrics.json` | Pre-training baseline comparison | C2 | Evidence | MSR-3 | REWARD_EVIDENCE |
| `outputs/eval/checkpoint_metrics.json` | Per-checkpoint trained examiner metrics | C2 | Training progress | MSR-3 | REWARD_EVIDENCE |
| `outputs/eval/final_metrics.json` | Final held-out eval results | C2 | Primary evidence | MSR-3 | REWARD_EVIDENCE |
| `outputs/eval/oracle_calibration.json` | Calibrated oracle weights | C1 | Oracle quality | — | PIPELINE |
| `outputs/eval/reference_cache.pkl` | KB response fingerprints | C1 | Oracle init | — | ENV_INNOV |
| `outputs/eval/reward_bounds.json` | Random-rollout reward bounds | C2 | Sanity bounds | — | PIPELINE |
| `outputs/plots/README.md` | W&B run ID provenance | C2 | MSR-3 audit trail | MSR-3 | REWARD_EVIDENCE |
| `outputs/plots/*.png` | All 9 evidence plots (real data) | C2 | Demo artifacts | MSR-3 | REWARD_EVIDENCE / STORYTELLING |
| `outputs/transcripts/before_transcript.json` | Definitional examiner weak episode | C2 | Before/after demo | — | STORYTELLING |
| `outputs/transcripts/after_transcript.json` | Trained examiner strong episode | C2 | Before/after demo | — | STORYTELLING |
| `README.md` | Project entry point; links; comparison table | ALL | Submission artifact | MSR-6, 7, 8 | ALL |
| `eval_config.json` | Frozen 50-seed eval suite configuration | ALL | Reproducibility | — | REWARD_EVIDENCE |
| `requirements.txt` | Python dependencies | ALL | Environment setup | MSR-2 | PIPELINE |
| `guardrails.md` | Supreme authority; MSR checklist; scope | VAL | Project governance | ALL | ALL |
| `architecture.md` | Technical reference; reward spec | VAL | Architecture | ALL | ALL |
| `implementation_plan.md` | Phase/task/gate plan | ALL | Project management | — | — |
| `implementation_coder1.md` | C1 playbook | C1 | C1 reference | — | — |
| `implementation_coder2.md` | C2 playbook | C2 | C2 reference | — | — |
| `implementation_validator.md` | Validator playbook | VAL | QC reference | — | — |
| `merge_procedure.md` | Git protocol | VAL | Git governance | — | — |
| `project_structure.md` | This file | VAL | Structure reference | — | — |
| `submission_checklist.md` | Pre-submit checklist | VAL | Final gate | ALL | ALL |
| `context_primer.md` | Ultra-compact AI session primer | ALL | AI tool usage | — | — |
| `mistakes.md` | LOCAL ONLY — AI error log | ALL | Error prevention | — | — |

---

## FILE OWNERSHIP RULES

1. **C1** may only create/edit files in `examiner_env/` and `tests/`
2. **C2** may only create/edit files in `training/`, `scripts/`, `hf_space/`, `notebooks/`, `outputs/`
3. **ALL** may edit `README.md`, `eval_config.json`, `requirements.txt`
4. **No new files** may be added after Phase 0 without explicit Validator approval (open a PR with just the new file, explain why it's needed, Validator reviews)
5. **Cross-ownership** edits require Validator to approve the PR separately for the cross-owned file

---

## LOCKED STATUS

> **This file is LOCKED after Phase 0 Gate is cleared.**
> Any structural changes require Validator approval and a dedicated PR.
> Adding files without approval is a scope violation (🟡 Degraded).

---

*Last updated: 2026-04-25 | Version 1.0 | Locked at Phase 0 completion*
