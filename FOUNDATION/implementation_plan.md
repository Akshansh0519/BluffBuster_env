# implementation_plan.md — Master Implementation Reference
## The Examiner (BluffBuster) | Phase-by-Phase Build Plan

> **Reading order:** `guardrails.md` → `context_primer.md` → this file → coder-specific doc (coder1 or coder2)

---

## HOW TO USE THIS DOCUMENT

- Every task has: **Goal | AI Prompt | Context | Expected Output | Judging Criterion | MSR | Time | Dependencies | Parallel | Sanity (3 conditions) | Re-prompt Strategy | Gate Flag**
- Tasks marked 🔀 are **merge gate** items — Validator must clear these before work continues.
- Tasks marked ⚡ can run in **parallel** with listed peer tasks.
- Sanity check = 3 binary conditions that must ALL be true before marking task complete.
- Status key: `[ ]` pending, `[→]` in progress, `[✓]` complete, `[🔴]` blocked

---

## STATUS TRACKER (Update each session)

| Phase | Gate Status | C1 Status | C2 Status | Date Cleared |
|-------|------------|-----------|-----------|--------------|
| Phase 0 | [ ] | [ ] | [ ] | — |
| Phase 1 | [ ] | [ ] | [ ] | — |
| Phase 2 | [ ] | [ ] | [ ] | — |
| Phase 3 | [ ] | [ ] | [ ] | — |
| Phase 4 | [ ] | [ ] | [ ] | — |
| Phase 5 | [ ] | [ ] | [ ] | — |

---

## PHASE 0 — FOUNDATION (Both Coders in Parallel)

> **Goal:** Repo structure, base schemas, skeleton code, W&B live, eval config created, DEBUG config defined.
> **Time estimate:** 2–3 hours (parallel)
> **Gate condition:** C1 can parse Ask and Classify JSON. C2 has Colab skeleton running. W&B receives a test metric.

---

### Task C1-0.1: KB Construction ⚡ (parallel with C2-0.1)

**Goal:** Implement `examiner_env/knowledge_base.py` — all 10 sections, each with key_concepts, mechanism_cues (tagged with cue_strength), common_misconceptions (tagged with misconception_severity), probe_templates (tagged with probe_type), evidence_weights, reference_responses.

**AI Prompt (paste-ready):**
```
Implement examiner_env/knowledge_base.py for "The Examiner" RL project.

Return a Python module that defines a KnowledgeBase dataclass (Pydantic v2) and builds
a dict[str, SectionKB] for all 10 sections.

Sections: S01 Gradient Descent, S02 Backpropagation, S03 Overfitting/Regularization,
S04 Attention Mechanisms, S05 Transformer Architecture, S06 Loss Functions/Geometry,
S07 Batch Normalization, S08 CNNs, S09 RL Basics, S10 Embeddings/Representation Learning.

Each SectionKB must have:
- key_concepts: list[str] (5–8 precise ML concepts)
- mechanism_cues: list[CuedPhrase] where CuedPhrase = (phrase: str, cue_strength: Literal["weak", "strong"])
  strong=1.0, weak=0.5. Include phrases like "chain rule", "vanishing gradient", "Jacobian" for S02.
- common_misconceptions: list[MisconPhrase] where MisconPhrase = (phrase: str, severity: Literal["minor", "major"])
  major=1.0, minor=0.5. These are shallow/wrong phrases fakers use.
- probe_templates: list[ProbeTemplate] where ProbeTemplate = (template: str, probe_type: Literal["definitional","mechanism","edge_case","counterexample","application"])
  Include 3–5 templates per section. Mechanism/edge_case probes are most valuable.
- evidence_weights: dict with keys "alpha"(default 1.5), "beta"(default 0.5), "gamma"(default 1.0)
- reference_responses: dict[str, str] keyed by (style, probe_type) — placeholder empty strings are acceptable.

Use Pydantic v2. Export: KB: dict[str, SectionKB] = build_kb()
Do not import anything outside standard library + pydantic.
```

**Context:** `architecture.md` §5 (KB structure), §3 of Base prompt (section list)
**Expected output:** `examiner_env/knowledge_base.py` with 10 fully populated sections
**Judging criterion:** ENV_INNOV (40%) — KB is the foundation of diagnostic quality
**MSR:** MSR-1 (partial)
**Time:** 45 min
**Dependencies:** None (first task)

**Sanity:**
1. All 10 section IDs (S01–S10) present in `build_kb()` output
2. Each section has ≥3 probe_templates; at least 1 is `probe_type="mechanism"` or `"edge_case"`
3. `mechanism_cues` for S02 includes "chain rule", "Jacobian", or "gradient" — not generic filler

**Re-prompt strategy:** If mechanism_cues are too generic, add: "Use highly specific ML-domain phrases. 'Forward pass' alone is NOT a mechanism cue. 'Jacobian of the loss wrt layer activations' IS."

---

### Task C1-0.2: Pydantic Action Schemas (`models.py`) ⚡

**Goal:** Strict Pydantic v2 models for AskAction, ClassifyAction, StudentProfile, EpisodeState, ValidationResult, MalformedAction, RewardBreakdown.

**AI Prompt (paste-ready):**
```
Implement examiner_env/models.py using Pydantic v2.

Required models:

class AskAction(BaseModel):
    action_type: Literal["ask"]
    section_id: str  # validated: must be in {"S01",...,"S10"}
    question_text: str  # validated: min_length=10, non-empty

class ClassifyAction(BaseModel):
    action_type: Literal["classify"]
    classifications: dict[str, Literal["KNOWS", "FAKING"]]
    # validated: all 10 keys S01–S10 must be present
    # validated: all values exactly "KNOWS" or "FAKING" (case-sensitive)

class MalformedAction(BaseModel):
    reason: str  # description of why parsing failed

class StudentProfile(BaseModel):
    knowledge_mode: Literal["KNOWS", "FAKING"]
    style: Literal["K1", "K2", "K3", "F1", "F2", "F3", "F4"]
    section_id: str
    verbosity: Literal["brief", "medium", "verbose"]
    confidence_pattern: Literal["hedging", "neutral", "confident"]
    mechanism_cue_emit_rate: float  # 0.0–1.0
    misconception_emit_rate: float  # 0.0–1.0
    style_specific_params: dict
    seed: int

class ValidationResult(BaseModel):
    valid: bool
    penalties: list[str]  # e.g. ["P_repetition", "P_invalid_sec"]
    info: dict

class EpisodeState(BaseModel):
    episode_seed: int
    section_ids: list[str]  # all 10 canonical section IDs
    true_labels: dict[str, Literal["KNOWS", "FAKING"]]  # hidden from examiner
    style_assignments: dict[str, str]  # hidden from examiner
    dialogue_history: list[dict]  # (section_id, question, response) tuples
    turn: int
    max_turns: int
    done: bool

Use Pydantic v2 field_validator for cross-field validation.
CANONICAL_SECTIONS = ["S01","S02","S03","S04","S05","S06","S07","S08","S09","S10"]
```

**Context:** `architecture.md` §6 (action schemas), §5 (StudentProfile)
**Expected output:** `examiner_env/models.py` with all 7 models, validators passing
**Judging criterion:** ENV_INNOV (structural correctness)
**MSR:** MSR-1 (partial)
**Time:** 30 min
**Dependencies:** None

**Sanity:**
1. `AskAction(action_type="ask", section_id="S11", question_text="x")` raises `ValidationError`
2. `ClassifyAction` with 9/10 sections raises `ValidationError`
3. `StudentProfile` with `style="F5"` raises `ValidationError`

---

### Task C1-0.3: Action Parser Skeleton (`action_parser.py`) ⚡

**Goal:** Strict JSON parser: `parse(text)` → `AskAction | ClassifyAction | MalformedAction`; `validate(action, sections, history)` → `ValidationResult`. No silent coercion. No fallback parsing.

**AI Prompt (paste-ready):**
```
Implement examiner_env/action_parser.py for "The Examiner" RL project.

CANONICAL_SECTIONS = ["S01","S02","S03","S04","S05","S06","S07","S08","S09","S10"]

def parse(text: str) -> AskAction | ClassifyAction | MalformedAction:
    """
    1. Try json.loads(text). If fails → return MalformedAction(reason="not valid JSON")
    2. Check "action_type" key exists → else MalformedAction(reason="missing action_type")
    3. If action_type == "ask":
         Try to construct AskAction(**data) → Pydantic v2 will validate
         If ValidationError → return MalformedAction(reason=str(e))
    4. If action_type == "classify":
         Try to construct ClassifyAction(**data) → Pydantic v2 will validate
         If ValidationError → return MalformedAction(reason=str(e))
    5. Else: return MalformedAction(reason="unknown action_type")
    NEVER modify/coerce the input text. NEVER try to extract JSON from prose.
    """

def validate(action: AskAction | ClassifyAction,
             canonical_sections: list[str],
             history: list[dict]) -> ValidationResult:
    """
    For AskAction:
      - Check section_id in canonical_sections → else penalty "P_invalid_sec"
      - Check near-duplicate: if any prior ask in same section has question Jaccard similarity >0.85
        → penalty "P_repetition"
    For ClassifyAction:
      - Check all canonical_sections present in classifications → else penalty "P_cov"
      - Check all labels are "KNOWS" or "FAKING" → already handled by Pydantic, but double-check
    Return ValidationResult(valid=len(penalties)==0, penalties=penalties, info={})
    """

Import: from examiner_env.models import AskAction, ClassifyAction, MalformedAction, ValidationResult
```

**Context:** `models.py`
**Expected output:** `examiner_env/action_parser.py` with both functions
**Time:** 45 min
**Dependencies:** C1-0.2 (models.py)

**Sanity:**
1. `parse("")` returns `MalformedAction`
2. `parse('{"action_type":"ask","section_id":"S01","question_text":"Why does momentum help?"}')` returns `AskAction`
3. `parse('{"action_type":"classify","classifications":{"S01":"KNOWS"}}')` returns `MalformedAction` (missing 9 sections)

---

### Task C1-0.4: Parser Unit Tests (`tests/test_parser.py`) ⚡

**Goal:** Full test coverage — 10 test cases, all 8 malformed cases + 2 valid cases + Jaccard duplicate test.

**AI Prompt (paste-ready):**
```
Write tests/test_parser.py for examiner_env/action_parser.py.

Use pytest. Import parse() and validate() from examiner_env.action_parser.

Test cases (one test function each):
1. test_empty_string: parse("") → MalformedAction
2. test_valid_ask: parse(valid_ask_json) → AskAction with correct fields
3. test_valid_classify_all_10: parse(valid_classify_json with all 10 S01–S10) → ClassifyAction
4. test_missing_action_type: parse('{"section_id":"S01","question_text":"Why?"}') → MalformedAction
5. test_missing_section_id: parse('{"action_type":"ask","question_text":"Why?"}') → MalformedAction
6. test_wrong_label_case: parse(classify_json with "knows" lowercase) → MalformedAction
7. test_partial_classify: parse(classify_json with only 5 sections) → MalformedAction
8. test_non_json_prose: parse("I think this student knows gradient descent.") → MalformedAction
9. test_nested_json_in_prose: parse('Here is my answer: {"action_type":"ask",...}') → MalformedAction
10. test_duplicate_ask_validation: same question to same section twice → validate() returns ValidationResult with "P_repetition" in penalties

valid_ask_json = '{"action_type":"ask","section_id":"S01","question_text":"Why does momentum help gradient descent?"}'
valid_classify_json = json.dumps({"action_type":"classify","classifications":{f"S0{i}":"KNOWS" if i<6 else "FAKING" for i in range(1,11)}})
```

**Context:** `action_parser.py`, `models.py`
**Expected output:** `tests/test_parser.py`, all 10 tests passing
**Time:** 30 min
**Dependencies:** C1-0.3

**Sanity:**
1. `pytest tests/test_parser.py -v` exits 0
2. All 8 malformed cases produce `isinstance(result, MalformedAction) == True`
3. Duplicate ask test catches the `P_repetition` penalty in `ValidationResult.penalties`

---

### Task C2-0.1: Repo Structure + Colab Skeleton ⚡ (parallel with C1-0.1)

**Goal:** Initialize repo with exact folder structure from `project_structure.md`. Create `notebooks/train_examiner.ipynb` with section headers and install cells.

**AI Prompt (paste-ready):**
```
Create the following directory structure for "The Examiner" RL project:

examiner_env/__init__.py
examiner_env/action_parser.py (empty skeleton with module docstring)
examiner_env/environment.py (empty skeleton)
examiner_env/student.py (empty skeleton)
examiner_env/knowledge_base.py (empty skeleton)
examiner_env/posterior_oracle.py (empty skeleton)
examiner_env/reward.py (empty skeleton)
examiner_env/question_features.py (empty skeleton)
examiner_env/baselines.py (empty skeleton)
examiner_env/models.py (empty skeleton)
examiner_env/calibration.py (empty skeleton)
tests/__init__.py
tests/test_parser.py (empty)
tests/test_student_styles.py (empty)
tests/test_posterior_oracle.py (empty)
tests/test_reward.py (empty)
training/__init__.py
training/train_grpo.py (empty skeleton)
training/eval.py (empty skeleton)
training/prompt_builder.py (empty skeleton)
training/reward_fn.py (empty skeleton)
training/config.py (empty skeleton)
scripts/select_transcripts.py (empty skeleton)
scripts/generate_plots.py (empty skeleton)
hf_space/app.py (empty skeleton)
hf_space/README.md (placeholder)
notebooks/train_examiner.ipynb (with cells: [1] installs, [2] imports, [3] config, [4] oracle calibration, [5] baseline eval, [6] training, [7] eval, [8] plots)
outputs/eval/.gitkeep
outputs/plots/.gitkeep
outputs/transcripts/.gitkeep
requirements.txt (with: openenv, unsloth, trl, wandb, gradio>=4.0, pydantic>=2.0, numpy, torch)
.gitignore (includes: mistakes.md, *.env, __pycache__, *.pyc, outputs/eval/*.pkl)

For notebooks/train_examiner.ipynb: each cell must use !pip install (not local imports).
No local file path assumptions in any cell. All paths relative to repo root.
```

**Context:** `project_structure.md`
**Expected output:** Full directory tree with skeleton files
**Time:** 30 min
**Dependencies:** None

**Sanity:**
1. `import notebooks/train_examiner.ipynb` — first cell is `!pip install openenv unsloth trl wandb gradio pydantic`
2. All directories in `project_structure.md` exist
3. No absolute paths in any skeleton file

---

### Task C2-0.2: W&B Setup + Eval Config ⚡

**Goal:** W&B logging scaffold and `eval_config.json` with 50 fixed seeds, held-out styles, held-out sections.

**AI Prompt (paste-ready):**
```
Create two artifacts for "The Examiner" project:

1. eval_config.json:
{
  "n_episodes": 50,
  "seeds": [<50 fixed integer seeds, e.g. range(1000, 1050)>],
  "demo_config": {
    "training_styles": ["F1", "F2"],
    "held_out_styles": ["F3"],
    "training_sections": ["S01","S02","S03","S04"],
    "held_out_sections": ["S05"]
  },
  "full_config": {
    "training_styles": ["F1", "F2", "F3"],
    "held_out_styles": ["F4"],
    "training_sections": ["S01","S02","S03","S04","S05","S06","S07","S08"],
    "held_out_sections": ["S09", "S10"]
  },
  "debug_config": {
    "training_styles": ["F1"],
    "held_out_styles": [],
    "training_sections": ["S01","S02","S03"],
    "held_out_sections": []
  }
}

2. W&B scaffold in training/reward_fn.py (add import wandb, wandb.log() calls):
   def log_reward_breakdown(breakdown: RewardBreakdown, step: int):
       wandb.log({
           "reward/R_total": breakdown.R_total,
           "reward/R_acc": breakdown.R_acc,
           "reward/R_asym": breakdown.R_asym,
           "reward/R_cal": breakdown.R_cal,
           "reward/R_eff": breakdown.R_eff,
           "reward/R_cov": breakdown.R_cov,
           "reward/R_info": breakdown.R_info,
           "reward/R_qual": breakdown.R_qual,
           "reward/R_div": breakdown.R_div,
           "reward/P_malformed": breakdown.P_malformed,
           "reward/P_repetition": breakdown.P_repetition,
           "reward/P_invalid_sec": breakdown.P_invalid_sec,
       }, step=step)
```

**Expected output:** `eval_config.json` (committed to repo), W&B scaffold in `reward_fn.py`
**Time:** 30 min
**Dependencies:** None

**Sanity:**
1. `json.loads(open("eval_config.json").read())` succeeds, `len(data["seeds"]) == 50`
2. W&B test metric `wandb.log({"test": 1.0})` appears in W&B dashboard
3. Held-out styles in `demo_config` do not overlap with `training_styles`

---

### Task C2-0.3: Training Config (`training/config.py`) ⚡

**AI Prompt (paste-ready):**
```
Implement training/config.py for "The Examiner" GRPO training.

Use Python dataclasses. Define:

@dataclass
class TrainingConfig:
    config_name: str
    sections: list[str]
    max_turns: int
    num_episodes: int
    fake_styles_train: list[str]
    eval_styles_held_out: list[str]
    held_out_sections: list[str]
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

DEBUG_CONFIG = TrainingConfig(
    config_name="DEBUG",
    sections=["S01","S02","S03"], max_turns=3, num_episodes=20,
    fake_styles_train=["F1"], eval_styles_held_out=[], held_out_sections=[],
    eval_episodes=10, model_name="Qwen/Qwen2.5-1.5B-Instruct",
    lora_rank=8, lora_alpha=16, max_seq_length=1024,
    batch_size=1, gradient_accumulation=4, learning_rate=5e-6,
    num_generations=4, bf16=True, use_4bit=True,
    beta_kl=0.04, advantage_clip=5.0,
    reward_variance_floor=0.05, reward_variance_ceiling=1.5,
    max_grad_norm=1.0, warmup_ratio=0.05,
    checkpoint_every_n_steps=10, eval_every_n_steps=10
)

DEMO_CONFIG = TrainingConfig(
    config_name="DEMO",
    sections=["S01","S02","S03","S04","S05"], max_turns=4, num_episodes=200,
    fake_styles_train=["F1","F2"], eval_styles_held_out=["F3"],
    held_out_sections=["S05"], eval_episodes=50,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    lora_rank=16, lora_alpha=32, max_seq_length=2048,
    batch_size=2, gradient_accumulation=8, learning_rate=5e-6,
    num_generations=8, bf16=True, use_4bit=True,
    beta_kl=0.04, advantage_clip=5.0,
    reward_variance_floor=0.05, reward_variance_ceiling=1.5,
    max_grad_norm=1.0, warmup_ratio=0.05,
    checkpoint_every_n_steps=50, eval_every_n_steps=50
)

FULL_CONFIG = TrainingConfig(
    config_name="FULL",
    sections=[f"S0{i}" if i<10 else f"S{i}" for i in range(1,11)],
    max_turns=6, num_episodes=500,
    fake_styles_train=["F1","F2","F3"], eval_styles_held_out=["F4"],
    held_out_sections=["S09","S10"], eval_episodes=100,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    lora_rank=16, lora_alpha=32, max_seq_length=2048,
    batch_size=2, gradient_accumulation=8, learning_rate=5e-6,
    num_generations=8, bf16=True, use_4bit=True,
    beta_kl=0.04, advantage_clip=5.0,
    reward_variance_floor=0.05, reward_variance_ceiling=1.5,
    max_grad_norm=1.0, warmup_ratio=0.05,
    checkpoint_every_n_steps=50, eval_every_n_steps=50
)
```

**Time:** 20 min | **Sanity:** (1) DEBUG sections=3 (2) DEMO sections=5 (3) No hyperparameters hardcoded outside this file

---

### 🔀 PHASE 0 MERGE GATE

**Gate conditions (ALL must pass):**
1. `parse(valid_ask_json)` returns `AskAction` without error
2. `parse(valid_classify_json_all_10)` returns `ClassifyAction` without error
3. `parse("")` returns `MalformedAction`
4. `pytest tests/test_parser.py -v` exits 0 (all 10 tests pass)
5. Colab notebook opens, first cell runs `!pip install` without error
6. `eval_config.json` valid JSON with 50 seeds
7. W&B receives a test metric from a 3-line test script
8. `DEBUG_CONFIG.sections == ["S01","S02","S03"]` in Python REPL

**MSR partial:** MSR-1 (env skeleton), MSR-2 (Colab skeleton)
**Validator action:** Review `models.py` — confirm AskAction rejects invalid section_id; review `eval_config.json` — confirm held-out styles do not overlap training styles.

---

## PHASE 1 — ENVIRONMENT CORE

> **Goal:** Full working environment: student simulator, posterior oracle, reward function, all baselines, all tests passing.
> **Time estimate:** 5–7 hours (C1 primary, C2 assists on eval runner)
> **Gate condition:** Single episode end-to-end, all unit tests pass, oracle calibration ≤0.18 Brier, reward bounded.

---

### Task C1-1.1: Student Simulator (`student.py`) ⚡

**Goal:** All 7 styles with parametrized leak rates, probe-sensitive modulation, deterministic RNG.

**AI Prompt (paste-ready):**
```
Implement examiner_env/student.py for "The Examiner" RL project.

STYLE SPECIFICATIONS (implement EXACTLY these rates):
K1: mechanism_cue_emit_rate=0.85, misconception_emit_rate=0.05
K2: mechanism_cue_emit_rate=0.55, misconception_emit_rate=0.05
K3: mechanism_cue_emit_rate=0.65, misconception_emit_rate=0.08
F1: mechanism_cue_emit_rate=0.15, misconception_emit_rate=0.30, collapse_under_mechanism_probe=0.80
F2: mechanism_cue_emit_rate=0.20, misconception_emit_rate=0.25, jargon_mirror_rate=0.70
F3: mechanism_cue_emit_rate=0.10, misconception_emit_rate=0.20, topic_drift_rate=0.60
F4: mechanism_cue_emit_rate=0.05, misconception_emit_rate=0.40, specificity_score_cap=0.20

PROBE DETECTION (apply to question_text):
probe_keywords = {why, how, mechanism, because, explain, edge case, counterexample,
                  what if, failure, when does, under what, why does, derive}
If probe_keyword present in question_text.lower():
  For FAKING styles: mechanism_cue_emit_rate *= 0.5; misconception_emit_rate *= 1.5
  For KNOWS styles: rates unchanged

RESPONSE ASSEMBLY:
def generate_response(question_text: str, section_id: str, profile: StudentProfile,
                      kb: KnowledgeBase, episode_seed: int, turn: int) -> str:
  rng = random.Random((episode_seed, turn, section_id))
  probe_detected = any(kw in question_text.lower() for kw in probe_keywords)
  effective_mech_rate = profile.mechanism_cue_emit_rate * (0.5 if probe_detected and profile.knowledge_mode=="FAKING" else 1.0)
  effective_misc_rate = profile.misconception_emit_rate * (1.5 if probe_detected and profile.knowledge_mode=="FAKING" else 1.0)
  
  If profile.knowledge_mode == "KNOWS":
    response_parts = []
    for cue in kb[section_id].mechanism_cues:
      if rng.random() < effective_mech_rate * (1.0 if cue.cue_strength=="strong" else 0.5):
        response_parts.append(cue.phrase)
    # Assemble natural-sounding response from parts
    # Apply verbosity: brief=1-2 sentences, medium=2-4, verbose=4+
  
  If profile.knowledge_mode == "FAKING":
    response_parts = []
    for misc in kb[section_id].common_misconceptions:
      if rng.random() < effective_misc_rate * (1.0 if misc.severity=="major" else 0.5):
        response_parts.append(misc.phrase)
    # Apply style_specific_params (collapse/mirror/drift/cap)
    # If F1 and probe_detected: collapse to "I'm not sure about the specific mechanism..."
    # If F2: mirror technical terms from question_text
    # If F3: pivot to adjacent section topic
    # If F4: very high-level, zero specificity

def sample_profile(section_id: str, knowledge_mode: Literal["KNOWS","FAKING"],
                   episode_seed: int, section_idx: int) -> StudentProfile:
  rng = random.Random((episode_seed, section_idx, knowledge_mode))
  if knowledge_mode == "KNOWS":
    style = rng.choice(["K1","K2","K3"])
  else:
    style = rng.choice(["F1","F2","F3","F4"])
  # Set rates from STYLE_SPECS table above
  # Sample verbosity from style-appropriate distribution
  return StudentProfile(...)

Import: from examiner_env.models import StudentProfile
Import: from examiner_env.knowledge_base import KnowledgeBase
```

**Time:** 120 min | **Dependencies:** C1-0.1, C1-0.2
**Judging:** ENV_INNOV (40%) — simulator quality is the primary novelty claim

**Sanity:**
1. F1 style response to "why" probe contains ≤1 strong mechanism_cue in ≥80% of 100 seeded samples
2. K1 style response contains ≥2 mechanism_cues on average over 100 seeded samples
3. `generate_response(q, "S01", profile, kb, seed=42, turn=1) == generate_response(q, "S01", profile, kb, seed=42, turn=1)` (byte-identical)

---

### Task C1-1.2: Student Style Tests (`tests/test_student_styles.py`)

**AI Prompt (paste-ready):**
```
Write tests/test_student_styles.py for examiner_env/student.py.

Test all 7 styles. Use pytest.

test_k1_mechanism_density: Over 50 seeded samples of K1 on S01, mechanism_cue keywords appear
  in >60% of responses (count keyword occurrences from KB[S01].mechanism_cues)
test_f1_collapses_under_probe: Over 50 samples, F1 response to "Why does gradient descent
  converge? Explain the mechanism." contains ≤1 mechanism_cue (vs K1 which contains ≥2)
test_f2_mirrors_jargon: F2 response mirrors ≥1 technical term from the question
test_f4_low_specificity: F4 response is high-level with zero KB mechanism cues (0 matches)
test_reproducibility_all_styles: For each of 7 styles, same (seed,turn,section,question)
  produces byte-identical response across 3 calls
test_styles_distinguishable: K1 vs F1 on same question → keyword overlap < 0.3 (Jaccard)
test_all_styles_non_empty: Every style produces non-empty string for every section
```

**Time:** 30 min | **Dependencies:** C1-1.1

**Sanity:**
1. `pytest tests/test_student_styles.py -v` exits 0
2. K1 and F1 responses to same question have Jaccard overlap < 0.3
3. All 7 styles produce non-empty, non-identical responses

---

### Task C1-1.3: Posterior Oracle (`posterior_oracle.py`)

**Goal:** LLR scorer, sigmoid posterior update, ΔH_t computation. **Must paste §6.1 equations verbatim into prompt.**

**AI Prompt (paste-ready):**
```
Implement examiner_env/posterior_oracle.py for "The Examiner" RL project.

EQUATIONS (implement EXACTLY — no improvisation):
evidence(r, s) = α * mechanism_cue_coverage(r, KB[s]) - γ * misconception_count(r, KB[s])
relevance(q, s) = β * probe_strength(q, KB[s].probe_templates)
LLR_t(s) = relevance(q, s) * evidence(r, s)   # clip to [-3.0, +3.0]
posterior_t(s) = sigmoid(logit(p_{t-1}(s)) + LLR_t(s))

DEFAULT WEIGHTS: α=1.5, β=0.5, γ=1.0 (loaded from KB[s].evidence_weights)

Helper functions:
def mechanism_cue_coverage(response: str, section_kb) -> float:
    # Count mechanism_cues from KB that appear in response
    # Weight: cue_strength=="strong" → 1.0, "weak" → 0.5
    # Normalize to [0, 1] by dividing by sum of max possible cue weight

def misconception_count(response: str, section_kb) -> float:
    # Count misconceptions from KB that appear in response
    # Weight: severity=="major" → 1.0, "minor" → 0.5
    # Normalize to [0, 1]

def probe_strength(question: str, probe_templates: list) -> float:
    # Binary: 1.0 if question contains mechanism/edge_case probe keywords
    # Keywords: {why, how, mechanism, derive, explain the, what happens when,
    #             edge case, counterexample, failure mode}
    # 0.5 if question matches a definitional template
    # 0.3 otherwise

def sigmoid(x: float) -> float: return 1.0 / (1.0 + exp(-x))
def logit(p: float) -> float: return log(p / (1 - p))  # clip p to [0.001, 0.999]

class PosteriorTracker:
    def __init__(self, section_ids: list[str], kb):
        self._posteriors = {s: 0.5 for s in section_ids}  # p_0 = 0.5
        self._trace = []  # list[dict[section_id, float]] per turn
        self._entropy_trace = []  # ΔH_t per turn
        self._kb = kb

    def update(self, section_id: str, question: str, response: str) -> float:
        # Compute LLR, update posterior, record trace, return ΔH_t

    def current_posteriors(self) -> dict[str, float]:  # p_t(s) for all s

    def entropy(self) -> float:
        # H = -(1/N) * sum_s [ p_s*log2(p_s) + (1-p_s)*log2(1-p_s) ]
        # Use log base 2. Clip p to [0.001, 0.999] before log.

    def get_trace(self) -> list[dict[str, float]]:  # full posterior trace per turn
    def get_entropy_gains(self) -> list[float]:  # ΔH_t per turn

CRITICAL: same (seed, dialogue_sequence) must produce byte-identical trace across runs.
```

**Time:** 75 min | **Dependencies:** C1-0.1, C1-0.2

**Sanity:**
1. `PosteriorTracker` initialized with `p_0(s) = 0.5` for all sections
2. After K1-style response (mechanism-rich) update, posterior for asked section moves above 0.5
3. After F1-style response (misconception-heavy) update, posterior moves below 0.5

---

### Task C1-1.4: Oracle Calibration (`calibration.py`) — BLOCKS TRAINING

**AI Prompt (paste-ready):**
```
Implement examiner_env/calibration.py for "The Examiner" project.

Goal: calibrate (α, β, γ) on a 200-episode held-out split before training.

def run_calibration(kb: KnowledgeBase, n_episodes: int = 200,
                    output_path: str = "outputs/eval/oracle_calibration.json") -> dict:
    """
    1. Generate 200 synthetic episodes:
       - Random KNOWS/FAKING partition per episode
       - Sample K1/K2/K3 for KNOWS, F1/F2/F3/F4 for FAKING (equal probability)
       - For each section, generate 2-3 questions from probe_templates
       - Generate responses via student.generate_response()
       - Run PosteriorTracker with current (α, β, γ)
    2. Compute terminal Brier score: mean((p_T(s) - true_label(s))^2) over all sections, episodes
    3. Compute terminal accuracy: mean(sign(p_T(s)-0.5) == true_label_binary(s))
    4. If Brier > 0.18 or accuracy < 0.75: try simple grid search over
       α ∈ [1.0, 1.5, 2.0], β ∈ [0.3, 0.5, 0.7], γ ∈ [0.8, 1.0, 1.2]
    5. Save best (α, β, γ) per section to output JSON

Output JSON structure:
{
  "global": {"alpha": 1.5, "beta": 0.5, "gamma": 1.0},
  "per_section": {
    "S01": {"alpha": 1.5, "beta": 0.5, "gamma": 1.0},
    ...
  },
  "calibration_metrics": {
    "mean_brier": 0.14,
    "terminal_accuracy": 0.78,
    "n_episodes": 200
  }
}

ASSERT: mean_brier <= 0.18 and terminal_accuracy >= 0.75 — raise CalibrationError if not met.
"""
```

**Time:** 60 min | **Dependencies:** C1-1.1, C1-1.3

**Sanity:**
1. `outputs/eval/oracle_calibration.json` exists after run
2. `calibration_metrics.mean_brier ≤ 0.18`
3. `calibration_metrics.terminal_accuracy ≥ 0.75`

---

### Task C1-1.5: Question Features (`question_features.py`) ⚡

**AI Prompt (paste-ready):**
```
Implement examiner_env/question_features.py.

def question_features(question_text: str, kb, section_id: str) -> float:
    """
    Question-side R_qual scorer. DO NOT take student_response as argument.
    Returns float in [0, 1].

    score = (
      0.40 * mechanism_probe_present(question_text, kb, section_id)
    + 0.30 * specificity_demand(question_text)
    + 0.30 * edge_case_or_counterexample(question_text)
    )

    mechanism_probe_present: 1.0 if question_text contains ANY of:
      - "why", "how", "mechanism", "explain the", "derive", "because"
      - OR any mechanism_cue keyword from KB[section_id].mechanism_cues
    Else 0.0.

    specificity_demand: 1.0 if question_text contains ANY of:
      - "example", "specific", "quantitative", "precise", "exactly",
        "give an instance", "for instance", "concrete"
    Else 0.0.

    edge_case_or_counterexample: 1.0 if question_text contains ANY of:
      - "edge case", "counterexample", "failure mode", "when does X fail",
        "boundary", "limitation", "what if", "what happens when", "breakdown"
    Else 0.0.

    CRITICAL: This function must be entirely question-side.
    It must not accept response, kb_response, oracle, or any answer-side data.
    """
```

**Time:** 45 min | **Dependencies:** C1-0.1

**Sanity:**
1. `question_features("What is gradient descent?", kb, "S01")` returns ~0.0 (purely definitional)
2. `question_features("Why does momentum help convergence and what happens when β→1?", kb, "S01")` returns ≥0.7
3. Function signature has no `response` or `student_response` parameter

---

### Task C1-1.6: Reward Function (`reward.py`) — CRITICAL, paste §6.2 verbatim

**AI Prompt (paste-ready):**
```
Implement examiner_env/reward.py for "The Examiner" RL project.

EXACT SPECIFICATIONS — do not improvise ANY weight, bound, or formula:

R_acc   = (1/N) * sum_s [+1.0 if classified[s]==true_label[s] else -1.0]     in [-1, +1]
R_asym  = -(lambda_FA * #false_accusations + lambda_FE * #false_exonerations) / N
          lambda_FA = 0.5  (KNOWS classified as FAKING)
          lambda_FE = 0.3  (FAKING classified as KNOWS)                        in [-0.5, 0]
R_cal   = (0.4/N) * sum_s sign(correct(s) - 0.5) * abs(2*p_T(s) - 1)         in [-0.4, +0.4]
          p_T(s) = terminal posterior from PosteriorTracker
R_eff   = 0.20 * max(0, (MAX_TURNS - turns_used) / MAX_TURNS) * (1 if R_acc > 0 else 0)  in [0, +0.20]
R_cov   = -0.30 * (1 if any_section_missing else 0) - 0.05 * (n_missing / 10)  in [-0.35, 0]
R_info  = 0.40 * clip(sum_t ΔH_t, 0, 1)                                        in [0, +0.40]
          ΔH_t from PosteriorTracker.get_entropy_gains(), summed then clipped to [0,1]
R_qual  = 0.10 * mean_over_asks(question_features(q, kb, section_id))          in [0, +0.10]
R_div   = 0.05 * (n_unique_sections_asked / min(turns_used, 10))               in [0, +0.05]
P_malformed   = -0.20 * n_malformed                                             ≤ 0
P_repetition  = -0.10 * n_near_duplicate_asks                                  ≤ 0
P_invalid_sec = -0.10 * n_invalid_section_asks                                 ≤ 0
R_total = sum of all above

@dataclass(frozen=True)
class RewardBreakdown:
    R_acc: float; R_asym: float; R_cal: float; R_eff: float; R_cov: float
    R_info: float; R_qual: float; R_div: float
    P_malformed: float; P_repetition: float; P_invalid_sec: float
    R_total: float
    posterior_trace: list[dict[str, float]]
    info_gain_per_turn: list[float]

def compute_reward(episode_result: EpisodeResult, kb: KnowledgeBase) -> RewardBreakdown:
    # Compute all components
    # ASSERT: abs(sum_of_components - R_total) < 1e-9
    # ASSERT: -2.05 <= R_total <= 1.95 — raise ValueError if violated
    # NEVER normalize R_total
    # NEVER call any LLM

EpisodeResult must contain: classifications, true_labels, turns_used, max_turns,
  dialogue_history, posterior_tracker, n_malformed, n_repetition, n_invalid_sec
```

**Time:** 90 min | **Dependencies:** C1-1.3, C1-1.5, C1-0.1, C1-0.2

**Sanity:**
1. All-correct episode with 4 questions: R_total ∈ [+0.8, +1.95], R_acc=+1.0
2. Malformed action: R_total decreases by exactly 0.20 (P_malformed = −0.20)
3. `sum([rb.R_acc, rb.R_asym, rb.R_cal, rb.R_eff, rb.R_cov, rb.R_info, rb.R_qual, rb.R_div, rb.P_malformed, rb.P_repetition, rb.P_invalid_sec]) ≈ rb.R_total ± 1e-9`

---

### Task C1-1.7: Reward Unit Tests (`tests/test_reward.py`)

**AI Prompt (paste-ready):**
```
Write tests/test_reward.py for examiner_env/reward.py.

test_all_correct_reward: All-correct classify, 4 mechanism probe questions → R_total > +0.8
test_all_wrong_confident: All-wrong classify, high posterior in wrong direction → R_total < -1.0
test_malformed_penalty: Episode with 1 malformed action → P_malformed == -0.20
test_decomposition_sums: R_acc+R_asym+R_cal+R_eff+R_cov+R_info+R_qual+R_div+P_mal+P_rep+P_inv == R_total ± 1e-9
test_bounds_random_episode: 100 random episodes → all R_total in [-2.05, +1.95]
test_no_normalization: R_total returned raw (not z-scored)
test_r_eff_gated: R_eff = 0 when R_acc <= 0 (even with unused turns)
test_r_qual_question_side: R_qual computed from question_text only — changing response doesn't change R_qual
test_r_info_nonnegative: R_info ∈ [0, +0.40] always (clipped to 0 on negative entropy change)
test_reproducibility: same EpisodeResult → byte-identical RewardBreakdown
```

**Time:** 45 min | **Dependencies:** C1-1.6

**Sanity:**
1. `pytest tests/test_reward.py -v` exits 0
2. All 10 tests pass
3. Decomposition test tolerance is 1e-9 (not 1e-6 — float precision matters)

---

### Task C1-1.8: Baselines (`baselines.py`)

**AI Prompt (paste-ready):**
```
Implement examiner_env/baselines.py for "The Examiner" project.

class RandomExaminer:
    def act(self, observation: dict) -> dict:
        # Sample random section_id from observation["section_ids"]
        # Sample random question from KB[section_id].probe_templates
        # After MAX_TURNS/2 asks: return classify with random KNOWS/FAKING per section

class DefinitionalExaminer:
    def act(self, observation: dict) -> dict:
        # Ask "What is [section_title]?" for each section in order
        # After asking all sections once: classify by response_length heuristic
        # (longer response → KNOWS, shorter → FAKING)
        # Primary comparison baseline for storytelling — it is weak by design

class BayesianHeuristicExaminer:
    def __init__(self, kb: KnowledgeBase):
        self._tracker = PosteriorTracker(all_sections, kb)
        self._probe_cycle = ["definitional", "mechanism", "edge_case", "counterexample"]
        self._asked_type = {s: 0 for s in all_sections}  # index into cycle

    def act(self, observation: dict) -> dict:
        # Find section with highest uncertainty: argmin |p_t(s) - 0.5|
        # among sections where |p_t(s) - 0.5| <= 0.4 (not yet confident)
        # Ask next probe_type in cycle for that section
        # If all sections |p_t(s) - 0.5| > 0.4 OR turns_remaining == 1:
        #   Classify: for each section, argmax(p_t(s) > 0.5) → KNOWS else FAKING
        # Update own PosteriorTracker after each response

All 3 baselines use same environment interface as TrainedExaminer.
All 3 complete episodes (no infinite loops, no crashes).
```

**Time:** 75 min | **Dependencies:** C1-1.3, C1-0.1

**Sanity:**
1. `RandomExaminer` produces valid JSON actions (Ask/Classify) for any observation
2. `DefinitionalExaminer` always generates "What is" style questions (contains "What is" or "Define")
3. `BayesianHeuristicExaminer` stop-condition fires (classified section within MAX_TURNS) for ≥50% of KNOWS sections on a 20-episode smoke run

---

### Task C1-1.9: Environment Class (`environment.py`) — CRITICAL

**Goal:** OpenEnv-inheriting `ExaminerEnv` with `reset()`, `step()`, `action_space`, `observation_space`. **Must paste OpenEnv base class interface into prompt.**

**AI Prompt (paste-ready):**
```
Implement examiner_env/environment.py for "The Examiner" RL project.

OPENENV BASE CLASS INTERFACE (inherit from this — do NOT reimplement base methods):
[PASTE THE ACTUAL OPENENV BASE CLASS INTERFACE HERE BEFORE RUNNING THIS PROMPT]

class ExaminerEnv(OpenEnv):  # inherit from OpenEnv — do NOT reinvent it
    def reset(self, seed: int = None, options: dict = None) -> tuple[dict, dict]:
        # Sample KNOWS/FAKING partition for each section
        # Sample style per section using sample_profile()
        # Initialize PosteriorTracker
        # Initialize dialogue_history = []
        # NEVER put true_labels, style_ids, or posteriors in the observation
        # Return (observation, info)
        # observation = {
        #   "section_titles": dict[section_id, title],
        #   "turn": 0,
        #   "remaining_turns": self.max_turns,
        #   "dialogue_history": []
        # }

    def step(self, action_text: str) -> tuple[dict, float, bool, bool, dict]:
        # 1. parse(action_text) → AskAction | ClassifyAction | MalformedAction
        # 2. If Malformed: accumulate penalty, continue episode (or end per config)
        # 3. If Ask: validate → get student response → update oracle → append to history
        # 4. If Classify: compute_reward → return done=True
        # 5. If turns exhausted: force classify (all FAKING) → compute_reward → done=True
        # Return (observation, 0.0, terminated, truncated, info) for Ask steps
        # Return (observation, R_total, True, False, reward_breakdown_dict) for Classify/exhaustion

    def _build_observation(self) -> dict:
        # Returns only: section_titles, turn, remaining_turns, dialogue_history
        # NEVER includes: true_labels, style_assignments, posterior values

Register:
import openenv
openenv.register(id="ExaminerEnv-v0", entry_point="examiner_env.environment:ExaminerEnv")
```

**⚠️ CRITICAL: Replace `[PASTE THE ACTUAL OPENENV BASE CLASS INTERFACE HERE]` with the real OpenEnv base class code before running this prompt.**

**Time:** 90 min | **Dependencies:** All C1-1.x tasks

**Sanity:**
1. `env.reset(seed=42)` returns observation with `section_titles` dict (no labels) and empty history
2. `env.step(valid_ask_json)` returns `(obs, 0.0, False, False, info)` with history length +1
3. `env.step(valid_classify_json)` returns `(obs, R_total, True, False, breakdown_dict)`

---

### Task C2-1.1: Frozen Eval Suite Runner (`training/eval.py`) ⚡

**AI Prompt (paste-ready):**
```
Implement training/eval.py for "The Examiner" RL project.

def run_eval(examiner, eval_config: dict, kb: KnowledgeBase,
             output_path: str = None) -> dict:
    """
    Run examiner on frozen eval suite. Return metrics dict.

    Metrics to compute (ALL required):
    - classification_accuracy: float (overall)
    - per_section_accuracy: dict[str, float]
    - false_accusation_rate: float  (KNOWS→FAKING rate)
    - false_exoneration_rate: float  (FAKING→KNOWS rate)
    - avg_turns_to_classify: float
    - avg_info_gain_per_turn: float  (mean ΔH_t from RewardBreakdown.info_gain_per_turn)
    - terminal_posterior_correctness: float  (% sections where sign(p_T-0.5)==truth)
    - calibration_ECE: float  (Expected Calibration Error, 10-bin)
    - calibration_brier: float
    - mean_R_qual: float
    - mean_R_info: float
    - mean_R_cal: float
    - parse_failure_rate: float  (#malformed / total_steps)
    - reward_mean: float
    - reward_std: float
    - per_style_accuracy: dict[str, float]  # e.g. {"K1": 0.8, "F1": 0.7, ...}

    Use eval_config["seeds"] for reproducibility.
    Use RewardBreakdown.posterior_trace for ECE/Brier/info_gain computations.
    ECE: 10 equal-width bins, |predicted_confidence - empirical_accuracy| per bin.
    """

def compute_ece(posteriors: list[float], labels: list[int], n_bins: int = 10) -> float:
    # posteriors: list of p_T(s) values
    # labels: 1 if KNOWS, 0 if FAKING
    # Returns ECE in [0, 1]
```

**Time:** 75 min | **Dependencies:** C1-1.8 (baselines), C1-1.9 (env)

**Sanity:**
1. `run_eval(RandomExaminer(), eval_config, kb)` completes without crash
2. All returned metrics are finite, non-NaN
3. `calibration_ECE ∈ [0, 1]`, `per_style_accuracy` has keys for all styles in eval suite

---

### 🔀 PHASE 1 MERGE GATE

**Gate conditions (ALL must pass — this gate blocks ALL training):**
1. `pytest tests/test_parser.py tests/test_student_styles.py tests/test_posterior_oracle.py tests/test_reward.py -v` → all tests pass
2. Single episode end-to-end: `env.reset(42)` → 3 ask steps → classify → R_total finite and in [−2.05, +1.95]
3. All 3 baselines complete 10 episodes without crash
4. `outputs/eval/oracle_calibration.json` exists, `mean_brier ≤ 0.18`, `terminal_accuracy ≥ 0.75`
5. `R_acc + R_asym + R_cal + R_eff + R_cov + R_info + R_qual + R_div + P_malformed + P_repetition + P_invalid_sec == R_total ± 1e-9` in test
6. Reward variance σ(R_total) ≥ 0.05 over 20 random episodes (reward is not constant)
7. `env._build_observation()` output contains NO true_labels, style_ids, or posteriors
8. MSR-1: `ExaminerEnv` inherits from OpenEnv (not re-implemented)
9. `run_eval(DefinitionalExaminer(), eval_config, kb)["classification_accuracy"] > run_eval(RandomExaminer(), eval_config, kb)["classification_accuracy"]` (sanity check)
10. `run_eval(BayesianHeuristicExaminer(kb), eval_config, kb)["avg_info_gain_per_turn"] > run_eval(DefinitionalExaminer(), eval_config, kb)["avg_info_gain_per_turn"]`

**MSR satisfied:** MSR-1 fully
**Validator action:** Run all 10 gate conditions manually. Log pass/fail to `implementation_validator.md`. Reject merge if any fail.

---

## PHASE 2 — TRAINING PIPELINE

> **Goal:** GRPO training pipeline wired. DEBUG smoke test passes pipeline health check.
> **Time estimate:** 3–4 hours
> **Gate condition:** DEBUG config runs 20 episodes, pipeline health confirmed, W&B receives per-component rewards.

---

### Task C2-2.1: Prompt Builder (`training/prompt_builder.py`)

**AI Prompt (paste-ready):**
```
Implement training/prompt_builder.py.

def build_prompt(observation: dict, action_schema_examples: dict = None) -> str:
    """
    observation contains: section_titles (dict), turn (int),
    remaining_turns (int), dialogue_history (list)

    Prompt template:
    ---
    You are an examiner. Determine which ML sections a student KNOWS vs FAKING.
    Ask targeted questions; then issue a final CLASSIFY action.

    Sections:
    {for s_id, title in section_titles.items(): "- {s_id}: {title}"}

    Turn: {turn} of {max_turns}. Remaining turns: {remaining_turns}.

    Dialogue History:
    {for turn, ask in enumerate(dialogue_history): "Turn {t+1} → Asked {s_id}: '{question}'\nStudent: '{response}'"}

    Output ONLY valid JSON matching one of:
    ASK:      {"action_type": "ask", "section_id": "<S01-S10>", "question_text": "<min 10 chars>"}
    CLASSIFY: {"action_type": "classify", "classifications": {"S01": "KNOWS"|"FAKING", ..., "S10": "KNOWS"|"FAKING"}}

    Your JSON:
    ---

    CRITICAL: Do NOT include any of the following in the prompt:
    - True labels (KNOWS/FAKING for any section)
    - Style IDs (K1, K2, K3, F1, F2, F3, F4)
    - Posterior probabilities or confidence scores
    - Any hidden state information
    """
```

**Time:** 30 min

**Sanity:**
1. Prompt contains all section IDs from `observation["section_titles"]`
2. Prompt includes JSON examples for BOTH Ask and Classify
3. String "KNOWS" does not appear as a hidden label (only as a schema example value in the Classify schema)

---

### Task C2-2.2: Reward Function Bridge (`training/reward_fn.py`)

**AI Prompt (paste-ready):**
```
Implement training/reward_fn.py — the bridge between TRL GRPOTrainer and examiner_env.

def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    TRL GRPOTrainer reward function interface.

    For each completion:
    1. action_parser.parse(completion) → action
    2. If MalformedAction: return P_malformed = -0.20 for this completion
    3. Else: run env.step(completion) to get reward
    4. Return R_total from RewardBreakdown

    CRITICAL: DO NOT re-implement any reward logic here.
    ALWAYS delegate to examiner_env.reward.compute_reward().
    AI will try to inline reward computation here — PREVENT THIS.

    W&B logging (call after computing rewards):
    wandb.log({
        "reward/R_total_batch_mean": mean(rewards),
        "reward/R_total_batch_std": std(rewards),
        "training/parse_failure_rate": n_malformed / len(completions),
    }, step=global_step)
    """
```

**Time:** 45 min | **Dependencies:** C1-1.9 (env), C2-2.1

**Sanity:**
1. Returns `list[float]` with same length as `completions`
2. Malformed completion returns exactly −0.20 (P_malformed for 1 malformed action)
3. `reward_fn` imports from `examiner_env.reward` — does NOT define its own reward formulas

---

### Task C2-2.3: Training Script (`training/train_grpo.py`)

**⚠️ AI will hallucinate TRL API. Paste current GRPOTrainer docs before running this prompt.**

**AI Prompt (paste-ready):**
```
Implement training/train_grpo.py using Unsloth + TRL GRPOTrainer.

[PASTE CURRENT TRL GRPOTrainer API DOCUMENTATION HERE]

from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from training.config import TrainingConfig
from training.reward_fn import reward_fn
from training.prompt_builder import build_prompt
from training.eval import run_eval
import wandb

def train(config: TrainingConfig, eval_config: dict):
    wandb.init(project="bluffbuster-examiner", config=vars(config))

    # Load model with Unsloth 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.use_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=config.lora_rank, lora_alpha=config.lora_alpha, ...
    )

    # GRPOConfig — match to our config exactly
    grpo_config = GRPOConfig(
        num_generations=config.num_generations,
        beta=config.beta_kl,
        ...
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        ...
    )

    # Checkpoint + eval hooks every N steps
    # Eval: run_eval(trained_model_examiner, eval_config, kb) → log to W&B
    # Checkpoint: save model to outputs/checkpoints/

    trainer.train()
    wandb.finish()

Log to W&B per step: episode_reward, per-component rewards, advantage_mean, advantage_std.
NEVER hardcode any hyperparameter — all come from config.
```

**Time:** 90 min | **Dependencies:** C2-2.2, C1-1.9

**Sanity:**
1. `from training.train_grpo import train` imports without error
2. W&B run initializes and receives `reward/R_total_batch_mean` metric
3. `train(DEBUG_CONFIG, eval_config)` runs 5 steps without crash

---

### Task C2-2.4: DEBUG Smoke Test

**Checklist — all must be true before Gate 2:**
- [ ] 20 episodes complete without crash or exception
- [ ] Every R_total finite and within [−2.05, +1.95] (check W&B logs)
- [ ] `σ(R_total) ∈ [0.05, 1.5]` over 20 episodes
- [ ] All 11 per-component rewards visible as separate W&B metrics
- [ ] `parse_failure_rate < 0.5` in W&B logs
- [ ] Posterior trace logged for ≥1 episode (check `info` dict in env)
- [ ] `outputs/eval/oracle_calibration.json` exists (should be from Phase 1)
- [ ] Eval suite produces checkpoint at step 0 with 4 baselines
- [ ] GRPO advantage normalization: A_i mean ≈ 0, std ≈ 1 in W&B
- [ ] **NOT required:** reward improvement over 20 episodes. Pipeline health is the only gate.

**Time:** 75 min (run + verify)

---

### 🔀 PHASE 2 MERGE GATE

1. DEBUG smoke test checklist fully green (all 9 items above)
2. W&B dashboard shows 11 separate reward metric keys (not just R_total)
3. `reward_fn.py` imports `compute_reward` from `examiner_env.reward` — grep confirms no inline reward logic
4. Prompt output does not contain true_labels or style IDs — Validator runs `build_prompt()` and checks output manually
5. MSR-2 partial: Colab notebook imports without error

---

## PHASE 3 — TRAINING RUN & EVIDENCE

> **Goal:** Real DEMO training run, real W&B artifacts, plots from actual data, behavior-selected transcripts.
> **Time estimate:** 3–5 hours (includes Colab A100 run time: ~2–3 hrs)
> **Gate condition:** `final_metrics.json` exists. Real plots generated from W&B data. Comparison table has all 4 examiners.

---

### Task C2-3.1: DEMO Training Run

- Run DEMO config on Colab A100 (200 episodes, 5 sections, F1+F2 training, Qwen2.5-7B-Instruct 4-bit)
- Pre-run: baseline eval → `baseline_metrics.json`
- During run: checkpoint every 50 steps → `checkpoint_metrics.json`
- Post-run: final eval on held-out F3 style + S05 section → `final_metrics.json`
- W&B run ID must be logged to `outputs/plots/README.md`
- **Time:** 2–3 hours Colab A100 runtime

**Sanity:**
1. W&B run exists and is accessible
2. `outputs/eval/final_metrics.json` exists with all required metrics
3. Trained examiner `classification_accuracy` ≥ `DefinitionalExaminer.classification_accuracy` on held-out eval (or document honestly if not)

---

### Task C1-3.1 / C2-3.2: Transcript Selection (`scripts/select_transcripts.py`)

**AI Prompt (paste-ready):**
```
Implement scripts/select_transcripts.py.

Load eval results:
  baseline = json.load("outputs/eval/baseline_metrics.json")
  final = json.load("outputs/eval/final_metrics.json")

Find correctness-flip episodes:
  flip_eps = [ep_seed for ep_seed in eval_config["seeds"]
              if baseline["DefinitionalExaminer"]["per_episode"][ep_seed]["correct"] == False
              and final["TrainedExaminer"]["per_episode"][ep_seed]["correct"] == True]

Select episode with largest R_info gap:
  best = max(flip_eps, key=lambda ep:
    final["TrainedExaminer"]["per_episode"][ep]["R_info"]
    - baseline["DefinitionalExaminer"]["per_episode"][ep]["R_info"])

Export:
  before_transcript.json = {
    "episode_seed": best,
    "examiner": "DefinitionalExaminer",
    "dialogue": [...],
    "classification": {...},
    "true_labels": {...},  # revealed at end for display
    "reward_breakdown": {...},
    "posterior_trace": [...],  # per-turn p_t(s)
    "R_info": ..., "R_total": ..., "correct": False
  }
  after_transcript.json = {same structure for TrainedExaminer on same seed}
```

**Time:** 30 min

**Sanity:**
1. `before_transcript.json` has low `R_info` and `correct: false`
2. `after_transcript.json` has higher `R_info` and `correct: true`
3. Both use same `episode_seed`

---

### Task C2-3.3: Plot Generation (`scripts/generate_plots.py`)

**Required plots (ALL from real data — MSR-3 blocker):**
1. R_total curve: mean ± std band per eval checkpoint step
2. Per-component small-multiples: R_acc, R_info, R_cal, R_qual, R_asym
3. Classification accuracy curve over training steps
4. False accusation + false exoneration curves (overlaid)
5. avg_info_gain_per_turn curve
6. Calibration ECE curve
7. Comparison bar chart: 4 examiners × held-out eval metrics
8. Per-style accuracy heatmap (7 styles × 10 sections)
9. Posterior trace for best AFTER transcript (per-section p_t over turns)

**⚠️ ALL plots must be generated from `checkpoint_metrics.json` / `final_metrics.json` / W&B logs. No synthetic data. No hardcoded values. MSR-3 is an immediate blocker.**

**Time:** 60 min

**Sanity:**
1. All 9 plots render without error
2. Comparison bar chart shows all 4 examiners
3. Plot data values match `final_metrics.json` within rounding

---

### 🔀 PHASE 3 MERGE GATE

1. `outputs/eval/final_metrics.json` exists with all required metrics
2. All 9 plots in `outputs/plots/` render without error
3. `before_transcript.json` and `after_transcript.json` both exist with `posterior_trace` populated
4. Comparison table [Random|Definitional|BayesianHeuristic|Trained] complete
5. W&B run ID logged — Validator verifies plots match W&B data
6. MSR-3 satisfied: plots are real, not synthetic

---

## PHASE 4 — DEPLOYMENT

> **Goal:** HF Space live, Colab runs top-to-bottom, README complete.
> **Time estimate:** 2–3 hours

### Task C2-4.1: HF Space (`hf_space/app.py`)

4-tab Gradio app — see `architecture.md` §11 for full spec.

**⚠️ CRITICAL: Test in incognito before marking done. Check Space logs for silent failures.**

**Sanity:**
1. Space URL opens in incognito without errors
2. Tab 1 runs a full episode (click "Run Episode" → shows dialogue, posterior trace, reward breakdown)
3. Tab 3 shows real data plots (not placeholder text)

**Time:** 120 min

---

### Task C2-4.2: Colab Notebook (`notebooks/train_examiner.ipynb`)

End-to-end: install → DEBUG smoke test → DEMO training → eval → plots.

**⚠️ CRITICAL: Every cell must work on clean Colab runtime. No local file paths. AI will add `open("/local/path/file.json")` — prevent this explicitly in prompt.**

**Time:** 45 min

---

### 🔀 PHASE 4 MERGE GATE

1. HF Space URL accessible in incognito — all 4 tabs functional
2. Colab runs top-to-bottom on clean runtime (Validator runs it)
3. README has all 6 sections; comparison table populated
4. README link to HF Space works
5. `git ls-files | grep -E ".mp4|.mov|.avi|.mkv"` returns empty (no video files)
6. MSR-5, MSR-6, MSR-7 satisfied

---

## PHASE 5 — STORYTELLING

> **Goal:** Writeup published, README polished, all MSRs closed.
> **Time estimate:** 1–2 hours

### Task ALL-5.1: Writeup / Mini-Blog

- HuggingFace mini-blog or short slide deck
- Must contain: 3-sentence narrative verbatim, Tab 2 screenshot, comparison table, reward curve, honest caveats
- Honest caveats from PROJECT IDENTITY: both paragraphs verbatim
- **Time:** 60 min

---

### 🔀 PHASE 5 MERGE GATE (= Final Submission Gate)

1. MSR-4: writeup URL accessible without login
2. MSR-8: README has blog link + Colab link + W&B link
3. All MSRs 1–9 confirmed by Validator
4. Scientific honesty check: honest caveats present in README and writeup
5. All 3 team members sign off
6. **SUBMIT**

---

## PHASE 6 — FINAL VALIDATION (10 minutes max)

Run `submission_checklist.md` in order. All checkboxes must be green. No exceptions.

---

*Last updated: 2026-04-25 | Version 1.0 | Consistent with guardrails.md v1.0*
