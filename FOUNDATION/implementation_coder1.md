# implementation_coder1.md — C1 Environment & Simulator Playbook
## The Examiner (BluffBuster) | Coder 1 Self-Contained Reference

> **Read this file + `context_primer.md` at every session start. Read `mistakes.md` to check active errors.**
> **C1 owns `examiner_env/` and `tests/`. Do NOT touch `training/`, `scripts/`, `hf_space/`, `notebooks/`.**

---

## C1 OWNERSHIP MAP

| File | Phase | Status | Gate |
|------|-------|--------|------|
| `examiner_env/__init__.py` | 0 | [ ] | — |
| `examiner_env/models.py` | 0 | [ ] | Gate 0 |
| `examiner_env/action_parser.py` | 0 | [ ] | Gate 0 |
| `tests/test_parser.py` | 0 | [ ] | Gate 0 |
| `examiner_env/knowledge_base.py` | 0 | [ ] | Gate 1 |
| `examiner_env/student.py` | 1 | [ ] | Gate 1 |
| `examiner_env/posterior_oracle.py` | 1 | [ ] | Gate 1 |
| `examiner_env/calibration.py` | 1 | [ ] | Gate 1 |
| `examiner_env/question_features.py` | 1 | [ ] | Gate 1 |
| `examiner_env/reward.py` | 1 | [ ] | Gate 1 |
| `examiner_env/baselines.py` | 1 | [ ] | Gate 1 |
| `examiner_env/environment.py` | 1 | [ ] | Gate 1 |
| `tests/test_student_styles.py` | 1 | [ ] | Gate 1 |
| `tests/test_posterior_oracle.py` | 1 | [ ] | Gate 1 |
| `tests/test_reward.py` | 1 | [ ] | Gate 1 |

---

## C1 MSR OWNERSHIP

| MSR | C1 Responsibility |
|-----|-------------------|
| MSR-1 | Environment compliance — `ExaminerEnv` inherits from OpenEnv; `action_space` and `observation_space` defined; registered via OpenEnv registry |
| MSR-3 (partial) | Reward wiring — `RewardBreakdown` dataclass with `posterior_trace` and `info_gain_per_turn` populated correctly for W&B and HF Space |

**C1's work drives ENV_INNOV (40%) — the simulator, oracle, and reward function are the primary novelty artifacts.**

---

## PARALLEL WORK MAP

```
Phase 0:
  C1: models.py, action_parser.py, test_parser.py, knowledge_base.py (simultaneously)
  C2: repo structure, Colab skeleton, W&B setup, eval_config.json, config.py
  → Sync at Gate 0: C1 passes parser tests; C2 has Colab skeleton running

Phase 1:
  C1: student.py → test_student_styles.py → posterior_oracle.py → test_posterior_oracle.py
      → calibration.py → question_features.py → reward.py → test_reward.py
      → environment.py → baselines.py
  C2: eval.py (frozen eval suite runner) — can work in parallel once C1 has environment skeleton
  → Sync at Gate 1: C1 passes all unit tests; C2 confirms eval runner works on baselines

Phase 2:
  C1: validate reward_fn.py matches reward.py (review C2's code, don't touch it)
      verify no reward logic is re-implemented in reward_fn.py
  C2: prompt_builder.py, reward_fn.py, train_grpo.py, DEBUG smoke test
  → Sync at Gate 2: smoke test passes, C1 confirms reward wiring correct

Phase 3:
  C1: select_transcripts.py (if C2 is overloaded — coordinate first)
      review per-style accuracy heatmap for any calibration bugs
  C2: DEMO training run, plot generation
```

---

## WHAT C1 MUST NOT DO

- Touch any file in `training/`, `scripts/`, `hf_space/`, `notebooks/`
- Use token-overlap divergence as a diagnostic score (removed — see `guardrails.md` §3)
- Allow LLM calls inside `reward.py` or `posterior_oracle.py`
- Implement adaptive difficulty in the environment
- Allow loose text parsing in `action_parser.py` (must return `MalformedAction`, never coerce)
- Implement a single-style fake student (must implement all 7 styles)
- Normalize `R_total` inside `compute_reward()` (trainer owns normalization)
- Use `posterior_tracker.current_posteriors()` as accuracy ground truth (shaping only)

---

## TASK REFERENCE (Quick Navigation)

| Task | File | Time | Key Risk |
|------|------|------|----------|
| C1-0.1 | `knowledge_base.py` | 45 min | Generic mechanism_cues |
| C1-0.2 | `models.py` | 30 min | Pydantic v1 syntax |
| C1-0.3 | `action_parser.py` | 45 min | Silent coercion |
| C1-0.4 | `tests/test_parser.py` | 30 min | Missing malformed cases |
| C1-1.1 | `student.py` | 120 min | Single-style, no probe modulation |
| C1-1.2 | `tests/test_student_styles.py` | 30 min | Statistical tests too loose |
| C1-1.3 | `posterior_oracle.py` | 75 min | Missing LLR clip |
| C1-1.4 | `calibration.py` | 60 min | Brier > 0.18 |
| C1-1.5 | `question_features.py` | 45 min | Response-side input |
| C1-1.6 | `reward.py` | 90 min | Weight improvisation |
| C1-1.7 | `tests/test_reward.py` | 45 min | Decomposition tolerance |
| C1-1.8 | `baselines.py` | 75 min | BayesianHeuristic stop bug |
| C1-1.9 | `environment.py` | 90 min | OpenEnv reimplementation |

---

## DETAILED TASK SPECS

### TASK C1-0.1: Knowledge Base (`knowledge_base.py`)

**Judging criterion:** ENV_INNOV (40%) — KB quality determines whether the diagnostic signal is meaningful
**MSR:** MSR-1 partial

**Before starting:**
- Open `architecture.md` §5 for full field specifications
- Have a list of ML textbook topics for each section handy

**Session prompt context to include:**
```
Project: The Examiner RL env. KB for 10 ML sections.
Sections: S01 Gradient Descent, S02 Backpropagation, S03 Overfitting,
S04 Attention, S05 Transformer, S06 Loss Functions, S07 Batch Norm,
S08 CNN, S09 RL Basics, S10 Embeddings.
```

**Key quality checks before marking done:**
- S02 (Backpropagation): mechanism_cues include "chain rule", "Jacobian", "partial derivative", "upstream gradient"
- S04 (Attention): mechanism_cues include "query", "key", "value", "dot product", "softmax", "attention scores"
- S01 (Gradient Descent): probe_templates include at least 1 `mechanism` type and 1 `edge_case` type
- Common misconceptions are genuinely wrong things students say, not just alternative phrasing of correct answers

**Common AI mistakes for this task:**
- Generating generic mechanism_cues like "learning", "optimization", "data" — too vague for the oracle to use
- Missing `evidence_weights` field (defaults α=1.5, β=0.5, γ=1.0)
- Probe templates that don't differentiate KNOWS from FAKING (e.g., "What is gradient descent?" alone isn't diagnostic)
- Only 1-2 probe_templates per section (need ≥3, ideally 4-5)

**Post-generation verification:**
```python
from examiner_env.knowledge_base import KB
assert len(KB) == 10, "Must have all 10 sections"
for s_id, section in KB.items():
    assert len(section.mechanism_cues) >= 3, f"{s_id} needs ≥3 mechanism cues"
    assert len(section.probe_templates) >= 3, f"{s_id} needs ≥3 probe templates"
    assert any(t.probe_type in ["mechanism","edge_case"] for t in section.probe_templates), \
        f"{s_id} needs diagnostic probe templates"
```

---

### TASK C1-0.2: Pydantic Action Schemas (`models.py`)

**Key Pydantic v2 patterns (NOT v1 syntax):**
```python
# Pydantic v2 field validator:
from pydantic import BaseModel, field_validator, model_validator

class ClassifyAction(BaseModel):
    action_type: Literal["classify"]
    classifications: dict[str, Literal["KNOWS", "FAKING"]]

    @model_validator(mode="after")
    def check_all_sections(self) -> "ClassifyAction":
        CANONICAL = {"S01","S02","S03","S04","S05","S06","S07","S08","S09","S10"}
        if set(self.classifications.keys()) != CANONICAL:
            raise ValueError(f"Must have exactly all 10 sections. Got: {set(self.classifications.keys())}")
        return self
```

**Common AI mistake:** Using Pydantic v1 `@validator` instead of v2 `@field_validator` / `@model_validator`.

**Post-generation verification:**
```python
from pydantic import ValidationError
from examiner_env.models import AskAction, ClassifyAction, StudentProfile

try:
    AskAction(action_type="ask", section_id="S99", question_text="Why?")
    assert False, "Should have raised"
except ValidationError:
    pass  # correct

try:
    ClassifyAction(action_type="classify", classifications={"S01": "KNOWS"})
    assert False, "Should have raised"
except ValidationError:
    pass  # correct
```

---

### TASK C1-0.3: Action Parser (`action_parser.py`)

**The golden rule:** Parser MUST return `MalformedAction` for ANY invalid input. It MUST NOT:
- Extract JSON from prose (e.g., `re.search(r'\{.*\}', text)`)
- Strip text before parsing
- Fall back to a "best guess" action
- Use `try: fix = text.replace("'", '"')` — any coercion is a blocker

**Jaccard similarity for near-duplicate detection:**
```python
def jaccard_similarity(q1: str, q2: str) -> float:
    words1 = set(q1.lower().split())
    words2 = set(q2.lower().split())
    if not words1 and not words2: return 1.0
    return len(words1 & words2) / len(words1 | words2)

# Near-duplicate threshold: 0.85
```

**Common AI mistake:** Using `sentence-transformer` cosine similarity by default — the spec says "sentence-transformer OR KB-cue Jaccard fallback." Use Jaccard to avoid adding a heavy dependency.

---

### TASK C1-1.1: Student Simulator (`student.py`)

**This is C1's most time-intensive and highest-risk task. Plan 120 min.**

**Probe sensitivity modulation — must implement EXACTLY:**
```python
PROBE_KEYWORDS = {"why", "how", "mechanism", "because", "explain", "edge case",
                  "counterexample", "what if", "failure", "when does", "under what",
                  "why does", "derive"}

def is_probe(question_text: str) -> bool:
    q = question_text.lower()
    return any(kw in q for kw in PROBE_KEYWORDS)

# In generate_response():
if is_probe(question_text) and profile.knowledge_mode == "FAKING":
    effective_mech_rate = profile.mechanism_cue_emit_rate * 0.5   # harder to fake
    effective_misc_rate = profile.misconception_emit_rate * 1.5    # more slippage
else:
    effective_mech_rate = profile.mechanism_cue_emit_rate
    effective_misc_rate = profile.misconception_emit_rate
```

**Deterministic RNG pattern (required):**
```python
import random

def generate_response(question_text, section_id, profile, kb, episode_seed, turn):
    rng = random.Random()
    rng.seed((episode_seed, turn, section_id))
    # ALL sampling in this function uses rng.random(), rng.choice(), etc.
    # DO NOT use random.random() (global state) — use rng.xxx()
```

**F1 collapse logic (implement in style_specific_params):**
```python
if profile.style == "F1" and is_probe(question_text):
    if rng.random() < profile.style_specific_params["collapse_under_mechanism_probe"]:  # 0.80
        return f"I recall that {misconception_snippet}... but I'm not sure about the exact mechanism."
```

**Post-generation behavioral test (run manually before pushing):**
```python
from examiner_env.student import generate_response, sample_profile
from examiner_env.knowledge_base import KB

# F1 under probe: should have ≤1 mechanism cue in 80%+ of 100 samples
f1_profile = sample_profile("S01", "FAKING", episode_seed=0, section_idx=0)
# Hack style to F1 for testing
f1_profile = f1_profile.model_copy(update={"style": "F1", "mechanism_cue_emit_rate": 0.15, "misconception_emit_rate": 0.30})
probe_q = "Why does gradient descent converge? Explain the mechanism."
mech_cues = [c.phrase for c in KB["S01"].mechanism_cues if c.cue_strength == "strong"]
count_with_cue = sum(
    any(cue.lower() in generate_response(probe_q, "S01", f1_profile, KB, seed, 1).lower()
        for cue in mech_cues)
    for seed in range(100)
)
print(f"F1 under probe: {100-count_with_cue}% have ≤1 cue (want ≥80%)")
```

---

### TASK C1-1.3: Posterior Oracle (`posterior_oracle.py`)

**LLR clip is CRITICAL — missing this causes posterior explosion on outlier responses:**
```python
def compute_llr(question: str, response: str, section_id: str, kb, alpha, beta, gamma) -> float:
    evidence = alpha * mechanism_cue_coverage(response, kb[section_id]) \
               - gamma * misconception_count(response, kb[section_id])
    relevance = beta * probe_strength(question, kb[section_id].probe_templates)
    llr = relevance * evidence
    return max(-3.0, min(3.0, llr))  # CLIP TO [-3, +3] — NEVER OMIT THIS
```

**Entropy formula (use log base 2):**
```python
import math

def binary_entropy(p: float) -> float:
    p = max(0.001, min(0.999, p))  # clip for numerical stability
    return -(p * math.log2(p) + (1-p) * math.log2(1-p))

def mean_entropy(posteriors: dict[str, float]) -> float:
    return sum(binary_entropy(p) for p in posteriors.values()) / len(posteriors)
```

**Test for determinism:**
```python
tracker1 = PosteriorTracker(["S01","S02"], kb)
tracker2 = PosteriorTracker(["S01","S02"], kb)
tracker1.update("S01", "Why does backprop use chain rule?", "Because...")
tracker2.update("S01", "Why does backprop use chain rule?", "Because...")
assert tracker1.current_posteriors() == tracker2.current_posteriors()
```

---

### TASK C1-1.5: Question Features (`question_features.py`)

**CRITICAL: This function must NEVER accept `student_response` as an argument.**

**Why this matters:** If R_qual depended on the student response, a fake student could "game" it by producing responses that make the question look good. The spec explicitly requires question-side only.

**Correct signature:**
```python
def question_features(question_text: str, kb, section_id: str) -> float:
    # score = 0.40 * mechanism_probe + 0.30 * specificity + 0.30 * edge_case
```

**Wrong (will be rejected by Validator):**
```python
def question_features(question_text: str, student_response: str, kb, section_id: str) -> float:
```

---

### TASK C1-1.6: Reward Function (`reward.py`)

**Decomposition assertion (must be in `compute_reward()`):**
```python
components_sum = (rb.R_acc + rb.R_asym + rb.R_cal + rb.R_eff + rb.R_cov
                  + rb.R_info + rb.R_qual + rb.R_div
                  + rb.P_malformed + rb.P_repetition + rb.P_invalid_sec)
assert abs(components_sum - rb.R_total) < 1e-9, \
    f"Decomposition contract violated: {components_sum} != {rb.R_total}"

# Bounds check:
assert -2.05 <= rb.R_total <= 1.95, f"R_total out of bounds: {rb.R_total}"
```

**R_cal formula (exact):**
```python
# R_cal = (0.4/N) * sum_s [ sign(correct(s) - 0.5) * abs(2*p_T(s) - 1) ]
# correct(s) = 1 if classified correctly, 0 if not
# sign(1 - 0.5) = +1 (reward confident-correct)
# sign(0 - 0.5) = -1 (penalize confident-wrong)
r_cal = (0.4 / N) * sum(
    (1.0 if correct[s] else -1.0) * abs(2.0 * p_T[s] - 1.0)
    for s in sections
)
```

**R_info formula (potential-based, must be clipped correctly):**
```python
# R_info = 0.40 * clip(sum_t ΔH_t, 0, 1)
# ΔH_t per turn, summed over episode, then clip the SUM to [0, 1], then multiply 0.40
entropy_gains = tracker.get_entropy_gains()  # list[float], ΔH_t per turn
total_entropy_reduction = sum(entropy_gains)
r_info = 0.40 * max(0.0, min(1.0, total_entropy_reduction))
```

**DO NOT normalize R_total:**
```python
# WRONG:
r_total_normalized = (r_total - mean_r) / (std_r + 1e-6)
return RewardBreakdown(..., R_total=r_total_normalized)

# CORRECT:
return RewardBreakdown(..., R_total=r_total)  # raw, bounded, not normalized
```

---

### TASK C1-1.8: Baselines (`baselines.py`)

**BayesianHeuristicExaminer — the most complex baseline (and most important for credibility):**

```python
class BayesianHeuristicExaminer:
    PROBE_CYCLE = ["definitional", "mechanism", "edge_case", "counterexample"]
    CONFIDENCE_THRESHOLD = 0.4  # |p_t - 0.5| > 0.4 → section is "decided"

    def __init__(self, kb):
        self._kb = kb
        self._tracker = None  # initialized per episode in reset()
        self._probe_idx = {}  # current probe_cycle index per section

    def reset(self, section_ids):
        self._tracker = PosteriorTracker(section_ids, self._kb)
        self._probe_idx = {s: 0 for s in section_ids}

    def act(self, observation, last_response=None):
        section_ids = list(observation["section_titles"].keys())
        remaining = observation["remaining_turns"]

        # If last response was given, update tracker
        if last_response:
            last_section, last_question, last_resp = last_response
            self._tracker.update(last_section, last_question, last_resp)

        # Find most uncertain section (closest to 0.5) not yet decided
        posteriors = self._tracker.current_posteriors()
        undecided = [s for s in section_ids
                     if abs(posteriors[s] - 0.5) <= self.CONFIDENCE_THRESHOLD]

        if remaining <= 1 or not undecided:
            # Classify all sections by argmax of posterior
            classifications = {s: "KNOWS" if posteriors[s] > 0.5 else "FAKING"
                               for s in section_ids}
            return {"action_type": "classify", "classifications": classifications}

        # Ask next probe type for most uncertain section
        target_section = min(undecided, key=lambda s: abs(posteriors[s] - 0.5))
        probe_type = self.PROBE_CYCLE[self._probe_idx[target_section] % len(self.PROBE_CYCLE)]
        self._probe_idx[target_section] += 1

        templates = [t for t in self._kb[target_section].probe_templates
                     if t.probe_type == probe_type]
        if not templates:
            templates = self._kb[target_section].probe_templates  # fallback
        question = templates[0].template  # deterministic, not random

        return {"action_type": "ask", "section_id": target_section, "question_text": question}
```

---

### TASK C1-1.9: Environment Class (`environment.py`)

**BEFORE WRITING THIS PROMPT: Look up the actual OpenEnv base class interface.**

```bash
# Find the OpenEnv installation and read its base class:
python -c "import openenv; import inspect; print(inspect.getsource(openenv.Env))"
```

**Paste the ACTUAL output into your AI prompt — do not describe it, paste it.**

**Observation dict (NEVER include hidden state):**
```python
def _build_observation(self) -> dict:
    return {
        "section_titles": {s: SECTION_TITLES[s] for s in self._config.sections},
        "turn": self._state.turn,
        "remaining_turns": self._state.max_turns - self._state.turn,
        "dialogue_history": [
            {"section_id": d["section_id"],
             "question": d["question"],
             "response": d["response"]}
            for d in self._state.dialogue_history
        ]
        # NEVER include: true_labels, style_assignments, posteriors, episode_seed
    }
```

**Episode termination on turn exhaustion:**
```python
if self._state.turn >= self._state.max_turns:
    # Force classify: all sections get FAKING (worst-case default)
    forced_classify = {s: "FAKING" for s in self._config.sections}
    episode_result = self._build_episode_result(classifications=forced_classify)
    breakdown = compute_reward(episode_result, self._kb)
    return self._build_observation(), breakdown.R_total, True, False, asdict(breakdown)
```

---

## VALIDATOR HANDOFF TEMPLATE

When ready for Gate 1 review, fill in this template and send to Validator:

```
HANDOFF: C1 → Validator | Phase 1 Gate
Date: [date]
MSR Status:
  MSR-1: [PASS / FAIL — reason]
  MSR-3 (partial): [PASS / FAIL — reason]

Tests passing:
  pytest tests/test_parser.py: [PASS/FAIL — n/n tests]
  pytest tests/test_student_styles.py: [PASS/FAIL — n/n tests]
  pytest tests/test_posterior_oracle.py: [PASS/FAIL — n/n tests]
  pytest tests/test_reward.py: [PASS/FAIL — n/n tests]

Gate conditions met:
  [ ] Single episode end-to-end without crash
  [ ] R_total finite and in [-2.05, +1.95]
  [ ] Decomposition contract: sum(components) == R_total ± 1e-9
  [ ] Oracle calibration: mean_brier ≤ 0.18
  [ ] Reward variance σ ≥ 0.05 over 20 random episodes
  [ ] Observation has no hidden state

Blockers for C2 to start training:
  [list any pending issues]
```

---

## SESSION START CHECKLIST (C1)

```
[ ] git pull origin main
[ ] Read mistakes.md — check active errors list
[ ] Check which phase/task is in progress
[ ] Paste context_primer.md into AI tool
[ ] Confirm: am I working on a file in examiner_env/ or tests/?
[ ] Confirm current task's "DO NOT" list from guardrails.md §3
[ ] Run relevant existing tests to confirm baseline: pytest tests/ -v
```

## SESSION END CHECKLIST (C1)

```
[ ] All 3 sanity conditions for completed task are TRUE
[ ] pytest tests/ -v passes (no regressions)
[ ] New AI mistakes logged to mistakes.md (do NOT commit mistakes.md)
[ ] Feature branch committed with structured commit message
[ ] If at gate: handoff template sent to Validator
[ ] Update ownership map table above (mark status)
```

---

*Last updated: 2026-04-25 | Version 1.0 | C1 Playbook for BluffBuster / The Examiner*
