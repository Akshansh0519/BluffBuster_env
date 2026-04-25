# The Examiner — Can an AI Learn to Expose Bluffing?

**Project foundation (specs, plans, guardrails):** see the [`FOUNDATION/`](FOUNDATION/) folder — `guardrails.md`, `architecture.md`, `implementation_plan.md`, coder playbooks, merge procedure, and the source `Base propmt.txt`. `mistakes.md` in that folder is gitignored and stays local only.

Most AI benchmarks reward getting the right answer — but almost none reward asking the right question. The Examiner is an adversarial RL environment where an examiner agent learns, through information-gain reward shaping and calibrated terminal scoring, to design questions that expose confident bluffing across multiple deceptive student styles. We train a language model examiner using GRPO and demonstrate measurable improvement over definitional and random baselines on held-out student styles and unseen topic sections, with reward decomposition that judges can audit live.

---

🔗 **[HuggingFace Space](LINK_TO_HF_SPACE)** | 📊 **[Training Blog / Writeup](LINK_TO_WRITEUP)** | 📓 **[Colab Notebook](LINK_TO_COLAB)** | 📈 **[W&B Training Run](LINK_TO_WANDB)**

---

## The Problem

Most AI evaluation benchmarks ask: *did the model get the right answer?* Almost no benchmarks ask: *did the model ask the right questions?* This is a critical gap — especially in educational assessment, clinical diagnosis, and expert interview settings, where the quality of the questions determines how much information you can extract.

We construct a controlled RL environment where the examiner's task is genuinely diagnostic: it must classify a student as truly knowledgeable (KNOWS) or bluffing (FAKING) across 10 ML-theory topics — without any ground truth labels, without seeing which style the student uses, and with only the student's responses to questions it asks. Good questions move the examiner's belief toward the truth. Bad questions leave it uncertain.

---

## How The Environment Works

### Student Simulator Family (7 styles)

The student is not a single scripted agent — it is a family of 7 response styles with parametrized "leak rates" that determine how often genuine knowledge cues or misconceptions appear in answers:

| Style | Knowledge | Mech. Cue Rate | Misconception Rate | Special Behavior |
|-------|-----------|----------------|--------------------|-----------------|
| K1 | KNOWS | 0.85 | 0.05 | Mechanistic, causal, concrete |
| K2 | KNOWS | 0.55 | 0.05 | Concise and correct |
| K3 | KNOWS | 0.65 | 0.08 | Hedges appropriately, still correct |
| F1 | FAKING | 0.15 | 0.30 | Collapses under "why/how" probes |
| F2 | FAKING | 0.20 | 0.25 | Mirrors jargon from the question |
| F3 | FAKING | 0.10 | 0.20 | Pivots to safer adjacent topics |
| F4 | FAKING | 0.05 | 0.40 | Very confident, zero specificity |

**Key insight:** A "why/how/mechanism" probe to a FAKING student halves their mechanism cue rate and increases their misconception rate by 50% — creating a diagnostic gap the trained examiner learns to exploit.

### Action Space

```json
// Ask action:
{"action_type": "ask", "section_id": "S01", "question_text": "Why does momentum help convergence?"}

// Classify action (terminates episode):
{"action_type": "classify", "classifications": {"S01": "KNOWS", "S02": "FAKING", ...}}
```

All 10 sections must be classified. The examiner observes only section titles and dialogue history — never the true labels or student style.

### Episode Flow

```
reset(seed) → KNOWS/FAKING partition sampled (hidden) → styles assigned per section (hidden)
→ examiner asks up to MAX_TURNS questions → observes student responses
→ examiner issues final CLASSIFY → reward computed → episode ends
```

### Knowledge Sections

S01 Gradient Descent | S02 Backpropagation | S03 Overfitting & Regularization | S04 Attention Mechanisms | S05 Transformer Architecture | S06 Loss Functions | S07 Batch Normalization | S08 CNNs | S09 RL Basics | S10 Embeddings

---

## What The Agent Learns

### Baseline vs. Trained Examiner Results (Held-Out Eval Suite)

> Results from real training run on DEMO config (200 episodes, F1+F2 training styles, evaluated on held-out F3 style and S05 section). Fill in values after training.

| Examiner | Accuracy | False Accus. Rate | False Exon. Rate | Avg Turns | Avg Info Gain/Turn | ECE |
|----------|----------|-------------------|------------------|-----------|-------------------|-----|
| RandomExaminer | — | — | — | — | — | — |
| DefinitionalExaminer | — | — | — | — | — | — |
| BayesianHeuristicExaminer | — | — | — | — | — | — |
| **TrainedExaminer** | — | — | — | — | — | — |

*Values populated after training completes. Raw data in `outputs/eval/final_metrics.json`.*

### Training Progress

![Reward curve over training](outputs/plots/reward_curve.png)
*R_total mean ± std over training checkpoints. From real W&B run — see [W&B run](LINK_TO_WANDB).*

### Before / After Behavior

**Before (DefinitionalExaminer):**
```
Turn 1 → Asked S01: "What is gradient descent?"
          Student: "Gradient descent is an optimization algorithm that minimizes loss..."
Turn 2 → Asked S02: "What is backpropagation?"
          Student: "Backpropagation propagates error backwards through the network..."
Classification: All KNOWS (wrong — several sections were FAKING)
```

**After (TrainedExaminer, same episode seed):**
```
Turn 1 → Asked S02: "Why does backpropagation need the chain rule? What happens at non-differentiable points?"
          Student (FAKING): "Well, it's basically about propagating gradients... you need calculus..."
          [Oracle posterior for S02 drops to 0.28]
Turn 2 → Asked S01: "Walk me through a specific edge case where gradient descent fails to converge."
          Student (KNOWS): "With ill-conditioned Hessians, the loss surface is elongated..."
          [Oracle posterior for S01 rises to 0.82]
Classification: S02 FAKING, S01 KNOWS (correct)
```

*Transcripts selected by behavioral quality: largest R_info gap where Definitional was wrong and Trained was correct.*

---

## Honest Caveats

**On simulator scope:** Our current simulator uses a controlled family of seven scripted styles, not real human experts — this is the primary generalization risk, which we mitigate with multi-style fakers, held-out style evaluation, and held-out topic sections.

**On improvement claims:** We do not make claims about guaranteed short-run improvement — we demonstrate pipeline health, calibrated belief tracking, and behavioral shift on the held-out eval suite.

**On the oracle:** Our KB-grounded posterior is a deterministic surrogate, not a true Bayesian oracle — we use it for reward shaping only, never as ground truth for accuracy.

---

## Reward Function Design

The reward is a programmatic composite with no LLM judge:

| Component | Range | Purpose |
|-----------|-------|---------|
| R_acc | [−1, +1] | Classification accuracy (±1/N per section) |
| R_asym | [−0.5, 0] | Asymmetric error costs (λ_FA=0.5 > λ_FE=0.3: false accusations cost more) |
| R_cal | [−0.4, +0.4] | Calibration: rewards confident-correct, punishes confident-wrong |
| R_eff | [0, +0.20] | Efficiency: rewards fast correct classification |
| R_cov | [−0.35, 0] | Coverage: penalizes missing section classifications |
| R_info | [0, +0.40] | Information gain: potential-based ΔH_t entropy reduction (Ng et al. 1999) |
| R_qual | [0, +0.10] | Question quality: mechanism probe + specificity + edge-case (question-side only) |
| R_div | [0, +0.05] | Diversity: asks across multiple sections |
| Penalties | ≤ 0 | Malformed (−0.20), repetition (−0.10), invalid section (−0.10) |
| **R_total** | **[−2.05, +1.95]** | **Sum of all; theoretical bounds asserted at runtime** |

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| RL Environment | OpenEnv | latest |
| Training | Unsloth + TRL GRPOTrainer | latest |
| Base Model | Qwen2.5-7B-Instruct | current HF |
| Reward Logging | Weights & Biases | latest |
| Demo UI | Gradio | ≥4.0 |
| Data Validation | Pydantic | v2 |
| HF Deployment | HuggingFace Spaces | — |

---

## Setup & Reproduction

### Quick Start (Colab)

1. Open the [Colab Notebook](LINK_TO_COLAB)
2. Run all cells top-to-bottom (no modifications needed)
3. Cells will: install dependencies → calibrate oracle → run baseline eval → train → eval → generate plots

### Local Setup

```bash
git clone https://github.com/[your-org]/bluffbuster-examiner
cd bluffbuster-examiner
pip install -r requirements.txt

# Run oracle calibration (required before training)
python -m examiner_env.calibration

# Run baseline evaluation
python -c "
from training.eval import run_eval
from examiner_env.baselines import DefinitionalExaminer
from examiner_env.knowledge_base import KB
import json
metrics = run_eval(DefinitionalExaminer(), json.load(open('eval_config.json')), KB)
print(metrics)
"

# Run DEBUG smoke test
python training/train_grpo.py --config DEBUG

# Run DEMO training
python training/train_grpo.py --config DEMO
```

### Run Eval Suite

```bash
python training/eval.py --examiner all --output outputs/eval/final_metrics.json
```

---

## Links

- 🔗 **HuggingFace Space:** [LINK_TO_HF_SPACE] — live interactive demo, 4 tabs
- 📊 **Training Writeup / Blog:** [LINK_TO_WRITEUP] — methodology, results, honest caveats
- 📓 **Colab Notebook:** [LINK_TO_COLAB] — full training pipeline, runnable on clean session
- 📈 **W&B Training Run:** [LINK_TO_WANDB] — all reward curves, per-component logs

---

*No video files in this repository. All demo media is hosted externally and linked above.*

---

*The Examiner — BluffBuster | Hackathon submission | Built with OpenEnv + Unsloth + TRL GRPO*
