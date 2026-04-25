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
  - bluffbuster
---

# The Examiner — BluffBuster

Most AI benchmarks reward getting the right answer — but almost none reward asking the right question. The Examiner is an adversarial RL environment where an examiner agent learns, through information-gain reward shaping and calibrated terminal scoring, to design questions that expose confident bluffing across multiple deceptive student styles. We train a language model examiner using GRPO and demonstrate measurable improvement over definitional and random baselines on held-out student styles and unseen topic sections, with reward decomposition that judges can audit live.

## Tabs

1. **Live Episode** — Run a single episode; watch the per-section posterior belief trace live
2. **Baseline vs Trained** — Same episode seed through 4 examiners side-by-side
3. **Training Evidence** — Real plots from the W&B training run
4. **Environment Details** — Reward formula, student style specs, action schema
