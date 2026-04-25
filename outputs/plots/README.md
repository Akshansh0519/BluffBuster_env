# Plots — Provenance Log

All plots in this directory are generated from real training run data.
No synthetic or mocked plots. MSR-3 compliance.

## W&B Run ID
> Fill after training run: `wandb_run_id = ""`

## Generated plots
- `reward_curve.png` — R_total mean ± std over training checkpoints
- `reward_components.png` — Small-multiples: R_acc, R_info, R_cal, R_qual, R_asym
- `accuracy_curve.png` — Classification accuracy over training steps
- `false_rates_curve.png` — False accusation + false exoneration overlaid
- `info_gain_curve.png` — avg_info_gain_per_turn over training steps
- `calibration_ece_curve.png` — Calibration ECE over training steps
- `comparison_bar.png` — [4 examiners] × [accuracy + info_gain + ECE]
- `per_style_heatmap.png` — 7 styles × 10 sections accuracy heatmap
- `posterior_trace_example.png` — Per-section p_t over turns (best AFTER transcript)
