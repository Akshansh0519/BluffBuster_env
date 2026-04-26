"""
app.py — BluffBuster GRPO Fix Test Space (Gradio 5, live-streaming logs).
"""

import os
import sys
import json
import types
import queue
import threading
import time
import traceback as _tb

# ── Python 3.13: audioop removed; pydub (Gradio dep) needs it ─────────────
def _patch_audioop():
    stub = types.ModuleType("audioop")
    stub.error = Exception
    for _fn in (
        "add", "avg", "bias", "byteswap", "cross", "findfit", "findmax",
        "getsample", "lin2adpcm", "lin2alaw", "lin2lin", "lin2ulaw",
        "max", "maxpp", "minmax", "mul", "ratecv", "reverse", "rms",
        "tostereo", "tomono", "ulaw2lin", "adpcm2lin", "alaw2lin",
    ):
        setattr(stub, _fn, lambda *a, **k: b"")
    sys.modules.setdefault("audioop",   stub)
    sys.modules.setdefault("pyaudioop", stub)

_patch_audioop()

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Path helpers ───────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(_ROOT, "outputs", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ── Lazy env loader ────────────────────────────────────────────────────────
_env_cache = {}


def _get_env():
    if "env" in _env_cache:
        return _env_cache["env"], _env_cache["kb"], _env_cache["baselines"]
    try:
        from examiner_env.environment import ExaminerEnv
        from examiner_env.knowledge_base import KB
        from examiner_env.baselines import (
            RandomExaminer, DefinitionalExaminer, BayesianHeuristicExaminer,
        )
        _env_cache["kb"] = KB
        _env_cache["env"] = ExaminerEnv(kb=KB)
        _env_cache["baselines"] = {
            "Random": RandomExaminer(),
            "Definitional": DefinitionalExaminer(),
            "BayesianHeuristic": BayesianHeuristicExaminer(KB),
        }
    except Exception as exc:
        print(f"WARNING: examiner_env not available: {exc}")
        _env_cache["env"] = _env_cache["kb"] = None
        _env_cache["baselines"] = {}
    return _env_cache["env"], _env_cache["kb"], _env_cache["baselines"]


# ── Tab 1: Live Episode ────────────────────────────────────────────────────

def _posterior_fig(trace: list):
    fig, ax = plt.subplots(figsize=(9, 4))
    if not trace:
        ax.text(0.5, 0.5, "No posterior trace", ha="center", va="center")
        ax.set_axis_off()
        return fig
    for s_id in list(trace[0].keys()):
        vals = [t.get(s_id, 0.5) for t in trace]
        ax.plot(range(1, len(vals) + 1), vals, marker="o", linewidth=1.8, label=s_id)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Turn"); ax.set_ylabel("P(KNOWS)")
    ax.set_title("Posterior Belief Trace")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def run_live_episode(seed: int, examiner_name: str):
    env, kb, baselines = _get_env()
    empty = _posterior_fig([])
    if env is None:
        return [], empty, {}, "Environment not available."
    examiner = baselines.get(examiner_name)
    if examiner is None:
        return [], empty, {}, f"Examiner '{examiner_name}' not available."
    obs, _ = env.reset(seed=int(seed))
    done, rows, step = False, [], 0
    if hasattr(examiner, "reset"):
        examiner.reset()
    while not done:
        obs, reward, terminated, truncated, info = env.step(examiner.act(obs))
        done = terminated or truncated
        hist = obs.get("dialogue_history", [])
        if hist:
            last = hist[-1]
            rows.append([step + 1, last.get("section_id", ""),
                         last.get("question", ""), last.get("response", "")])
        step += 1
    bd = info.get("reward_breakdown")
    fig = _posterior_fig(bd.posterior_trace if bd and bd.posterior_trace else [])
    rd = ({k: round(getattr(bd, k, 0.0), 4)
           for k in ["R_acc","R_asym","R_cal","R_eff","R_cov",
                     "R_info","R_qual","R_div","P_malformed","R_total"]}
          if bd else {})
    gt = "### Ground Truth\n" + "\n".join(
        f"- **{s}**: {v}" for s, v in sorted(info.get("true_labels", {}).items())
    )
    return rows, fig, rd, gt


# ── Tab 4: Training Launcher with live streaming logs ──────────────────────

class _QueueWriter:
    """Redirects write() calls into a Queue so a generator can stream them."""
    def __init__(self, q: queue.Queue):
        self._q = q
    def write(self, text: str):
        if text:
            self._q.put(text)
    def flush(self):
        pass
    def isatty(self):
        return False


STEP_RE = None  # compiled lazily


def _estimate_total_steps(config_name: str) -> int:
    """Rough step count so we can show X/Y progress."""
    # steps = ceil(num_episodes / batch_size)
    return {
        "DEBUG":     5,
        "DEMO":      100,
        "DEMO_FAST": 50,
        "FULL":      250,
        "FULL_FAST": 150,
    }.get(config_name, 0)


def launch_baseline_stream(limit: int, max_new_tokens: int):
    """
    Generator — streams the dumb-baseline run logs to a Gradio Textbox.

    Runs `scripts.run_dumb_baseline.main` in a background thread with the
    requested limit/tokens; redirects stdout/stderr into a queue exactly
    like launch_training_stream(). Returns the path to the comparison
    plot via a final "[done]" sentinel that the caller picks up.
    """
    log_q: queue.Queue = queue.Queue()
    error_box: list = [None]
    done_flag: list = [False]

    def _run():
        old_stdout, old_stderr = sys.stdout, sys.stderr
        writer = _QueueWriter(log_q)
        sys.stdout = sys.stderr = writer
        try:
            os.makedirs("outputs/eval", exist_ok=True)
            os.makedirs("outputs/plots", exist_ok=True)

            from training.dumb_examiner import load_dumb_examiner
            from training.eval import run_eval
            from examiner_env.knowledge_base import KB
            from scripts.compare_baselines import build_comparison

            with open("eval_config.json") as f:
                eval_cfg = json.load(f)
            seeds = eval_cfg["seeds"]
            n = max(1, min(int(limit), len(seeds)))
            eval_cfg = {**eval_cfg, "seeds": seeds[:n]}
            print(f"[baseline] Using first {n} seeds (of {len(seeds)} available).")

            print(f"[baseline] Loading Qwen/Qwen2.5-1.5B-Instruct (4-bit)...")
            t0 = time.time()
            examiner = load_dumb_examiner(
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                max_seq_length=1024,
                use_4bit=True,
                max_new_tokens=int(max_new_tokens),
            )
            print(f"[baseline] Model loaded in {time.time() - t0:.1f}s.")

            print(f"[baseline] Running eval on {n} seeds "
                  f"(no LoRA, no training)...")
            t1 = time.time()
            out_path = "outputs/eval/dumb_baseline_metrics.json"
            metrics = run_eval(examiner, eval_cfg, KB, output_path=out_path)
            print(f"[baseline] Eval finished in {time.time() - t1:.1f}s.")

            print("\n=== UNTRAINED 1.5B BASELINE ===")
            for k in ["classification_accuracy", "avg_info_gain_per_turn",
                      "false_accusation_rate", "reward_mean",
                      "calibration_ECE", "parse_failure_rate"]:
                print(f"  {k:<28}: {metrics.get(k, float('nan')):+.4f}")

            print("\n[baseline] Generating comparison artifacts...")
            cmp = build_comparison()
            print(f"[baseline] Plot   : {cmp['plot_path']}")
            print(f"[baseline] Table  : {cmp['table_path']}")
            print(f"[baseline] Sources present: {cmp['n_sources']}/4")

            token = os.environ.get("HF_TOKEN")
            if token:
                try:
                    from huggingface_hub import HfApi, create_repo
                    repo_id = os.environ.get(
                        "CHECKPOINT_REPO",
                        "Samarth1401/bluffbuster-checkpoints",
                    )
                    api = HfApi(token=token)
                    try:
                        create_repo(repo_id=repo_id, token=token,
                                    private=True, repo_type="model",
                                    exist_ok=True)
                    except Exception:
                        pass
                    api.upload_file(
                        path_or_fileobj=out_path,
                        path_in_repo="dumb_baseline/dumb_baseline_metrics.json",
                        repo_id=repo_id, repo_type="model", token=token,
                        commit_message="dumb baseline metrics",
                    )
                    print(f"[baseline] Uploaded metrics to {repo_id}")
                except Exception as e:
                    print(f"[baseline] Hub upload failed ({e!r})")
            else:
                print("[baseline] HF_TOKEN not set — local file only.")

        except Exception:
            error_box[0] = _tb.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            done_flag[0] = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    accumulated = ""
    start_time = time.time()
    while not done_flag[0] or not log_q.empty():
        batch = []
        try:
            while True:
                batch.append(log_q.get_nowait())
        except queue.Empty:
            pass
        if batch:
            accumulated += "".join(batch)
        elapsed = int(time.time() - start_time)
        mins, secs = divmod(elapsed, 60)
        header = (
            f"╔══ Untrained 1.5B Baseline  "
            f"{'RUNNING' if not done_flag[0] else 'DONE'}  "
            f"elapsed {mins:02d}:{secs:02d}\n"
            f"║   limit={limit}  max_new_tokens={max_new_tokens}\n"
            f"╚══════════════════════════════════════════════\n\n"
        )
        yield header + accumulated
        if done_flag[0] and log_q.empty():
            break
        time.sleep(0.6)

    elapsed = int(time.time() - start_time)
    mins, secs = divmod(elapsed, 60)
    if error_box[0]:
        accumulated += (
            f"\n{'='*60}\nFAILED after {mins:02d}:{secs:02d}\n"
            f"{'='*60}\n{error_box[0]}"
        )
    else:
        accumulated += (
            f"\n{'='*60}\n"
            f"COMPLETE in {mins:02d}:{secs:02d}\n"
            f"{'='*60}\n"
            f"Open the **Comparison** tab and click Refresh to view the\n"
            f"4-way bar chart and metrics table.\n"
        )
    yield accumulated


def refresh_comparison():
    """Regenerate plot + table from whatever JSONs exist on disk."""
    try:
        from scripts.compare_baselines import build_comparison
        result = build_comparison()
        plot_path = result["plot_path"]
        table_md = "## Examiner Comparison\n\n" + result["table"]
        sources = ", ".join(result["sources"]) or "(none yet)"
        status = (
            f"**Sources present ({result['n_sources']}/4):** {sources}\n\n"
            f"Run order: classical baselines auto-populate when training "
            f"starts; **Dumb LLM** comes from the **Baseline (GPU)** tab; "
            f"**Trained LLM** comes from the **Train (GPU)** tab when a "
            f"FAST/FULL run completes."
        )
        plot = plot_path if plot_path and os.path.exists(plot_path) else None
        return plot, table_md, status
    except Exception as exc:
        return None, f"```\n{_tb.format_exc()}\n```", f"Error: {exc}"


def launch_training_stream(config_name: str):
    """
    Generator — yields live log text to gr.Textbox.
    Runs training in a background thread; captures all print() output via
    sys.stdout redirect into a Queue. Yields the accumulated log every ~1 s.
    """
    import re

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        yield "ERROR: WANDB_API_KEY secret not set in this Space.\n"
        return

    log_q: queue.Queue = queue.Queue()
    result_box: list = [None]
    error_box:  list = [None]
    done_flag:  list = [False]

    def _run():
        old_stdout, old_stderr = sys.stdout, sys.stderr
        writer = _QueueWriter(log_q)
        sys.stdout = sys.stderr = writer
        try:
            import torch
            from training.config import get_config
            from training.train_grpo import train
            from examiner_env.calibration import run_calibration
            from examiner_env.knowledge_base import KB

            for d in ["outputs/eval", "outputs/plots",
                      "outputs/transcripts", "outputs/checkpoints"]:
                os.makedirs(d, exist_ok=True)

            cal_path = "outputs/eval/oracle_calibration.json"
            if not os.path.exists(cal_path):
                print("Running oracle calibration (200 episodes)...")
                run_calibration(KB, n_episodes=200, output_path=cal_path)

            with open(cal_path) as f:
                cal = json.load(f)
            brier = cal["calibration_metrics"]["mean_brier"]
            if brier > 0.18:
                print(f"ERROR: Oracle Brier={brier:.4f} > 0.18 — recalibrate.")
                return

            with open("eval_config.json") as f:
                eval_config = json.load(f)

            config = get_config(config_name)
            result_box[0] = train(config, eval_config)

        except Exception:
            error_box[0] = _tb.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            done_flag[0] = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    accumulated = ""
    # Match the tqdm trainer progress bar: "  5%|▌         | 2/50 [...]"
    # Also match "[step N]" lines we print ourselves in the callback.
    # We cap against total_steps so stray episode counts (e.g. "0/100" in
    # dataset construction) don't pollute the progress display.
    total_steps = _estimate_total_steps(config_name)
    _tqdm_pat  = re.compile(r"\|\s*(\d+)/" + str(total_steps) + r"\s*[\[|]")
    _step_pat  = re.compile(r"\[step\s+(\d+)\]", re.IGNORECASE)
    steps_seen: set = set()
    start_time = time.time()
    heartbeat_tick = 0

    while not done_flag[0] or not log_q.empty():
        # Drain everything currently in the queue
        batch = []
        try:
            while True:
                batch.append(log_q.get_nowait())
        except queue.Empty:
            pass

        if batch:
            chunk = "".join(batch)
            accumulated += chunk
            # Extract step numbers for progress display.
            # Only accept values <= total_steps to avoid episode counts
            # like "0/100" polluting the step counter.
            for m in _tqdm_pat.finditer(chunk):
                v = int(m.group(1))
                if total_steps == 0 or v <= total_steps:
                    steps_seen.add(v)
            for m in _step_pat.finditer(chunk):
                v = int(m.group(1))
                if total_steps == 0 or v <= total_steps:
                    steps_seen.add(v)
        else:
            # Heartbeat every 5 s when there's no output
            heartbeat_tick += 1
            if heartbeat_tick % 5 == 0:
                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                ticker = "." * (heartbeat_tick // 5 % 4)
                heartbeat = f"\n[running{ticker}  elapsed {mins:02d}:{secs:02d}]"
                accumulated_with_hb = accumulated + heartbeat
                # Build progress header
                header = _progress_header(config_name, steps_seen, total_steps,
                                          elapsed, running=True)
                yield header + accumulated_with_hb
                time.sleep(1.0)
                continue

        # Build progress summary header
        elapsed = int(time.time() - start_time)
        header = _progress_header(config_name, steps_seen, total_steps,
                                  elapsed, running=not done_flag[0])
        yield header + accumulated
        time.sleep(0.5)

    # Final yield with result
    elapsed = int(time.time() - start_time)
    mins, secs = divmod(elapsed, 60)

    if error_box[0]:
        final_status = (
            f"\n{'='*60}\n"
            f"FAILED after {mins:02d}:{secs:02d}\n"
            f"{'='*60}\n"
            f"{error_box[0]}"
        )
    elif result_box[0]:
        fm = result_box[0]
        final_status = (
            f"\n{'='*60}\n"
            f"COMPLETE in {mins:02d}:{secs:02d}\n"
            f"  classification_accuracy : {fm.get('classification_accuracy', float('nan')):.3f}\n"
            f"  avg_info_gain_per_turn  : {fm.get('avg_info_gain_per_turn', float('nan')):.4f}\n"
            f"  calibration_ECE         : {fm.get('calibration_ECE', float('nan')):.4f}\n"
            f"  reward_mean             : {fm.get('reward_mean', float('nan')):.4f}\n"
            f"{'='*60}\n"
            f"Artifacts: outputs/eval/  |  WandB: https://wandb.ai\n"
            f"(project: bluffbuster-examiner)"
        )
    else:
        final_status = f"\n[finished in {mins:02d}:{secs:02d} — no result captured]"

    header = _progress_header(config_name, steps_seen, total_steps,
                              elapsed, running=False)
    yield header + accumulated + final_status


def _progress_header(config_name: str, steps_seen: set, total_steps: int,
                     elapsed_s: int, running: bool) -> str:
    mins, secs = divmod(elapsed_s, 60)
    status = "RUNNING" if running else "DONE"
    if not steps_seen or total_steps == 0:
        bar = "[----------]  0%"
        step_str = "step ?/?  "
    else:
        done = max(steps_seen)
        pct = min(100, int(done / total_steps * 100))
        filled = pct // 10
        bar = "[" + "#" * filled + "-" * (10 - filled) + f"]  {pct}%"
        step_str = f"step {done}/{total_steps}  "
    if running and elapsed_s > 0 and steps_seen and total_steps > 0:
        done_steps = max(steps_seen)
        rate = done_steps / elapsed_s if elapsed_s else 0
        remaining = (total_steps - done_steps) / rate if rate > 0 else 0
        eta_m, eta_s = divmod(int(remaining), 60)
        eta_str = f"ETA {eta_m:02d}:{eta_s:02d}"
    else:
        eta_str = ""
    header = (
        f"╔══ {config_name} training  {status}  "
        f"elapsed {mins:02d}:{secs:02d}  {eta_str}\n"
        f"║   {bar}  {step_str}\n"
        f"╚══════════════════════════════════════════════\n\n"
    )
    return header


# ── Environment Details ────────────────────────────────────────────────────

ENV_MD = """
## Student Simulator (7 styles)

| Style | Knowledge | Mech. Cue Rate | Misc. Rate |
|-------|-----------|----------------|------------|
| K1 | KNOWS | 0.85 | 0.05 |
| K2 | KNOWS | 0.55 | 0.05 |
| K3 | KNOWS | 0.65 | 0.08 |
| F1 | FAKING | 0.15 | 0.30 |
| F2 | FAKING | 0.20 | 0.25 |
| F3 | FAKING | 0.10 | 0.20 |
| F4 | FAKING | 0.05 | 0.40 |

## Reward Components

| Component | Range | Purpose |
|-----------|-------|---------|
| R_acc | [-1, +1] | Classification accuracy |
| R_asym | [-0.5, 0] | Asymmetric error cost (false accusation costs more) |
| R_cal | [-0.4, +0.4] | Calibration quality |
| R_eff | [0, +0.20] | Efficiency: fast correct classification |
| R_info | [0, +0.40] | Information gain per turn |
| R_qual | [0, +0.10] | Question quality (WHY/HOW/edge-case) |
| **R_total** | **[-2.05, +1.95]** | Sum of all components |

## GRPO Fix (this Space)

Patch branch: `fix/c2-unsloth-grpo-patch`

- `_safe_chunked_hidden_states_selective_log_softmax` — guaranteed 2D output,
  Parameter-before-Tensor detection, hidden→logits projection, shape-mismatch crop
- `_install_unsloth_chunked_logsoftmax_patch()` — sweeps `sys.modules` twice
  (post `FastLanguageModel.from_pretrained` + post `GRPOTrainer.__init__`)
- No `environment_factory`; ExaminerEnv rolls out inside `reward_func` per completion
"""


# ── Build Gradio app ───────────────────────────────────────────────────────

def create_app():
    with gr.Blocks(title="BluffBuster GRPO Fix") as app:

        gr.Markdown(
            "# BluffBuster — GRPO Fix Test\n"
            "Branch `fix/c2-unsloth-grpo-patch` — Unsloth chunked log-softmax patch. "
            "Use **Train (GPU)** to verify no crash on step 1."
        )

        # ── Tab 1: Live Episode ──
        with gr.Tab("Live Episode"):
            with gr.Row():
                seed_in = gr.Number(value=1000, label="Episode Seed", precision=0)
                exam_dd = gr.Dropdown(
                    choices=["Definitional", "BayesianHeuristic", "Random"],
                    value="Definitional",
                    label="Examiner",
                )
                run_btn = gr.Button("Run Episode", variant="primary")
            dialogue_tbl = gr.Dataframe(
                headers=["Turn", "Section", "Question", "Student Response"],
                label="Dialogue", wrap=True,
            )
            posterior_plt = gr.Plot(label="Posterior Belief Trace")
            reward_json  = gr.JSON(label="Reward Breakdown")
            gt_md        = gr.Markdown()
            run_btn.click(
                fn=run_live_episode,
                inputs=[seed_in, exam_dd],
                outputs=[dialogue_tbl, posterior_plt, reward_json, gt_md],
            )

        # ── Tab 2: Training Evidence ──
        with gr.Tab("Training Evidence"):
            gr.Markdown(
                "Plots appear here after training completes. "
                "Run DEBUG in the **Train (GPU)** tab first."
            )
            for fname, lbl in [
                ("reward_curve.png",   "R_total over Training"),
                ("accuracy_curve.png", "Accuracy over Training"),
                ("comparison_bar.png", "4-Examiner Comparison"),
                ("info_gain_curve.png","Avg Info Gain/Turn"),
            ]:
                p = os.path.join(PLOT_DIR, fname)
                gr.Image(value=p if os.path.exists(p) else None, label=lbl)

        # ── Tab 3: Environment Details ──
        with gr.Tab("Environment Details"):
            gr.Markdown(ENV_MD)

        # ── Tab 4: Train (GPU) ──
        with gr.Tab("Train (GPU)"):
            gr.Markdown(
                "## GRPO Training Launcher\n\n"
                "Credentials are pre-loaded from Space secrets. "
                "Pick a config and click **Launch** — logs stream live.\n\n"
                "**If the run was interrupted, just click Launch again — it will resume from the last checkpoint automatically.**\n\n"
                "| Config | Model | Episodes | Est. time (A10G) | Est. cost | Notes |\n"
                "|--------|-------|----------|------------------|-----------|-------|\n"
                "| DEBUG     | Qwen 1.5B 4-bit | 20  | ~10 min  | ~$0.18  | Smoke test only |\n"
                "| DEMO_FAST | Qwen 1.5B 4-bit | 100 | ~25 min  | ~$0.45  | ✅ Recommended if time is short |\n"
                "| FULL_FAST | Qwen 1.5B 4-bit | 300 | ~60 min  | ~$1.05  | ✅ Full evidence with 1.5B |\n"
                "| DEMO      | Qwen 7B 4-bit   | 200 | ~2 hr    | ~$2.10  | Higher quality, more time |\n"
                "| FULL      | Qwen 7B 4-bit   | 500 | ~5 hr    | ~$5.25  | Best quality, needs time |\n"
            )
            cfg_dd    = gr.Dropdown(
                choices=["DEBUG", "DEMO_FAST", "FULL_FAST", "DEMO", "FULL"],
                value="DEMO_FAST", label="Config",
            )
            train_btn = gr.Button("Launch Training", variant="primary", size="lg")
            train_out = gr.Textbox(
                label="Live Training Log",
                lines=30, max_lines=60,
                interactive=False,
                placeholder=(
                    "Training log will stream here in real time.\n\n"
                    "You will see:\n"
                    "  - Oracle calibration check\n"
                    "  - Baseline eval (3 examiners)\n"
                    "  - Model loading + LoRA\n"
                    "  - [unsloth-patch] confirmation line\n"
                    "  - Per-step reward logs from TRL\n"
                    "  - Final eval metrics\n"
                ),
            )
            stop_note = gr.Markdown(
                "_Training runs on the Space GPU. Closing the browser does not stop it._"
            )
            train_btn.click(
                fn=launch_training_stream,
                inputs=[cfg_dd],
                outputs=[train_out],
            )

        # ── Tab 5: Baseline (GPU) — untrained 1.5B baseline runner ──
        with gr.Tab("Baseline (GPU)"):
            gr.Markdown(
                "## Untrained 1.5B Baseline (the LLM 'before' row)\n\n"
                "Runs **the same Qwen2.5-1.5B-Instruct, the same prompt builder, "
                "the same env, the same seeds** — but with **NO LoRA** and "
                "**no GRPO training**. This is the scientifically-correct "
                "baseline to compare your trained model against, because the "
                "only varying factor is the GRPO LoRA delta.\n\n"
                "Defaults are tuned for **<10 min on A10G**:\n\n"
                "| Knob | Default | Why |\n"
                "|------|---------|-----|\n"
                "| `limit` (eval seeds) | 15 | Matches FAST_CONFIG.eval_episodes; "
                "subset of the full 50-seed suite (seed-matched). |\n"
                "| `max_new_tokens` | 64 | Enough for one valid ASK/CLASSIFY "
                "JSON; halves wall-clock vs 128. |\n"
                "| model | Qwen2.5-1.5B-Instruct | Same as `FAST_CONFIG`. |\n"
                "| precision | 4-bit | Same as training. |\n\n"
                "After the run completes, click **Refresh** in the "
                "**Comparison** tab to see the 4-way bar chart."
            )
            with gr.Row():
                bl_limit = gr.Slider(
                    minimum=5, maximum=50, step=5, value=15,
                    label="Eval seeds (--limit)",
                )
                bl_tokens = gr.Slider(
                    minimum=32, maximum=256, step=16, value=64,
                    label="max_new_tokens",
                )
            bl_btn = gr.Button("Run Untrained Baseline", variant="primary",
                               size="lg")
            bl_out = gr.Textbox(
                label="Live Baseline Log",
                lines=24, max_lines=50, interactive=False,
                placeholder=(
                    "Click Run Untrained Baseline to start.\n\n"
                    "Phases:\n"
                    "  1. Load Qwen2.5-1.5B-Instruct (4-bit, ~1-2 min)\n"
                    "  2. Run frozen eval on N seeds (~3-6 min)\n"
                    "  3. Save outputs/eval/dumb_baseline_metrics.json\n"
                    "  4. Build 4-way comparison plot + table\n"
                    "  5. Upload metrics to HF Hub (if HF_TOKEN set)"
                ),
            )
            bl_btn.click(
                fn=launch_baseline_stream,
                inputs=[bl_limit, bl_tokens],
                outputs=[bl_out],
            )

        # ── Tab 6: Comparison — 4-way bar + table ──
        with gr.Tab("Comparison"):
            gr.Markdown(
                "## 4-way Examiner Comparison\n\n"
                "Pulls in whatever metric files exist on disk and renders the "
                "bar chart + markdown table. Re-click **Refresh** after "
                "running the **Baseline (GPU)** tab and after a training run "
                "completes."
            )
            cmp_btn = gr.Button("Refresh Comparison", variant="primary")
            cmp_status = gr.Markdown()
            cmp_plot = gr.Image(label="four_way_comparison.png", type="filepath")
            cmp_table = gr.Markdown()
            cmp_btn.click(
                fn=refresh_comparison,
                inputs=[],
                outputs=[cmp_plot, cmp_table, cmp_status],
            )

    return app


app = create_app()

if __name__ == "__main__":
    app.launch()
