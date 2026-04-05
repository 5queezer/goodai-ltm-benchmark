#!/usr/bin/env python3
"""Bayesian ablation search over MuninnDB dream consolidation phases.

Uses Optuna TPE sampler to find the phase combination that maximises
benchmark score across PersonaChat, MultiWOZ, and the core LTM datasets.

Usage:
    /tmp/ltmbench-venv/bin/python run_ablation.py [--trials N] [--resume]

Expects:
    - ~/muninn-bench binary (built from feature/dream-engine-phase2)
    - /tmp/ltmbench-venv with optuna, muninn-sdk, datasets installed
    - OPENROUTER_API_KEY set for LLM response generation
"""

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MUNINN_BIN = Path.home() / "muninn-bench"
BENCH_DIR = Path(__file__).resolve().parent
BENCH_CONFIG = BENCH_DIR / "configurations" / "muninn_ablation.yml"
BENCH_VENV_PYTHON = Path("/tmp/ltmbench-venv/bin/python")
MUNINN_PORT = 18475
MUNINN_DATA = Path("/tmp/muninn-ablation-data")
STUDY_DB = f"sqlite:///{BENCH_DIR / 'data' / 'ablation_study.db'}"
STUDY_NAME = "dream_phase_ablation"

PHASE_IDS = ["0", "1", "2", "2b", "3", "4", "5", "6"]
PHASE_LABELS = {
    "0": "Orient", "1": "Relevance Decay", "2": "Semantic Dedup",
    "2b": "LLM Adjudication", "3": "Schema Promotion",
    "4": "Bidirectional Stability", "5": "Transitive Inference",
    "6": "Dream Journal",
}

RUN_NAME = "Dream Phase Ablation"
AGENT_NAME = "MuninnChatSession - google_gemini-3.1-flash-lite-preview"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BENCH_DIR / "data" / "ablation.log"),
    ],
)
log = logging.getLogger("ablation")


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def kill_bench_server():
    """Kill any running muninn-bench process on the benchmark port."""
    try:
        result = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if f":{MUNINN_PORT}" in line and "muninn" in line:
                # Extract PID
                for part in line.split():
                    if "pid=" in part:
                        pid = int(part.split("pid=")[1].split(",")[0].split(")")[0])
                        os.kill(pid, signal.SIGTERM)
                        log.info("Killed old server pid=%d", pid)
    except Exception:
        pass
    # Also try pkill as fallback
    subprocess.run(["pkill", "-f", "muninn-bench.*--rest-addr"], capture_output=True)
    time.sleep(1)


def start_server(dream_phases: str) -> subprocess.Popen:
    """Start MuninnDB server with the given MUNINN_DREAM_PHASES value."""
    # Fresh data dir
    if MUNINN_DATA.exists():
        shutil.rmtree(MUNINN_DATA)
    MUNINN_DATA.mkdir(parents=True)

    env = os.environ.copy()
    if dream_phases:
        env["MUNINN_DREAM_PHASES"] = dream_phases
    else:
        env.pop("MUNINN_DREAM_PHASES", None)

    proc = subprocess.Popen(
        [
            str(MUNINN_BIN), "--daemon",
            "--data", str(MUNINN_DATA),
            "--rest-addr", f"127.0.0.1:{MUNINN_PORT}",
            "--ui-addr", "127.0.0.1:18496",
            "--mcp-addr", "127.0.0.1:18760",
            "--grpc-addr", "127.0.0.1:18477",
            "--mbp-addr", "127.0.0.1:18474",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=open(BENCH_DIR / "data" / "muninn_server.log", "w"),
    )
    return proc


def wait_for_health(timeout: float = 30.0) -> bool:
    """Wait until the MuninnDB REST endpoint responds."""
    import urllib.request
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{MUNINN_PORT}/api/health"
    while time.monotonic() < deadline:
        try:
            req = urllib.request.urlopen(url, timeout=2)
            if req.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

def run_benchmark() -> dict[str, float]:
    """Run the ablation benchmark and return per-dataset normalised scores.

    Returns dict like {"PersonaChat": 0.75, "MultiWOZ": 0.33, ...}
    """
    # Clear old results for this run
    results_dir = BENCH_DIR / "data" / "tests" / RUN_NAME / "results" / AGENT_NAME
    if results_dir.exists():
        shutil.rmtree(results_dir)

    # Keep definitions across trials — they're deterministic per config.
    # This avoids re-downloading HF data and regenerating examples.

    env = os.environ.copy()
    env["LTM_BENCH_EVAL_MODEL"] = "openrouter/openai/gpt-4.1-mini"
    env["BROWSER"] = "/bin/true"  # prevent webbrowser.open_new_tab from opening reports

    # Build the command (equivalent to run_bench.py but with ablation config)
    cmd = [
        str(BENCH_VENV_PYTHON), "-c",
        (
            "import os, sys; "
            "os.environ['LTM_BENCH_EVAL_MODEL'] = 'openrouter/openai/gpt-4.1-mini'; "
            "import model_interfaces.muninn_interface as mi; "
            "mi.MuninnChatSession._active_vault = property(lambda self: self.vault); "
            "sys.argv = ['run_benchmark', "
            "'-c', 'configurations/muninn_ablation.yml', "
            f"'-a', 'muninn(url=http://127.0.0.1:{MUNINN_PORT},vault=default,model=google/gemini-3.1-flash-lite-preview)', "
            "'-y', '-l']; "
            "from runner.run_benchmark import main; main()"
        ),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BENCH_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min max
        )
        if result.returncode != 0:
            log.warning("Benchmark exited %d: %s", result.returncode, result.stderr[-500:] if result.stderr else "")
    except subprocess.TimeoutExpired:
        log.warning("Benchmark timed out after 1800s — collecting partial results")

    return parse_results()


def parse_results() -> dict[str, float]:
    """Parse result JSONs and compute per-dataset normalised scores."""
    results_dir = BENCH_DIR / "data" / "tests" / RUN_NAME / "results" / AGENT_NAME
    scores: dict[str, list[tuple[int, int]]] = {}

    if not results_dir.exists():
        log.warning("No results directory found: %s", results_dir)
        return {}

    for json_file in results_dir.rglob("*.json"):
        dataset_name = json_file.parent.name
        try:
            with open(json_file) as f:
                data = json.load(f)
            score = data.get("score", 0)
            max_score = data.get("max_score", 0)
            if max_score > 0:
                scores.setdefault(dataset_name, []).append((score, max_score))
        except Exception as e:
            log.warning("Failed to parse %s: %s", json_file, e)

    normalised: dict[str, float] = {}
    for dataset, pairs in scores.items():
        total_score = sum(s for s, _ in pairs)
        total_max = sum(m for _, m in pairs)
        normalised[dataset] = total_score / total_max if total_max > 0 else 0.0

    return normalised


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    """Single-objective: composite score across all datasets."""
    # Sample phase combination
    enabled = []
    for p in PHASE_IDS:
        param_name = f"phase_{p}"
        if trial.suggest_categorical(param_name, [True, False]):
            enabled.append(p)

    # Must have at least one phase
    if not enabled:
        log.info("Trial %d: no phases selected, returning 0", trial.number)
        return 0.0

    phase_str = ",".join(enabled)
    phase_desc = ", ".join(f"{p}({PHASE_LABELS[p]})" for p in enabled)
    log.info("Trial %d: phases=[%s]", trial.number, phase_desc)

    # Start server
    kill_bench_server()
    proc = start_server(phase_str)

    try:
        if not wait_for_health():
            log.error("Trial %d: server failed to start", trial.number)
            proc.terminate()
            return 0.0

        log.info("Trial %d: server up, running benchmark...", trial.number)
        dataset_scores = run_benchmark()

        if not dataset_scores:
            log.warning("Trial %d: no scores returned", trial.number)
            return 0.0

        # Log per-dataset scores
        for ds, score in sorted(dataset_scores.items()):
            log.info("Trial %d: %s = %.3f", trial.number, ds, score)
            trial.set_user_attr(f"score_{ds}", score)

        # Composite: mean of all dataset scores
        composite = sum(dataset_scores.values()) / len(dataset_scores)
        log.info("Trial %d: composite = %.3f (phases=%s)", trial.number, composite, phase_str)
        trial.set_user_attr("phases", phase_str)
        trial.set_user_attr("phase_desc", phase_desc)

        return composite

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Also run a baseline trial (all phases)
# ---------------------------------------------------------------------------

def run_baseline(study: optuna.Study):
    """Enqueue a baseline trial with all phases enabled."""
    study.enqueue_trial({f"phase_{p}": True for p in PHASE_IDS})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dream phase ablation search")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    args = parser.parse_args()

    # Ensure data dir exists
    (BENCH_DIR / "data").mkdir(exist_ok=True)

    if not MUNINN_BIN.exists():
        log.error("MuninnDB binary not found at %s — run bench-server.sh first", MUNINN_BIN)
        sys.exit(1)

    storage = optuna.storages.RDBStorage(STUDY_DB)

    if args.resume:
        study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
        log.info("Resumed study with %d completed trials", len(study.trials))
    else:
        # Delete existing study if any
        try:
            optuna.delete_study(study_name=STUDY_NAME, storage=storage)
        except KeyError:
            pass
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        # Ensure baseline (all phases) is the first trial
        run_baseline(study)
        log.info("Created new study, baseline enqueued")

    log.info("Starting %d trials...", args.trials)

    try:
        study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    except KeyboardInterrupt:
        log.info("Interrupted — results saved to %s", STUDY_DB)
    finally:
        kill_bench_server()

    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)

    if study.best_trial:
        bt = study.best_trial
        print(f"\nBest trial #{bt.number}:")
        print(f"  Composite score: {bt.value:.3f}")
        print(f"  Phases: {bt.user_attrs.get('phase_desc', bt.user_attrs.get('phases', '?'))}")
        for key, val in sorted(bt.user_attrs.items()):
            if key.startswith("score_"):
                print(f"  {key[6:]}: {val:.3f}")

    print(f"\nTop 10 trials:")
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True,
    )
    for t in sorted_trials[:10]:
        phases = t.user_attrs.get("phases", "?")
        print(f"  #{t.number:3d}  score={t.value:.3f}  phases={phases}")

    print(f"\nFull results: {STUDY_DB}")
    print(f"Log: {BENCH_DIR / 'data' / 'ablation.log'}")


if __name__ == "__main__":
    main()
