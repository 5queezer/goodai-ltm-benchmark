#!/usr/bin/env python3
"""Run three Gemini benchmark variants: baseline (no dream), full phases, best combo (1,2,5)."""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

MUNINN_BIN = Path.home() / "muninn-bench"
BENCH_DIR = Path(__file__).resolve().parent
BENCH_VENV_PYTHON = Path("/tmp/ltmbench-venv/bin/python")
MUNINN_PORT = 18475
MUNINN_DATA = Path("/tmp/muninn-bench-data")
MODEL = "google/gemini-3.1-flash-lite-preview"
AGENT_NAME = f"MuninnChatSession - {MODEL.replace('/', '_')}"

VARIANTS = [
    {"name": "Gemini Baseline vault-isolated", "dream": False, "phases": None},
    {"name": "Gemini Full Phases vault-isolated", "dream": True, "phases": None},
    {"name": "Gemini Best 1-2-5 vault-isolated", "dream": True, "phases": "1,2,5"},
]


def kill_server():
    """Kill muninn-bench on the benchmark port."""
    try:
        result = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if f":{MUNINN_PORT}" in line and "muninn" in line:
                for part in line.split():
                    if "pid=" in part:
                        pid = int(part.split("pid=")[1].split(",")[0].split(")")[0])
                        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass
    time.sleep(1)


def start_server(dream_phases=None):
    if MUNINN_DATA.exists():
        shutil.rmtree(MUNINN_DATA)
    MUNINN_DATA.mkdir(parents=True)
    env = os.environ.copy()
    env["MUNINN_DEFAULT_VAULT_PUBLIC"] = "true"
    if dream_phases:
        env["MUNINN_DREAM_PHASES"] = dream_phases
    else:
        env.pop("MUNINN_DREAM_PHASES", None)
    return subprocess.Popen(
        [str(MUNINN_BIN), "--daemon",
         "--data", str(MUNINN_DATA),
         "--rest-addr", f"127.0.0.1:{MUNINN_PORT}",
         "--ui-addr", "127.0.0.1:18496",
         "--mcp-addr", "127.0.0.1:18760",
         "--grpc-addr", "127.0.0.1:18477",
         "--mbp-addr", "127.0.0.1:18474"],
        env=env, stdout=subprocess.DEVNULL,
        stderr=open(BENCH_DIR / "data" / "muninn_server.log", "w"),
    )


def wait_for_health(timeout=30.0):
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{MUNINN_PORT}/api/health", timeout=2).status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def make_config(run_name):
    """Create a variant config YAML with the given run_name."""
    import yaml
    base = BENCH_DIR / "configurations" / "muninn_benchmark.yml"
    with open(base) as f:
        cfg = yaml.safe_load(f)
    cfg["config"]["run_name"] = run_name
    out = BENCH_DIR / "configurations" / f"_trio_{run_name.replace(' ', '_').lower()}.yml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return out


def run_benchmark(run_name, dream_enabled):
    dream_flag = "true" if dream_enabled else "false"
    config_path = make_config(run_name)
    env = os.environ.copy()
    env["LTM_BENCH_EVAL_MODEL"] = "openrouter/openai/gpt-4.1-mini"
    env["BROWSER"] = "/bin/true"

    cmd = [
        str(BENCH_VENV_PYTHON), "-c",
        (
            "import os, sys; "
            "os.environ['LTM_BENCH_EVAL_MODEL'] = 'openrouter/openai/gpt-4.1-mini'; "
            f"sys.argv = ['run_benchmark', "
            f"'-c', '{config_path.relative_to(BENCH_DIR)}', "
            f"'-a', 'muninn(url=http://127.0.0.1:{MUNINN_PORT},vault=default,model={MODEL},dream={dream_flag})', "
            f"'-y', '-l']; "
            "from runner.run_benchmark import main; main()"
        ),
    ]

    try:
        result = subprocess.run(cmd, cwd=str(BENCH_DIR), env=env,
                                capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            print(f"  Warning: benchmark exited {result.returncode}")
    except subprocess.TimeoutExpired:
        print("  Warning: benchmark timed out")


def parse_results(run_name):
    results_dir = BENCH_DIR / "data" / "tests" / run_name / "results" / AGENT_NAME
    scores = {}
    if not results_dir.exists():
        return scores
    for json_file in results_dir.rglob("*.json"):
        if json_file.name in ("runstats.json", "master_log.jsonl"):
            continue
        dataset_name = json_file.parent.name
        try:
            with open(json_file) as f:
                data = json.load(f)
            score = data.get("score", 0)
            max_score = data.get("max_score", 0)
            if max_score > 0:
                scores.setdefault(dataset_name, []).append((score, max_score))
        except Exception:
            pass
    normalised = {}
    for dataset, pairs in scores.items():
        total_score = sum(s for s, _ in pairs)
        total_max = sum(m for _, m in pairs)
        normalised[dataset] = total_score / total_max if total_max > 0 else 0.0
    return normalised


def main():
    all_results = {}

    for i, variant in enumerate(VARIANTS):
        name = variant["name"]
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(VARIANTS)}] {name}")
        print(f"{'='*60}")

        kill_server()

        # Clean old results for this run
        run_dir = BENCH_DIR / "data" / "tests" / name
        if run_dir.exists():
            shutil.rmtree(run_dir)

        proc = start_server(variant["phases"])
        if not wait_for_health():
            print("  ERROR: server failed to start")
            proc.terminate()
            continue

        print(f"  Server up (phases={variant['phases'] or 'all'}), running benchmark...")
        start = time.time()
        run_benchmark(name, variant["dream"])
        elapsed = time.time() - start

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

        scores = parse_results(name)
        all_results[name] = scores

        if scores:
            composite = sum(scores.values()) / len(scores)
            print(f"  Done in {elapsed:.0f}s — composite: {composite:.3f}")
            for ds, score in sorted(scores.items()):
                print(f"    {ds}: {score:.3f}")
        else:
            print(f"  Done in {elapsed:.0f}s — no scores")

    kill_server()

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    datasets = sorted(set(ds for scores in all_results.values() for ds in scores))
    header = f"{'Variant':<30}" + "".join(f"{ds:>12}" for ds in datasets) + f"{'Composite':>12}"
    print(header)
    print("-" * len(header))

    for name, scores in all_results.items():
        if not scores:
            continue
        composite = sum(scores.values()) / len(scores)
        row = f"{name:<30}"
        for ds in datasets:
            row += f"{scores.get(ds, 0.0):>12.3f}"
        row += f"{composite:>12.3f}"
        print(row)


if __name__ == "__main__":
    main()
