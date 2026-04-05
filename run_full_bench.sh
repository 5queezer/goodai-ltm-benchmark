#!/usr/bin/env bash
# Run 4-pass benchmark: nemotron vs gpt-4o-mini, baseline vs dream
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

SERVER="http://127.0.0.1:18475"
CONFIG="configurations/muninn_benchmark_full.yml"

echo "============================================================"
echo "PASS 1/4: Nemotron Baseline (no dream)"
echo "============================================================"
python run_pass.py "Nemotron Baseline" false "$SERVER" "$CONFIG"

echo ""
echo "============================================================"
echo "PASS 2/4: Nemotron Dream"
echo "============================================================"
python run_pass.py "Nemotron Dream" true "$SERVER" "$CONFIG"

echo ""
echo "============================================================"
echo "PASS 3/4: GPT-4o-mini Baseline (no dream)"
echo "============================================================"
python run_pass.py "GPT4oMini Baseline" false "$SERVER" "$CONFIG" "openrouter/openai/gpt-4o-mini"

echo ""
echo "============================================================"
echo "PASS 4/4: GPT-4o-mini Dream"
echo "============================================================"
python run_pass.py "GPT4oMini Dream" true "$SERVER" "$CONFIG" "openrouter/openai/gpt-4o-mini"

echo ""
echo "ALL 4 PASSES COMPLETE"
echo "Results in: data/tests/MuninnDB Benchmark Full - 1k/results/"
