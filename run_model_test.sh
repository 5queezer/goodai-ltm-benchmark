#!/usr/bin/env bash
# Run baseline + dream for multiple models
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

SERVER="http://127.0.0.1:18475"
CONFIG="configurations/muninn_benchmark_full.yml"

# Model list: label model_id
run_model() {
    local label="$1"
    local model="$2"
    local dream="$3"
    local suffix
    if [ "$dream" = "true" ]; then suffix="Dream"; else suffix="Baseline"; fi
    local agent="${label} ${suffix}"

    echo "============================================================"
    echo "PASS: ${agent} (${model})"
    echo "============================================================"
    python run_pass.py "$agent" "$dream" "$SERVER" "$CONFIG" "openrouter/${model}"
}

# GLM 4.5 Air
run_model "GLM4.5Air" "z-ai/glm-4.5-air:free" false
run_model "GLM4.5Air" "z-ai/glm-4.5-air:free" true

# Gemma 4 26B A4B
run_model "Gemma4-26B" "google/gemma-4-26b-a4b-it" false
run_model "Gemma4-26B" "google/gemma-4-26b-a4b-it" true

# Qwen 3.6 Plus (free — may rate-limit)
run_model "Qwen3.6Plus" "qwen/qwen3.6-plus:free" false
run_model "Qwen3.6Plus" "qwen/qwen3.6-plus:free" true

echo ""
echo "ALL PASSES COMPLETE"
