#!/bin/bash
set -e
cd "$(dirname "$0")"

# Clear old results
rm -rf "data/tests/MuninnDB Benchmark - 1k/results/MuninnChatSession - nvidia/nemotron-3-super-120b-a12b:free"

/tmp/ltmbench-venv/bin/python run_bench.py

echo ""
echo "=== RESULTS ==="
for f in data/tests/MuninnDB\ Benchmark\ -\ 1k/results/MuninnChatSession\ -\ nvidia/nemotron-3-super-120b-a12b:free/*/*.json; do
  [ -f "$f" ] || continue
  name=$(basename "$(dirname "$f")")
  idx=$(basename "$f" .json)
  score=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('score','N/A'))")
  echo "$name/$idx: $score"
done

echo ""
echo "=== ENGRAMS ==="
curl -sS 'http://127.0.0.1:18475/api/engrams?vault=default&limit=500' | python3 -c "import json,sys; d=json.load(sys.stdin); e=d.get('engrams',d) if isinstance(d,dict) else d; print('Surviving:', len(e))"
