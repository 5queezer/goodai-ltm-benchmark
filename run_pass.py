"""Run a single benchmark pass with configurable agent name and dream setting."""
import os
import sys

os.environ["LTM_BENCH_EVAL_MODEL"] = "openrouter/openai/gpt-4o-mini"
os.environ.pop("OPENAI_API_KEY", None)

import litellm
litellm.openai_key = None
litellm.completion_cost = lambda *a, **kw: 0.0

EVAL_MODEL = "openrouter/openai/gpt-4o-mini"
import utils.llm as llm_utils
llm_utils.GPT_CHEAPEST = EVAL_MODEL
llm_utils.GPT_4_TURBO_BEST = EVAL_MODEL
litellm.model_alias_map.update({
    "gpt-4-turbo": EVAL_MODEL, "gpt-4-turbo-2024-04-09": EVAL_MODEL,
    "gpt-3.5-turbo": EVAL_MODEL, "gpt-3.5-turbo-0125": EVAL_MODEL, "gpt-4": EVAL_MODEL,
})

import model_interfaces.muninn_interface as mi

agent_label = sys.argv[1]  # e.g. "MuninnDB Baseline"
dream = sys.argv[2] == "true"  # "true" or "false"
server = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:18475"
config = sys.argv[4] if len(sys.argv) > 4 else "configurations/muninn_benchmark.yml"
llm_model = sys.argv[5] if len(sys.argv) > 5 else None

mi.MuninnChatSession.name = property(lambda self: agent_label)
mi.MuninnChatSession.__dataclass_fields__["dream_enabled"].default = dream
if llm_model:
    mi.MuninnChatSession.__dataclass_fields__["llm_model"].default = llm_model

# Derive run name from config filename
import yaml
with open(config) as f:
    run_name = yaml.safe_load(f)["config"]["run_name"]

import shutil
results_dir = f"data/tests/{run_name}/results/{agent_label}"
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)

sys.argv = [
    "run_benchmark",
    "-c", config,
    "-a", f"muninn(url={server},vault=default)",
    "-y", "-l",
]

from runner.run_benchmark import main
main()
