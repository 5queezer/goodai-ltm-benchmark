"""Run baseline vs dream benchmark comparison for PR charts."""
import os
import sys
import shutil

# Route ALL LLM eval calls through OpenRouter (avoid invalid OPENAI_API_KEY)
os.environ["LTM_BENCH_EVAL_MODEL"] = "openrouter/openai/gpt-4o-mini"
os.environ.pop("OPENAI_API_KEY", None)  # prevent litellm from using OpenAI directly

# Patch litellm to use OpenRouter for OpenAI model aliases
import litellm
litellm.openai_key = None
# Disable cost tracking — litellm doesn't have pricing for OpenRouter-routed models
litellm.completion_cost = lambda *a, **kw: 0.0

# Patch GPT constants and alias map to route everything through OpenRouter
import utils.llm as llm_utils
EVAL_MODEL = "openrouter/openai/gpt-4o-mini"
llm_utils.GPT_CHEAPEST = EVAL_MODEL
llm_utils.GPT_4_TURBO_BEST = EVAL_MODEL
litellm.model_alias_map.update({
    "gpt-4-turbo": EVAL_MODEL,
    "gpt-4-turbo-2024-04-09": EVAL_MODEL,
    "gpt-3.5-turbo": EVAL_MODEL,
    "gpt-3.5-turbo-0125": EVAL_MODEL,
    "gpt-4": EVAL_MODEL,
})

RUN_NAME = "MuninnDB Benchmark - 1k"
RESULTS_BASE = f"data/tests/{RUN_NAME}/results"
SERVER = "http://127.0.0.1:18475"

import model_interfaces.muninn_interface as mi


def run_benchmark(agent_label: str, dream_enabled: bool):
    """Run one benchmark pass with a distinct agent name."""
    agent_dir = os.path.join(RESULTS_BASE, agent_label)
    if os.path.exists(agent_dir):
        shutil.rmtree(agent_dir)

    # Override agent name and dream setting
    mi.MuninnChatSession.name = property(lambda self: agent_label)
    mi.MuninnChatSession.dream_enabled = dream_enabled

    sys.argv = [
        "run_benchmark",
        "-c", "configurations/muninn_benchmark.yml",
        "-a", f"muninn(url={SERVER},vault=default)",
        "-y", "-l",
    ]

    # Re-import to reset state
    from importlib import reload
    import runner.run_benchmark as rb
    reload(rb)
    rb.main()


if __name__ == "__main__":
    print("=" * 60)
    print("PASS 1: Baseline (no dream consolidation)")
    print("=" * 60)
    run_benchmark("MuninnDB Baseline", dream_enabled=False)

    print()
    print("=" * 60)
    print("PASS 2: With Dream Engine (0.99 threshold)")
    print("=" * 60)
    run_benchmark("MuninnDB Dream", dream_enabled=True)

    print()
    print("BOTH PASSES COMPLETE")
    print(f"Results in: {RESULTS_BASE}/")
