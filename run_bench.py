"""Benchmark wrapper — patches vault isolation for auth-less test servers."""
import os
import sys

os.environ["LTM_BENCH_EVAL_MODEL"] = "openrouter/openai/gpt-4.1-mini"

# Patch: disable vault isolation (test server has no auth for dynamic vaults)
import model_interfaces.muninn_interface as mi
mi.MuninnChatSession._active_vault = property(lambda self: self.vault)

sys.argv = [
    "run_benchmark",
    "-c", "configurations/muninn_benchmark.yml",
    "-a", "muninn(url=http://127.0.0.1:18475,vault=default)",
    "-y", "-l",
]

from runner.run_benchmark import main
main()
