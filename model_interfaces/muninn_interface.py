"""MuninnDB model interface for goodai-ltm-benchmark.

Bridges the benchmark's conversational ChatSession protocol with MuninnDB's
engram write/recall API. Uses a local LLM (OpenAI-compatible) to generate
responses from recalled context.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import httpx

from model_interfaces.interface import ChatSession

logger = logging.getLogger(__name__)


@dataclass
class MuninnChatSession(ChatSession):
    """ChatSession backed by MuninnDB for long-term memory.

    Writes user messages as engrams, recalls relevant context for each query,
    and uses a local LLM to generate responses from recalled memories.
    """

    # MuninnDB connection
    muninn_url: str = "http://localhost:8476"
    muninn_token: str = ""
    vault: str = "default"

    # LLM for response generation (OpenAI-compatible endpoint)
    llm_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "nvidia/nemotron-3-super-120b-a12b:free"

    # Recall tuning
    max_recall: int = 15
    recall_threshold: float = 0.1

    # Dream consolidation
    dream_enabled: bool = True
    dream_dry_run: bool = False
    dream_force: bool = True

    # Trace capture
    trace_enabled: bool = True

    # Always local (no cost tracking)
    is_local: bool = True

    # Internal state
    _write_count: int = 0
    _session_tag: str = ""
    _run_id: str = ""
    _vault_counter: int = 0
    _muninn_healthy: bool = True
    _io_errors: int = 0

    # System prompt for the LLM
    _system_prompt: str = (
        "You are a recall assistant. "
        "Reply with ONLY the direct answer. "
        "No reasoning, no explanation, no preamble, no chain-of-thought. "
        "If the memory context contains the answer, state it in the fewest words possible. "
        "If not, reply: I don't remember."
    )

    def __post_init__(self):
        super().__post_init__()
        self._session_tag = f"bench_{uuid.uuid4().hex[:12]}"
        self._run_id = uuid.uuid4().hex[:12]
        self._loop = asyncio.new_event_loop()
        self._http_client = None  # lazy init
        self._muninn_client = None  # lazy init
        self._trace_file = None

    @property
    def name(self):
        return f"MuninnChatSession - {self.llm_model}"

    @property
    def _active_vault(self):
        return f"{self.vault}-{self._run_id}-{self._vault_counter}"

    def _ensure_clients(self):
        """Lazily initialize HTTP clients."""
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=30.0)
        if self._muninn_client is None:
            # Import here to avoid hard dependency at module level
            try:
                from muninn import MuninnClient
                self._muninn_client = MuninnClient(
                    self.muninn_url,
                    token=self.muninn_token or None,
                )
                self._loop.run_until_complete(self._muninn_client.__aenter__())
            except ImportError as exc:
                raise RuntimeError(
                    "MuninnDB Python SDK not found. "
                    "Install it as a dependency (pip install muninn-sdk)."
                ) from exc

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:
        if agent_response is not None:
            # Filler message — skip vault interaction entirely
            return agent_response

        self._ensure_clients()

        # 1. Recall relevant memories BEFORE writing (so the current message
        #    doesn't pollute its own recall results)
        recalled = self._recall(user_message)

        # 2. Generate response using LLM with recalled context
        response = self._generate_response(user_message, recalled)

        # 3. Capture trace if enabled
        if self.trace_enabled:
            self._write_trace(user_message, recalled, response)

        # 4. Write the user message as an engram
        self._write_engram(user_message)

        # Never return None — the benchmark's flatten_context crashes on it
        return response or "Understood."

    def _recall(self, query: str) -> list[dict]:
        """Recall relevant engrams from the vault."""
        try:
            result = self._loop.run_until_complete(
                self._muninn_client.activate(
                    vault=self._active_vault,
                    context=[query],
                    max_results=self.max_recall,
                    threshold=self.recall_threshold,
                )
            )
            return [
                {"concept": item.concept, "content": item.content, "score": item.score}
                for item in result.activations
            ]
        except Exception as e:
            self._muninn_healthy = False
            self._io_errors += 1
            logger.error("Recall failed (error #%d): %s", self._io_errors, e)
            return []

    def _write_engram(self, message: str):
        """Write a message as an engram to the vault."""
        try:
            concept = message.strip()
            self._loop.run_until_complete(
                self._muninn_client.write(
                    vault=self._active_vault,
                    concept=concept,
                    content=message,
                    tags=["ltm_bench", self._session_tag],
                )
            )
            self._write_count += 1
        except Exception as e:
            self._muninn_healthy = False
            self._io_errors += 1
            logger.error("Write failed (error #%d): %s", self._io_errors, e)

    def _write_trace(self, query, engrams, response, dataset=None):
        if self._trace_file is None:
            trace_dir = os.path.join("data", "traces")
            os.makedirs(trace_dir, exist_ok=True)
            self._trace_file = open(
                os.path.join(trace_dir, f"muninn_{self.run_name}_{self._run_id}.jsonl"), "a"
            )
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "engrams": [
                {"score": e.get("score"), "concept": e.get("concept"), "content": e.get("content", "")}
                for e in engrams
            ],
            "response": response,
        }
        self._trace_file.write(json.dumps(entry) + "\n")
        self._trace_file.flush()

    def _generate_response(self, user_message: str, recalled: list[dict]) -> str:
        """Generate a response using the LLM with recalled context."""
        if recalled:
            memory_lines = [f"- {mem['content']}" for mem in recalled]
            memory_context = "\n".join(memory_lines)
        else:
            memory_context = "(no relevant memories found)"

        system_msg = (
            f"{self._system_prompt}\n\n"
            f"Memory context:\n{memory_context}"
        )

        headers = {"Content-Type": "application/json"}
        # Only send OpenRouter API key to verified OpenRouter endpoints
        llm_host = (urlparse(self.llm_url).hostname or "").lower()
        if llm_host in ("openrouter.ai", "www.openrouter.ai"):
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        for attempt in range(5):
            try:
                resp = self._http_client.post(
                    f"{self.llm_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.llm_model,
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_message},
                        ],
                        "max_tokens": 256,
                        "temperature": 0.0,
                    },
                )
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code >= 500:
                    logger.warning("LLM server error: %d", resp.status_code)
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code < 200 or resp.status_code >= 300:
                    logger.warning("LLM request failed: %d %s", resp.status_code, resp.text[:200])
                    return "I'm not sure."
                data = resp.json()
                if "error" in data:
                    logger.warning("LLM error: %s", data["error"])
                    return "I'm not sure."
                content = data["choices"][0]["message"].get("content")
                return content or "Understood."
            except Exception as e:
                logger.error("LLM generation failed: %s", e)
                return "I'm not sure."
        return "I'm not sure."

    def _trigger_dream(self):
        """Trigger dream consolidation on the MuninnDB server."""
        self._ensure_clients()
        try:
            resp = self._http_client.post(
                f"{self.muninn_url}/api/dream",
                json={"force": self.dream_force, "scope": self._active_vault, "dry_run": self.dream_dry_run},
                timeout=120.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                logger.info(
                    "Dream completed: %s, reports=%d",
                    data.get("total_duration", "?"),
                    len(data.get("reports", [])),
                )
            else:
                logger.warning("Dream trigger failed: %d %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("Dream trigger error: %s", e)

    def reset(self):
        """Reset by triggering dream consolidation, then rotating session tag."""
        if self.dream_enabled:
            self._trigger_dream()
        self._vault_counter += 1
        self._session_tag = f"bench_{uuid.uuid4().hex[:12]}"
        self._write_count = 0
        logger.info("Reset: new session %s in vault %s", self._session_tag, self._active_vault)

    def save(self):
        """Persist adapter state to disk."""
        self.save_path.mkdir(parents=True, exist_ok=True)
        state = {
            "session_tag": self._session_tag,
            "write_count": self._write_count,
            "muninn_url": self.muninn_url,
            "llm_url": self.llm_url,
            "llm_model": self.llm_model,
            "run_id": self._run_id,
            "vault_counter": self._vault_counter,
            "muninn_healthy": self._muninn_healthy,
            "io_errors": self._io_errors,
        }
        with open(self.save_path / "muninn_state.json", "w") as f:
            json.dump(state, f)

    def load(self):
        """Restore adapter state from disk."""
        state_file = self.save_path / "muninn_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            self._session_tag = state["session_tag"]
            self._write_count = state["write_count"]
            self._run_id = state.get("run_id", self._run_id)
            self._vault_counter = state.get("vault_counter", self._vault_counter)
            self._muninn_healthy = state.get("muninn_healthy", True)
            self._io_errors = state.get("io_errors", 0)

    def close(self):
        """Deterministic resource cleanup with health summary."""
        if not self._muninn_healthy:
            logger.warning(
                "Session ended with %d Muninn I/O errors — benchmark results may be unreliable",
                self._io_errors,
            )
        try:
            if self._trace_file is not None:
                self._trace_file.close()
                self._trace_file = None
        except Exception as e:
            logger.debug("Error closing trace file: %s", e)
        try:
            if self._muninn_client is not None:
                self._loop.run_until_complete(self._muninn_client.__aexit__(None, None, None))
                self._muninn_client = None
        except Exception as e:
            logger.debug("Error closing Muninn client: %s", e)
        try:
            if self._http_client is not None:
                self._http_client.close()
                self._http_client = None
        except Exception as e:
            logger.debug("Error closing HTTP client: %s", e)
        try:
            if self._loop is not None:
                self._loop.close()
                self._loop = None
        except Exception as e:
            logger.debug("Error closing event loop: %s", e)

    def __del__(self):
        """Clean up async resources."""
        self.close()
