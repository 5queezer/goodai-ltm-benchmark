"""MuninnDB model interface for goodai-ltm-benchmark.

Bridges the benchmark's conversational ChatSession protocol with MuninnDB's
engram write/recall API. Uses a local LLM (OpenAI-compatible) to generate
responses from recalled context.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

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

    # Always local (no cost tracking)
    is_local: bool = True

    # Internal state
    _write_count: int = 0
    _session_tag: str = ""

    # System prompt for the LLM
    _system_prompt: str = (
        "You are a helpful assistant with access to a memory system. "
        "Answer questions using ONLY the provided memory context. "
        "Be extremely brief — one sentence or less. "
        "If the context contains the answer, state it directly. "
        "If not, say you don't remember."
    )

    def __post_init__(self):
        super().__post_init__()
        self._session_tag = f"bench_{int(time.time())}"
        self._loop = asyncio.new_event_loop()
        self._http_client = None  # lazy init
        self._muninn_client = None  # lazy init

    @property
    def name(self):
        return f"MuninnChatSession - {self.llm_model}"

    def _ensure_clients(self):
        """Lazily initialize HTTP clients."""
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=30.0)
        if self._muninn_client is None:
            # Import here to avoid hard dependency at module level
            try:
                import sys
                sys.path.insert(0, "/home/christian/Projects/muninndb/sdk/python")
                from muninn import MuninnClient
                self._muninn_client = MuninnClient(
                    self.muninn_url,
                    token=self.muninn_token or None,
                )
                self._loop.run_until_complete(self._muninn_client.__aenter__())
            except ImportError:
                raise RuntimeError(
                    "MuninnDB Python SDK not found. "
                    "Install from /home/christian/Projects/muninndb/sdk/python"
                )

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

        # 3. Write the user message as an engram
        self._write_engram(user_message)

        # Never return None — the benchmark's flatten_context crashes on it
        return response or "Understood."

    def _recall(self, query: str) -> list[dict]:
        """Recall relevant engrams from the vault."""
        try:
            result = self._loop.run_until_complete(
                self._muninn_client.activate(
                    vault=self.vault,
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
            logger.warning("Recall failed: %s", e)
            return []

    def _write_engram(self, message: str):
        """Write a message as an engram to the vault."""
        try:
            concept = message[:120].strip()
            self._loop.run_until_complete(
                self._muninn_client.write(
                    vault=self.vault,
                    concept=concept,
                    content=message,
                    tags=["ltm_bench", self._session_tag],
                )
            )
            self._write_count += 1
        except Exception as e:
            logger.warning("Write failed: %s", e)

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
        # Support OpenRouter and other providers that need bearer auth
        import os
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
                data = resp.json()
                if "error" in data:
                    code = data["error"].get("code", 0)
                    if code == 429:
                        import time as _time
                        _time.sleep(2 ** attempt)
                        continue
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
                json={"force": True, "scope": self.vault},
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
        self._trigger_dream()
        self._session_tag = f"bench_{int(time.time())}"
        self._write_count = 0
        logger.info("Reset: new session %s in vault %s", self._session_tag, self.vault)

    def save(self):
        """Persist adapter state to disk."""
        self.save_path.mkdir(parents=True, exist_ok=True)
        state = {
            "session_tag": self._session_tag,
            "write_count": self._write_count,
            "muninn_url": self.muninn_url,
            "llm_url": self.llm_url,
            "llm_model": self.llm_model,
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

    def __del__(self):
        """Clean up async resources."""
        try:
            if self._muninn_client is not None:
                self._loop.run_until_complete(self._muninn_client.__aexit__(None, None, None))
            if self._http_client is not None:
                self._http_client.close()
            self._loop.close()
        except Exception:
            pass
