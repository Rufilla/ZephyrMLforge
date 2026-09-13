# _____________________________________________________________________________
#
# @file claude_code.py
# @brief Claude Code agent implementation, driven through the headless CLI
# @version 0.1
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Claude Code agent implementation, driven through the headless CLI.

Uses an existing Claude Code installation instead of an API key, which suits a
workstation that already has one. Two consequences are worth knowing:

- Every call carries the Claude Code harness, roughly 14k tokens of system prompt
  before the pipeline's own. The prompt is stable across iterations, so only the
  first call of a run pays for it; the rest are served from the prompt cache.
- Each call runs in an empty scratch directory, so a CLAUDE.md or the contents of
  whatever directory the pipeline was started from cannot change a proposal.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from typing import Any

from ..core.config import AIAgentConfig
from .base import BaseAgent, ModelProposal

logger = logging.getLogger(__name__)


class ClaudeCodeAgent(BaseAgent):
    """AI agent implementation using a local Claude Code installation."""

    DEFAULT_MODEL = "session default"

    def __init__(self, api_key: str = "", config: AIAgentConfig | None = None):
        super().__init__(api_key, config)

        # Held for the agent's lifetime and removed with it.
        self._scratch_dir = tempfile.TemporaryDirectory(prefix="zephyr-ml-forge-agent-")
        self._session_cost_usd = 0.0

    def propose_model(self, context: dict[str, Any]) -> ModelProposal:
        """Request a proposal, reporting the cost the CLI accounted for."""
        proposal = super().propose_model(context)
        logger.info(f"Claude Code session cost so far: ${self._session_cost_usd:.4f}")
        return proposal

    def _request(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        """Run one headless Claude Code turn and return its text.

        Raises:
            RuntimeError: the CLI is missing, timed out, or reported a failure. Never
                ValueError, which the base class reserves for an unparseable proposal.
        """
        command = [
            self.config.cli_path,
            "--print",
            self._flatten(messages),
            # Replaces Claude Code's own system prompt: the pipeline's instructions
            # should govern, not the coding-assistant persona.
            "--system-prompt",
            system_prompt,
            "--output-format",
            "json",
            # The agent only has to return JSON; it has no business reading files.
            "--allowed-tools",
            "",
        ]

        if self.config.model:
            command += ["--model", self.config.model]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                cwd=self._scratch_dir.name,
                timeout=self.config.cli_timeout_s,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                f"Claude Code CLI not found at '{self.config.cli_path}'. Install it, or "
                "set ai_agent.cli_path to its location."
            ) from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"Claude Code did not answer within {self.config.cli_timeout_s}s"
            ) from error

        if completed.returncode != 0:
            raise RuntimeError(
                f"Claude Code exited {completed.returncode}: "
                f"{(completed.stderr or completed.stdout).strip()[:500]}"
            )

        return self._read_envelope(completed.stdout)

    def _read_envelope(self, stdout: str) -> str:
        """Extract the result text from the CLI's JSON envelope.

        Raises:
            RuntimeError: the envelope was absent, malformed, or reported an error.
        """
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Claude Code returned no JSON envelope: {stdout.strip()[:500]}"
            ) from error

        if envelope.get("is_error"):
            raise RuntimeError(f"Claude Code reported an error: {envelope.get('result')}")

        self._session_cost_usd += float(envelope.get("total_cost_usd") or 0.0)

        result = envelope.get("result")
        if not isinstance(result, str):
            raise RuntimeError(f"Claude Code envelope has no result text: {envelope.keys()}")

        return result

    @staticmethod
    def _flatten(messages: list[dict[str, str]]) -> str:
        """Render the conversation as one prompt.

        Each invocation is a fresh session, so a retry has to carry the exchange that
        provoked it rather than relying on the session remembering.
        """
        if len(messages) == 1:
            return messages[0]["content"]

        return "\n\n".join(
            f"{'User' if message['role'] == 'user' else 'Assistant'}: {message['content']}"
            for message in messages
        )
