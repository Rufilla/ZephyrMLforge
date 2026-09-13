# _____________________________________________________________________________
#
# @file claude.py
# @brief Claude (Anthropic) agent implementation
# @version 0.2
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Claude (Anthropic) agent implementation.
"""

from __future__ import annotations

import logging

from anthropic import Anthropic

from ..core.config import AIAgentConfig
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ClaudeAgent(BaseAgent):
    """AI agent implementation using Anthropic's Claude API."""

    DEFAULT_MODEL = "claude-sonnet-5"

    def __init__(self, api_key: str, config: AIAgentConfig | None = None):
        super().__init__(api_key, config)
        self.client = Anthropic(api_key=api_key)

    def _request(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        """Send one request to the Messages API and return the text blocks.

        Raises:
            RuntimeError: the API call failed.
        """
        try:
            # No temperature: sampling parameters were removed from the Messages API for
            # the current model family, and the SDK rejects the argument outright.
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=messages,
            )
        except Exception as error:
            logger.exception("Claude API error")
            raise RuntimeError(f"Failed to get model proposal from Claude: {error}") from error

        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Claude's reply was cut off at the {self.config.max_tokens}-token limit. "
                "Raise ai_agent.max_tokens; thinking is drawn from the same budget."
            )

        # A response may carry non-text blocks; concatenating only the text ones keeps
        # this working if the model or its settings start producing them.
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

        if not text.strip():
            # Every token went to thinking, which no retry at this budget will change.
            raise RuntimeError(
                f"Claude returned no text, only "
                f"{[getattr(b, 'type', '?') for b in response.content]}. "
                "Raise ai_agent.max_tokens so the answer fits alongside the thinking."
            )

        return text
