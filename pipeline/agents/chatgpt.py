# _____________________________________________________________________________
#
# @file chatgpt.py
# @brief ChatGPT (OpenAI) agent implementation
# @version 0.2
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
ChatGPT (OpenAI) agent implementation.
"""

from __future__ import annotations

import logging

from openai import OpenAI

from ..core.config import AIAgentConfig
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ChatGPTAgent(BaseAgent):
    """AI agent implementation using OpenAI's ChatGPT API."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str, config: AIAgentConfig | None = None):
        super().__init__(api_key, config)
        self.client = OpenAI(api_key=api_key)

    def _request(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        """Send one chat completion request and return its content.

        Raises:
            RuntimeError: the API call failed.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, *messages],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as error:
            logger.exception("ChatGPT API error")
            raise RuntimeError(f"Failed to get model proposal from ChatGPT: {error}") from error

        choice = response.choices[0]

        if choice.finish_reason == "length":
            raise RuntimeError(
                f"ChatGPT's reply was cut off at the {self.config.max_tokens}-token "
                "limit. Raise ai_agent.max_tokens."
            )

        return choice.message.content or ""
