# _____________________________________________________________________________
#
# @file factory.py
# @brief Factory for creating AI agent instances
# @version 0.2
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Factory for creating AI agent instances.
"""

from __future__ import annotations

from ..core.config import AIAgentConfig, AIProvider
from .base import BaseAgent
from .chatgpt import ChatGPTAgent
from .claude import ClaudeAgent
from .claude_code import ClaudeCodeAgent
from .pseudo import PseudoAgent

AGENTS_BY_PROVIDER = {
    AIProvider.CHATGPT: ChatGPTAgent,
    AIProvider.CLAUDE: ClaudeAgent,
    AIProvider.CLAUDE_CODE: ClaudeCodeAgent,
    AIProvider.PSEUDO: PseudoAgent,
}


def create_agent(config: AIAgentConfig) -> BaseAgent:
    """
    Create an AI agent instance from configuration.

    Raises:
        ValueError: the provider is not supported, or its API key is unset.
    """
    agent_class = AGENTS_BY_PROVIDER.get(config.provider)
    if agent_class is None:
        raise ValueError(f"Unsupported AI provider: {config.provider}")

    return agent_class(api_key=config.get_api_key(), config=config)
