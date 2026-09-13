"""The Claude Code provider, against a stubbed CLI."""

from __future__ import annotations

import json
import subprocess

import pytest

from pipeline.agents.claude_code import ClaudeCodeAgent
from pipeline.core.config import AIAgentConfig, AIProvider
from tests.test_agents import VALID_MODEL_CODE, valid_response


def envelope(result: str, is_error: bool = False, cost: float = 0.0125) -> str:
    """One CLI JSON envelope, shaped as `claude -p --output-format json` returns it."""
    return json.dumps(
        {
            "type": "result",
            "subtype": "error" if is_error else "success",
            "is_error": is_error,
            "result": result,
            "session_id": "051dec63-22e4-49aa-b9c2-2c5ad679084c",
            "total_cost_usd": cost,
            "usage": {"input_tokens": 2, "output_tokens": 4},
        }
    )


class StubbedCLI:
    """Stands in for subprocess.run, recording the command it was given."""

    def __init__(self, stdout: str = "", returncode: int = 0, raises: Exception | None = None):
        self.stdout = stdout
        self.returncode = returncode
        self.raises = raises
        self.commands: list[list[str]] = []
        self.kwargs: list[dict] = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        self.kwargs.append(kwargs)

        if self.raises is not None:
            raise self.raises

        return subprocess.CompletedProcess(
            args=command, returncode=self.returncode, stdout=self.stdout, stderr=""
        )


@pytest.fixture
def stub_cli(monkeypatch):
    """Install a stubbed CLI and hand it back for inspection."""

    def install(stdout: str = "", returncode: int = 0, raises: Exception | None = None):
        stub = StubbedCLI(stdout=stdout, returncode=returncode, raises=raises)
        monkeypatch.setattr(subprocess, "run", stub)
        return stub

    return install


def make_agent(**overrides) -> ClaudeCodeAgent:
    """Build the agent with a configuration that needs no API key."""
    return ClaudeCodeAgent(config=AIAgentConfig(provider="claude_code", **overrides))


def test_returns_the_proposal_from_the_envelope(stub_cli):
    stub_cli(stdout=envelope(valid_response()))

    proposal = make_agent().propose_model({"iteration": 1})

    assert proposal.architecture_description == "Two layer MLP"
    assert proposal.model_code == VALID_MODEL_CODE


def test_needs_no_api_key():
    assert AIProvider.CLAUDE_CODE.needs_api_key is False
    assert AIAgentConfig(provider="claude_code").get_api_key() == ""


def test_runs_headless_with_the_pipeline_system_prompt(stub_cli):
    stub = stub_cli(stdout=envelope(valid_response()))

    make_agent().propose_model({"iteration": 1, "task": {"type": "regression"}})

    command = stub.commands[0]
    assert command[0] == "claude"
    assert "--print" in command
    assert command[command.index("--output-format") + 1] == "json"
    # The pipeline's instructions must replace Claude Code's own persona.
    assert "TFLITE MICRO OPERATION CONSTRAINTS" in command[command.index("--system-prompt") + 1]


def test_denies_the_agent_every_tool(stub_cli):
    stub = stub_cli(stdout=envelope(valid_response()))

    make_agent().propose_model({})

    command = stub.commands[0]
    assert command[command.index("--allowed-tools") + 1] == ""


def test_runs_outside_the_project_directory(stub_cli):
    # A CLAUDE.md in the working directory would otherwise reach the proposal.
    stub = stub_cli(stdout=envelope(valid_response()))

    agent = make_agent()
    agent.propose_model({})

    assert stub.kwargs[0]["cwd"] == agent._scratch_dir.name
    assert stub.kwargs[0]["stdin"] == subprocess.DEVNULL


def test_passes_a_configured_model_through(stub_cli):
    stub = stub_cli(stdout=envelope(valid_response()))

    make_agent(model="claude-opus-5").propose_model({})

    command = stub.commands[0]
    assert command[command.index("--model") + 1] == "claude-opus-5"


def test_omits_the_model_flag_when_unset(stub_cli):
    stub = stub_cli(stdout=envelope(valid_response()))

    make_agent().propose_model({})

    assert "--model" not in stub.commands[0]


def test_retry_carries_the_rejected_exchange(stub_cli):
    # Each invocation is a fresh session, so the retry prompt alone would not tell the
    # agent what it got wrong.
    stub = stub_cli(stdout=envelope("not json at all"))

    with pytest.raises(RuntimeError, match="No valid proposal"):
        make_agent(max_retries=1).propose_model({})

    retry_prompt = stub.commands[1][2]
    assert "Assistant: not json at all" in retry_prompt
    assert "could not be parsed" in retry_prompt


def test_reports_a_missing_cli_with_the_path_it_tried(stub_cli):
    stub_cli(raises=FileNotFoundError())

    with pytest.raises(RuntimeError, match="not found at 'claude-absent'"):
        make_agent(cli_path="claude-absent").propose_model({})


def test_reports_a_timeout_rather_than_hanging(stub_cli):
    stub_cli(raises=subprocess.TimeoutExpired(cmd="claude", timeout=300))

    with pytest.raises(RuntimeError, match="did not answer within 300s"):
        make_agent().propose_model({})


def test_reports_a_non_zero_exit(stub_cli):
    stub_cli(stdout="", returncode=1)

    with pytest.raises(RuntimeError, match="Claude Code exited 1"):
        make_agent().propose_model({})


def test_reports_an_error_envelope(stub_cli):
    stub_cli(stdout=envelope("Credit balance too low", is_error=True))

    with pytest.raises(RuntimeError, match="Credit balance too low"):
        make_agent().propose_model({})


def test_reports_output_that_is_not_an_envelope(stub_cli):
    # A CLI that printed a warning instead of JSON must not look like a bad proposal.
    stub_cli(stdout="Warning: no stdin data received in 3s\n")

    with pytest.raises(RuntimeError, match="no JSON envelope"):
        make_agent().propose_model({})


def test_accumulates_the_reported_cost(stub_cli):
    stub_cli(stdout=envelope(valid_response(), cost=0.0975))

    agent = make_agent()
    agent.propose_model({})
    agent.propose_model({})

    assert agent._session_cost_usd == pytest.approx(0.195)
