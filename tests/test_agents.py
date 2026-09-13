"""Response parsing, retrying and the offline agent."""

from __future__ import annotations

import inspect
import json
import re
from types import SimpleNamespace

import pytest

from pipeline.agents.base import BaseAgent
from pipeline.agents.pseudo import PseudoAgent
from pipeline.core.config import AIAgentConfig

VALID_MODEL_CODE = "import tensorflow as tf\n\ndef create_model(input_shape):\n    return None\n"


def valid_response(**overrides) -> str:
    """A well-formed proposal as the provider would return it."""
    return json.dumps(
        {
            "architecture_description": "Two layer MLP",
            "model_code": VALID_MODEL_CODE,
            "weight_reuse_layers": [],
            "rationale": "Small enough for the arena",
            **overrides,
        }
    )


class ScriptedAgent(BaseAgent):
    """Agent returning a fixed sequence of responses, for testing the retry loop."""

    def __init__(self, responses: list[str], max_retries: int = 2):
        super().__init__("", AIAgentConfig(provider="pseudo", max_retries=max_retries))
        self.responses = list(responses)
        self.requests: list[list[dict[str, str]]] = []

    def _request(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        self.requests.append(list(messages))
        return self.responses.pop(0)


class FailingAgent(BaseAgent):
    """Agent whose provider call always fails."""

    def __init__(self):
        super().__init__("", AIAgentConfig(provider="pseudo"))

    def _request(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        raise RuntimeError("Provider returned 503")


def test_parses_a_well_formed_response():
    proposal = ScriptedAgent([valid_response()]).propose_model({"iteration": 1})

    assert proposal.architecture_description == "Two layer MLP"
    assert proposal.model_code == VALID_MODEL_CODE


def test_keeps_a_literal_newline_escape_inside_the_generated_code():
    # json.loads has already decoded the escapes; unescaping a second time turned a
    # printed '\n' inside the model code into a real line break and broke the file.
    code = 'import tensorflow as tf\n\ndef create_model(input_shape):\n    print("a\\nb")\n'
    proposal = ScriptedAgent([valid_response(model_code=code)]).propose_model({})

    assert '"a\\nb"' in proposal.model_code
    assert proposal.model_code.count("\n") == code.count("\n")


def test_extracts_json_from_a_fenced_block():
    response = f"Here you go:\n```json\n{valid_response()}\n```\nHope that helps."

    assert ScriptedAgent([response]).propose_model({}).architecture_description == "Two layer MLP"


def test_retries_after_an_unparseable_response_and_succeeds():
    agent = ScriptedAgent(["not json at all", valid_response()])

    proposal = agent.propose_model({})

    assert proposal.architecture_description == "Two layer MLP"
    assert len(agent.requests) == 2
    assert "could not be parsed" in agent.requests[1][-1]["content"]


def test_reports_a_provider_failure_as_itself():
    # The retry branch used to reference the response variable that the failed call
    # never assigned, reporting an UnboundLocalError instead of the outage.
    with pytest.raises(RuntimeError, match="Provider returned 503"):
        FailingAgent().propose_model({})


def test_gives_up_after_the_configured_retries():
    agent = ScriptedAgent(["junk", "junk", "junk"], max_retries=2)

    with pytest.raises(RuntimeError, match="No valid proposal after 3 attempts"):
        agent.propose_model({})


def test_rejects_a_response_with_no_create_model_function():
    response = valid_response(model_code="import tensorflow as tf\n")
    agent = ScriptedAgent([response, response, response])

    with pytest.raises(RuntimeError, match="create_model"):
        agent.propose_model({})


def test_rejects_a_response_missing_a_required_field():
    response = json.dumps({"model_code": VALID_MODEL_CODE})
    agent = ScriptedAgent([response, response, response])

    with pytest.raises(RuntimeError, match="Missing required field"):
        agent.propose_model({})


def test_rejects_weight_reuse_layers_of_the_wrong_type():
    response = valid_response(weight_reuse_layers="dense_1")
    agent = ScriptedAgent([response, response, response])

    with pytest.raises(RuntimeError, match="wrong type"):
        agent.propose_model({})


def test_remembers_only_the_successful_exchange():
    agent = ScriptedAgent(["not json at all", valid_response()])

    agent.propose_model({})

    assert len(agent._conversation_history) == 2
    assert agent._conversation_history[1]["content"] == valid_response()


def test_system_prompt_states_the_configured_model_limits():
    agent = ScriptedAgent([valid_response()])
    context = {"constraints": {"model_limits": {"max_parameters": 4096, "max_layers": 4}}}

    prompt = agent._build_system_prompt(context)

    assert "Maximum model parameters: 4096" in prompt
    assert "Maximum layers: 4" in prompt


# Operations the prompt names, against the resolver method that registers each. A
# proposal using an operation the firmware lacks builds, flashes and then faults, so
# the two lists have to agree in both directions.
OPERATION_REGISTRATIONS = {
    "Conv2D": "AddConv2D",
    "DepthwiseConv2D": "AddDepthwiseConv2D",
    "MaxPooling2D": "AddMaxPool2D",
    "AveragePooling2D": "AddAveragePool2D",
    "Reshape": "AddReshape",
    "Add": "AddAdd",
}


def firmware_source() -> str:
    """The firmware the pipeline builds, read from the single source tree."""
    from pipeline.runners.firmware import DEFAULT_APP_SOURCE

    return (DEFAULT_APP_SOURCE / "src" / "main_functions.cpp").read_text()


@pytest.mark.parametrize("operation,registration", sorted(OPERATION_REGISTRATIONS.items()))
def test_the_prompt_agrees_with_the_firmware_resolver(operation, registration):
    agent = ScriptedAgent([valid_response()])
    prompt = agent._build_system_prompt({"task": {"type": "regression"}})

    # Bounded so that 'SeparableConv2D' in the prohibition is not read as 'Conv2D'.
    mention = re.compile(rf"(?<![A-Za-z]){re.escape(operation)}(?![A-Za-z])")

    is_registered = f"resolver.{registration}();" in firmware_source()
    is_offered = bool(mention.search(prompt.split("DO NOT use")[0]))
    is_forbidden = bool(mention.search(prompt.split("DO NOT use")[-1]))

    assert is_offered is is_registered, (
        f"{operation}: offered={is_offered} registered={is_registered}"
    )
    assert is_forbidden is not is_registered


def test_the_resolver_is_large_enough_for_what_the_prompt_offers():
    import re

    source = firmware_source()
    declared = int(re.search(r"MicroMutableOpResolver<(\d+)>", source).group(1))
    registered = len(re.findall(r"resolver\.Add\w+\(", source))

    assert registered <= declared, f"{registered} operations registered in a resolver of {declared}"


def test_pseudo_agent_proposes_a_softmax_output_for_classification():
    proposal = PseudoAgent().propose_model({"iteration": 1, "task": {"type": "classification"}})

    assert "softmax" in proposal.model_code
    assert "sparse_categorical_crossentropy" in proposal.model_code


def test_pseudo_agent_proposes_a_linear_output_for_regression():
    proposal = PseudoAgent().propose_model({"iteration": 1, "task": {"type": "regression"}})

    assert "softmax" not in proposal.model_code
    assert "loss='mse'" in proposal.model_code


def test_pseudo_agent_narrows_the_model_on_later_iterations():
    agent = PseudoAgent()

    first = agent.propose_model({"iteration": 1, "task": {"type": "regression"}})
    third = agent.propose_model({"iteration": 3, "task": {"type": "regression"}})

    assert "Dense(16" in first.model_code
    assert "Dense(8" in third.model_code


class StubMessages:
    """Records the keyword arguments a provider sends to its SDK."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=self.response_text)],
        )


def test_claude_sends_only_arguments_the_installed_sdk_accepts():
    # Mocking _request never exercises the real call, so a parameter the SDK has
    # dropped - temperature, removed from the Messages API - reaches production as a
    # TypeError on the first proposal of a run.
    from anthropic.resources.messages import Messages

    from pipeline.agents.claude import ClaudeAgent

    agent = ClaudeAgent(api_key="test-key", config=AIAgentConfig(provider="claude"))
    stub = StubMessages(valid_response())
    agent.client = SimpleNamespace(messages=stub)

    agent.propose_model({"iteration": 1})

    # Raises TypeError for any argument the installed SDK does not take.
    inspect.signature(Messages.create).bind(None, **stub.kwargs)


def test_chatgpt_sends_only_arguments_the_installed_sdk_accepts():
    from openai.resources.chat.completions import Completions

    from pipeline.agents.chatgpt import ChatGPTAgent

    agent = ChatGPTAgent(api_key="test-key", config=AIAgentConfig(provider="chatgpt"))
    recorded: dict = {}

    def create(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=valid_response()),
                )
            ]
        )

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    agent.propose_model({"iteration": 1})

    inspect.signature(Completions.create).bind(None, **recorded)


def test_claude_passes_the_configured_model_and_token_budget():
    from pipeline.agents.claude import ClaudeAgent

    agent = ClaudeAgent(
        api_key="test-key",
        config=AIAgentConfig(provider="claude", model="claude-opus-5", max_tokens=4096),
    )
    stub = StubMessages(valid_response())
    agent.client = SimpleNamespace(messages=stub)

    agent.propose_model({"iteration": 1})

    assert stub.kwargs["model"] == "claude-opus-5"
    assert stub.kwargs["max_tokens"] == 4096


def test_claude_reads_only_the_text_blocks_of_a_response():
    from pipeline.agents.claude import ClaudeAgent

    agent = ClaudeAgent(api_key="test-key", config=AIAgentConfig(provider="claude"))
    agent.client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                stop_reason="end_turn",
                content=[
                    SimpleNamespace(type="thinking", thinking="deliberating"),
                    SimpleNamespace(type="text", text=valid_response()),
                ],
            )
        )
    )

    assert agent.propose_model({}).architecture_description == "Two layer MLP"


def stop_response(reason: str = "Within 3% of the noise floor; no proposal would do better."):
    """A response that ends the run instead of proposing a model."""
    return json.dumps({"stop": True, "stop_reason": reason})


def test_a_stop_response_carries_no_model():
    proposal = ScriptedAgent([stop_response()]).propose_model({"iteration": 6})

    assert proposal.should_stop is True
    assert "noise floor" in proposal.stop_reason
    assert proposal.model_code == ""


def test_a_stop_response_needs_no_architecture_or_code():
    # The usual required fields do not apply: there is nothing to build.
    agent = ScriptedAgent([json.dumps({"stop": True, "stop_reason": "done"})])

    assert agent.propose_model({}).should_stop is True


def test_a_stop_response_must_say_why():
    response = json.dumps({"stop": True})
    agent = ScriptedAgent([response, response, response])

    with pytest.raises(RuntimeError, match="stop_reason"):
        agent.propose_model({})


def test_an_ordinary_proposal_does_not_stop():
    assert ScriptedAgent([valid_response()]).propose_model({}).should_stop is False


def test_the_option_to_stop_is_offered_only_when_the_run_allows_it():
    agent = ScriptedAgent([valid_response()])

    assert "ENDING THE RUN" in agent._build_system_prompt({"may_stop": True})
    assert "ENDING THE RUN" not in agent._build_system_prompt({"may_stop": False})
    assert "ENDING THE RUN" not in agent._build_system_prompt({})


def test_the_prompt_quotes_the_irreducible_floor():
    # Without it the agent cannot tell a solved task from a stalled search.
    agent = ScriptedAgent([valid_response()])
    context = {
        "constraints": {"accuracy": {"metric": "mse", "irreducible_floor": 0.010933}},
        "task": {"type": "regression"},
    }

    assert "Irreducible floor: 0.010933" in agent._build_user_prompt(context)


def test_the_prompt_omits_the_floor_when_there_is_none():
    agent = ScriptedAgent([valid_response()])
    context = {
        "constraints": {"accuracy": {"metric": "accuracy"}},
        "task": {"type": "classification"},
    }

    assert "Irreducible floor" not in agent._build_user_prompt(context)


def test_claude_names_a_reply_cut_off_by_the_token_limit():
    # Thinking is drawn from max_tokens, so a tight budget truncates the answer and the
    # parse error that follows blames the model for output we cut off ourselves.
    from pipeline.agents.claude import ClaudeAgent

    agent = ClaudeAgent(api_key="test-key", config=AIAgentConfig(provider="claude"))
    agent.client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                stop_reason="max_tokens",
                content=[SimpleNamespace(type="text", text='{"architecture_desc')],
            )
        )
    )

    with pytest.raises(RuntimeError, match="cut off at the .* limit"):
        agent.propose_model({})


def test_claude_names_a_reply_that_was_all_thinking():
    from pipeline.agents.claude import ClaudeAgent

    agent = ClaudeAgent(api_key="test-key", config=AIAgentConfig(provider="claude"))
    agent.client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="thinking", thinking="deliberating")],
            )
        )
    )

    with pytest.raises(RuntimeError, match="returned no text"):
        agent.propose_model({})


def test_chatgpt_names_a_reply_cut_off_by_the_token_limit():
    from pipeline.agents.chatgpt import ChatGPTAgent

    agent = ChatGPTAgent(api_key="test-key", config=AIAgentConfig(provider="chatgpt"))
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="length",
                            message=SimpleNamespace(content='{"arch'),
                        )
                    ]
                )
            )
        )
    )

    with pytest.raises(RuntimeError, match="cut off at the .* limit"):
        agent.propose_model({})


def test_the_prompt_warns_when_the_previous_model_was_under_trained():
    # Without this the agent reads a truncated fit as a poor architecture and abandons
    # a direction that was never given a chance.
    agent = ScriptedAgent([valid_response()])
    context = {
        "previous_iteration": {
            "status": "success",
            "metrics": {"accuracy_value": 0.762},
            "weight_reuse": {"failures": []},
            "epochs_trained": 60,
            "is_under_trained": True,
        }
    }

    prompt = agent._build_user_prompt(context)

    assert "under-trained" in prompt
    assert "60 epochs" in prompt


def test_the_prompt_stays_quiet_about_a_converged_fit():
    agent = ScriptedAgent([valid_response()])
    context = {
        "previous_iteration": {
            "status": "success",
            "metrics": {"accuracy_value": 0.872},
            "weight_reuse": {"failures": []},
            "epochs_trained": 23,
            "is_under_trained": False,
        }
    }

    assert "under-trained" not in agent._build_user_prompt(context)


def test_the_prompt_carries_the_error_bar_on_the_previous_measurement():
    # A metric quoted to six decimal places reads as exact, and the agent then treats a
    # difference in the fourth as something it achieved.
    agent = ScriptedAgent([valid_response()])
    context = {
        "previous_iteration": {
            "status": "success",
            "metrics": {
                "accuracy_value": 0.872,
                "accuracy_standard_error": 0.0106,
                "accuracy_sample_count": 1000,
            },
            "weight_reuse": {"failures": []},
        }
    }

    prompt = agent._build_user_prompt(context)

    assert "0.010600" in prompt
    assert "1000 validation samples" in prompt


def test_the_prompt_tells_the_agent_when_its_best_two_results_are_not_separated():
    agent = ScriptedAgent([valid_response()])
    context = {
        "best_so_far": {
            "iteration": 4,
            "accuracy": 0.872,
            "standard_error": 0.0106,
            "sample_count": 1000,
            "latency_ms": 13.1,
        },
        "unresolved_ranking": "Iteration 4 and iteration 1 differ by less than the standard error",
    }

    prompt = agent._build_user_prompt(context)

    assert "± 0.010600" in prompt
    assert "differ by less than the standard error" in prompt
    assert "Chasing a difference that small" in prompt
