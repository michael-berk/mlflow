import asyncio
import json
from unittest import mock
import openai
import pytest

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.exceptions.function_exceptions import FunctionExecutionException
from semantic_kernel.exceptions.kernel_exceptions import KernelInvokeException
from semantic_kernel.functions import FunctionResult, KernelArguments
from semantic_kernel.utils.telemetry.model_diagnostics.decorators import (
    are_sensitive_events_enabled,
)

import mlflow.semantic_kernel
from mlflow.entities import SpanType
from mlflow.semantic_kernel.autolog import SemanticKernelSpanProcessor
from mlflow.tracing.provider import _get_tracer

from tests.tracing.helper import get_traces


async def _create_and_invoke_kernel_simple(mock_openai):
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)

    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            service_id="chat-gpt",
            ai_model_id="gpt-4o-mini",
            async_client=openai_client,
        )
    )
    return await kernel.invoke_prompt("Is sushi the best food ever?")


async def _create_and_invoke_kernel_complex(mock_openai):
    # Set up kernel + OpenAI service
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)
    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            service_id="chat-gpt",
            ai_model_id="gpt-4o-mini",
            async_client=openai_client,
        )
    )

    settings = kernel.get_prompt_execution_settings_from_service_id("chat-gpt")
    settings.max_tokens = 100
    settings.temperature = 0.7
    settings.top_p = 0.8

    chat_function = kernel.add_function(
        plugin_name="ChatBot",
        function_name="Chat",
        prompt="{{$chat_history}}{{$user_input}}",
        template_format="semantic-kernel",
        prompt_execution_settings=settings,
    )

    # Prepare input
    chat_history = ChatHistory(
        system_message="You are a chat bot named Mosscap, dedicated to figuring out what people need."
    )
    chat_history.add_user_message("Hi there, who are you?")
    chat_history.add_assistant_message(
        "I am Mosscap, a chat bot. I'm trying to figure out what people need."
    )
    user_input = "I want to find a hotel in Seattle with free wifi and a pool."

    return await kernel.invoke(
        chat_function,
        KernelArguments(
            user_input=user_input,
            chat_history=chat_history,
        ),
    )


async def _create_and_invoke_chat_agent(mock_openai):
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)
    service = OpenAIChatCompletion(
        service_id="chat-gpt",
        ai_model_id="gpt-4o-mini",
        async_client=openai_client,
    )
    agent = ChatCompletionAgent(
        service=service,
        name="sushi_agent",
        instructions="You are a master at all things sushi. But, you are not very smart.",
    )
    return await agent.get_response(messages="How do I make sushi?")


def test_override_of_span_processor():
    mlflow.semantic_kernel.autolog()

    from mlflow.tracing.provider import _MLFLOW_TRACER_PROVIDER

    span_processor = _MLFLOW_TRACER_PROVIDER._active_span_processor._span_processors[0]
    assert isinstance(span_processor, SemanticKernelSpanProcessor)

    # Ensure the tracer is not reset on _get_tracer invocation
    _ = _get_tracer(__name__)
    span_processor = _MLFLOW_TRACER_PROVIDER._active_span_processor._span_processors[0]
    assert isinstance(span_processor, SemanticKernelSpanProcessor)

    # Assert genai attributes will be logged
    assert are_sensitive_events_enabled()


@pytest.mark.asyncio
async def test_sk_invoke_simple(mock_openai):
    mlflow.semantic_kernel.autolog()
    _ = await _create_and_invoke_kernel_simple(mock_openai)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.request_id
    assert trace.info.experiment_id == "0"
    assert trace.info.timestamp_ms > 0
    assert isinstance(trace.data.spans, list) and len(trace.data.spans) >= 2

    root_span = next((s for s in trace.data.spans if s.parent_id is None), None)
    child_span = next((s for s in trace.data.spans if s.parent_id == root_span.span_id), None)

    assert root_span is not None and "mlflow.traceRequestId" in root_span.attributes
    assert child_span is not None and "gen_ai.operation.name" in child_span.attributes

    inputs = child_span.attributes["mlflow.spanInputs"]
    if isinstance(inputs, str):
        inputs = json.loads(inputs)
    assert isinstance(inputs, dict)

    outputs = child_span.attributes["mlflow.spanOutputs"]
    if isinstance(outputs, str):
        outputs = json.loads(outputs)
    assert isinstance(outputs, list)


@pytest.mark.asyncio
async def test_sk_invoke_complex(mock_openai):
    mlflow.semantic_kernel.autolog()
    _ = await _create_and_invoke_kernel_complex(mock_openai)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    spans = trace.data.spans
    assert len(spans) == 2

    root_span = next(s for s in spans if s.parent_id is None)
    child_span = next(s for s in spans if s.parent_id == root_span.span_id)

    root_dict = root_span.to_dict()
    child_dict = child_span.to_dict()

    # Validate root span
    assert root_dict["name"] == "ChatBot-Chat"
    assert root_dict["attributes"]["mlflow.traceRequestId"].replace('"', '') == trace.info.request_id
    assert root_dict["attributes"]["mlflow.spanType"] == '"UNKNOWN"'

    # Validate child span
    assert child_dict["name"] == "chat.completions gpt-4o-mini"
    assert child_dict["parent_id"] == root_span.span_id
    attributes = child_dict["attributes"]

    assert attributes["mlflow.spanType"] == "CHAT_MODEL"
    assert attributes["gen_ai.operation.name"] == "chat.completions"
    assert attributes["gen_ai.system"] == "openai"
    assert attributes["gen_ai.request.model"] == "gpt-4o-mini"
    assert attributes["gen_ai.response.id"] == "chatcmpl-123"
    assert attributes["gen_ai.response.finish_reason"] == "FinishReason.STOP"
    assert attributes["gen_ai.usage.input_tokens"] == 9
    assert attributes["gen_ai.usage.output_tokens"] == 12

    # Validate the input prompt was captured
    span_inputs = json.loads(attributes["mlflow.spanInputs"])
    assert isinstance(span_inputs, dict)
    assert "messages" in span_inputs
    assert any("I want to find a hotel in Seattle with free wifi and a pool." in m["content"]
            for m in span_inputs["messages"])

@pytest.mark.asyncio
async def test_sk_invoke_agent(mock_openai):
    mlflow.semantic_kernel.autolog()
    _ = await _create_and_invoke_chat_agent(mock_openai)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    spans = trace.data.spans
    assert len(spans) == 3

    root_span = next(s for s in spans if s.parent_id is None)
    child_span = next(s for s in spans if s.parent_id == root_span.span_id)
    grandchild_span = next(s for s in spans if s.parent_id == child_span.span_id)

    root = root_span.to_dict()
    assert root["name"] == "invoke_agent sushi_agent"
    assert root["attributes"]["mlflow.spanType"] == '"UNKNOWN"'
    assert root["attributes"]["gen_ai.operation.name"] == "invoke_agent"
    assert root["attributes"]["gen_ai.agent.name"] == "sushi_agent"

    child = child_span.to_dict()
    assert child["name"] == "AutoFunctionInvocationLoop"
    assert child["attributes"]["mlflow.spanType"] == '"UNKNOWN"'
    assert "sk.available_functions" in child["attributes"]

    grandchild = grandchild_span.to_dict()
    assert grandchild["name"].startswith("chat.completions")
    assert grandchild["attributes"]["mlflow.spanType"] == "CHAT_MODEL"
    assert grandchild["attributes"]["gen_ai.request.model"] == "gpt-4o-mini"
    assert "How do I make sushi?" in grandchild["attributes"]["mlflow.spanInputs"]
    assert grandchild["attributes"]["gen_ai.response.finish_reason"] == "FinishReason.STOP"



@pytest.mark.asyncio
async def test_sk_autolog_trace_on_exception(mock_openai):
    mlflow.semantic_kernel.autolog()
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)

    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            service_id="chat-gpt",
            ai_model_id="gpt-4o-mini",
            async_client=openai_client,
        )
    )

    with mock.patch.object(
        openai_client.chat.completions, "create", side_effect=RuntimeError("thiswillfail")
    ):
        with pytest.raises(KernelInvokeException) as exc_info:
            await kernel.invoke_prompt("Hello?")

        assert isinstance(exc_info.value.__cause__, FunctionExecutionException)
        assert "thiswillfail" in str(exc_info.value.__cause__)

    traces = get_traces()
    assert traces, "No traces recorded"
    trace = traces[0]

    assert trace.info.status == "ERROR"
    error_span = next(s for s in trace.data.spans if s.status.status_code == "ERROR")
    exception_event = next((e for e in error_span.events if e.name == "exception"), None)
    assert exception_event
    assert "thiswillfail" in exception_event.attributes["exception.message"]


@pytest.mark.asyncio
async def test_tracing_autolog_with_active_span(mock_openai):
    mlflow.semantic_kernel.autolog()

    with mlflow.start_span("parent") as span:
        response = await _create_and_invoke_kernel_simple(mock_openai)

    assert isinstance(response, FunctionResult)

    traces = get_traces()
    assert len(traces) == 1

    trace = traces[0]
    spans = trace.data.spans
    assert len(spans) == 3

    assert trace.info.request_id is not None
    assert trace.info.status == "OK"
    assert trace.info.tags["mlflow.traceName"] == "parent"

    parent = trace.data.spans[0]
    assert parent.name == "parent"
    assert parent.parent_id is None
    assert parent.attributes["mlflow.spanType"] == SpanType.UNKNOWN

    child = trace.data.spans[1]
    assert child.parent_id == parent.span_id
    assert parent.attributes["mlflow.spanType"] == SpanType.UNKNOWN

    grandchild = trace.data.spans[2]
    assert grandchild.name == "chat.completions gpt-4o-mini"
    assert grandchild.parent_id == child.span_id
    attributes = grandchild.to_dict()["attributes"]
    assert attributes["mlflow.spanType"] == "CHAT_MODEL"
    assert attributes["gen_ai.operation.name"] == "chat.completions"
    assert attributes["gen_ai.system"] == "openai"
    assert attributes["gen_ai.request.model"] == "gpt-4o-mini"
    assert (
        json.loads(attributes["mlflow.spanInputs"])["messages"][0]["content"] == "Is sushi the best food ever?"
    )


@pytest.mark.asyncio
async def test_tracing_attribution_with_threaded_calls(mock_openai):
    mlflow.semantic_kernel.autolog() 

    n = 2 
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)

    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            service_id="chat-gpt",
            ai_model_id="gpt-4o-mini",
            async_client=openai_client,
        )
    )

    async def call(prompt: str):
        return await kernel.invoke_prompt(prompt)

    prompts = [f"What is this number: {i}" for i in range(n)]
    _ = await asyncio.gather(*(call(p) for p in prompts))

    traces = get_traces()
    assert len(traces) == n

    for trace in traces:
        spans = trace.data.spans
        assert len(spans) == 2

        root_span = next(s for s in spans if s.parent_id is None)
        child_span = next(s for s in spans if s.parent_id == root_span.span_id)

        root_dict = root_span.to_dict()
        child_dict = child_span.to_dict()

        assert root_dict["attributes"]["mlflow.spanType"] == '"UNKNOWN"'
        assert child_dict["name"].startswith("chat.completions")
        assert child_dict["attributes"]["mlflow.spanType"] == "CHAT_MODEL"
        input_str = child_dict["attributes"].get("mlflow.spanInputs", "")
        assert any(prompt in input_str for prompt in prompts)