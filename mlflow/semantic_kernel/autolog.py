import json
import logging
import os
from types import MappingProxyType
from typing import Any, Optional
from pydantic import BaseModel

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import (
    NoOpTracerProvider,
    ProxyTracerProvider,
    get_current_span,
    get_tracer_provider,
    set_tracer_provider,
)
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.streaming_content_mixin import StreamingContentMixin
from semantic_kernel.utils.telemetry.model_diagnostics import (
    gen_ai_attributes as model_gen_ai_attributes,
)
from semantic_kernel.utils.telemetry.model_diagnostics.decorators import (
    CHAT_COMPLETION_OPERATION,
    CHAT_STREAMING_COMPLETION_OPERATION,
    TEXT_COMPLETION_OPERATION,
    TEXT_STREAMING_COMPLETION_OPERATION,
    are_sensitive_events_enabled,
)

from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.entities.trace_status import TraceStatus
from mlflow.tracing.constant import (
    SpanAttributeKey,
    TokenUsageKey,
)
from mlflow.tracing.fluent import start_span  # local import to avoid circular deps
from mlflow.tracing.fluent import start_span_no_context
from mlflow.tracing.provider import detach_span_from_context, set_span_in_context
from mlflow.tracing.trace_manager import InMemoryTraceManager

from mlflow.types.chat import (
    ChatMessage,
    Function,
    TextContentPart,
    ToolCall,
)

_logger = logging.getLogger(__name__)

# NB: Use global variable instead of the instance variable of the processor, because sometimes
# multiple span processor instances can be created and we need to share the same map.
_OTEL_SPAN_ID_TO_MLFLOW_SPAN_AND_TOKEN = {}


def _set_logging_env_variables():
    # NB: these environment variables are required to enable the telemetry for
    # genai fields in Semantic Kernel, which are currently marked as experimental.
    # https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/observability/telemetry-with-console
    os.environ["SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS"] = "true"
    os.environ["SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE"] = "true"

    # Reset the diagnostics module which is initialized at import time
    from semantic_kernel.utils.telemetry.model_diagnostics.decorators import (
        MODEL_DIAGNOSTICS_SETTINGS,
    )

    MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics = (
        os.getenv("SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS", "").lower() == "true"
    )

    MODEL_DIAGNOSTICS_SETTINGS.enable_otel_diagnostics_sensitive = (
        os.getenv("SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE", "").lower()
        == "true"
    )


def setup_semantic_kernel_tracing():
    _set_logging_env_variables()

    # NB: This logic has a known issue that it does not work when Semantic Kernel program is
    # executed before calling this setup is called. This is because Semantic Kernel caches the
    # tracer instance in each module (ref:https://github.com/microsoft/semantic-kernel/blob/6ecf2b9c2c893dc6da97abeb5962dfc49bed062d/python/semantic_kernel/functions/kernel_function.py#L46),
    # which prevent us from updating the span processor setup for the tracer.
    # Therefore, `mlflow.semantic_kernel.autolog()` should always be called before running the
    # Semantic Kernel program.
    provider = get_tracer_provider()
    sk_processor = SemanticKernelSpanProcessor()
    if isinstance(provider, (NoOpTracerProvider, ProxyTracerProvider)):
        new_provider = SDKTracerProvider()
        new_provider.add_span_processor(sk_processor)
        set_tracer_provider(new_provider)
    else:
        if not any(
            isinstance(p, SemanticKernelSpanProcessor)
            for p in provider._active_span_processor._span_processors
        ):
            provider.add_span_processor(sk_processor)


class DummySpanExporter:
    # NB: Dummy NoOp exporter that does nothing, because OTel span processor requires an exporter
    def on_end(self, span: OTelReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass


class SemanticKernelSpanProcessor(SimpleSpanProcessor):
    def __init__(self):
        self.span_exporter = DummySpanExporter()

    def on_start(self, span: OTelSpan, parent_context: Optional[Context] = None):
        otel_span_id = span.get_span_context().span_id
        parent_span_id = span.parent.span_id if span.parent else None
        parent_st = _OTEL_SPAN_ID_TO_MLFLOW_SPAN_AND_TOKEN.get(parent_span_id)
        parent_span = parent_st[0] if parent_st else None

        # Convert MappingProxyType (immutable mapping) to dict to satisfy MLflow span API
        attrs = span.attributes
        if isinstance(attrs, MappingProxyType):
            attrs = dict(attrs)

        mlflow_span = start_span_no_context(
            name=span.name,
            parent_span=parent_span,
            attributes=attrs,
        )
        token = set_span_in_context(mlflow_span)
        _OTEL_SPAN_ID_TO_MLFLOW_SPAN_AND_TOKEN[otel_span_id] = (mlflow_span, token)

    def on_end(self, span: OTelReadableSpan) -> None:
        st = _OTEL_SPAN_ID_TO_MLFLOW_SPAN_AND_TOKEN.pop(span.get_span_context().span_id, None)
        if st is None:
            _logger.debug("Span not found in the map. Skipping end.")
            return

        mlflow_span, token = st
        attributes = (
            dict(span.attributes)
            if isinstance(span.attributes, MappingProxyType)
            else span.attributes
        )
        mlflow_span.set_attributes(attributes)
        _set_token_usage(mlflow_span, attributes)

        if mlflow_span.span_type or mlflow_span.span_type == SpanType.UNKNOWN:
            mlflow_span.set_span_type(_get_span_type(span))

        detach_span_from_context(token)
        mlflow_span.end()


def _get_live_span_from_otel_span_id(otel_span_id: str) -> Optional[LiveSpan]:
    if span_and_token := _OTEL_SPAN_ID_TO_MLFLOW_SPAN_AND_TOKEN.get(otel_span_id):
        return span_and_token[0]
    else:
        _logger.warning(
            f"Live span not found for OTel span ID: {otel_span_id}. "
            "Cannot map OTel span ID to MLflow span ID, so we will skip registering "
            "additional attributes. "
        )
        return None


def _serialize_semantic_kernel_result(result: Any) -> str:
    """Convert Semantic Kernel result to JSON-serializable format."""
    try:
        if hasattr(result, "value") and result.value is not None:
            if isinstance(result.value, list):
                return json.dumps(
                    [
                        item.to_dict() if hasattr(item, "to_dict") else str(item)
                        for item in result.value
                    ]
                )
            elif hasattr(result.value, "to_dict"):
                return json.dumps(result.value.to_dict())
            else:
                return json.dumps(str(result.value))
        elif hasattr(result, "to_dict"):
            return json.dumps(result.to_dict())
        else:
            return json.dumps(str(result))
    except Exception as e:
        _logger.warning(f"Failed to serialize result: {e}")
        return json.dumps(str(result))


def _get_span_type(span: OTelSpan) -> str:
    span_type = None

    if hasattr(span, "attributes") and (
        operation := span.attributes.get(model_gen_ai_attributes.OPERATION)
    ):
        span_map = {
            CHAT_COMPLETION_OPERATION: SpanType.CHAT_MODEL,
            CHAT_STREAMING_COMPLETION_OPERATION: SpanType.CHAT_MODEL,
            TEXT_COMPLETION_OPERATION: SpanType.LLM,
            TEXT_STREAMING_COMPLETION_OPERATION: SpanType.LLM,
            "execute_tool": SpanType.TOOL,
        }
        span_type = span_map.get(operation)

    return span_type or SpanType.UNKNOWN


def _set_token_usage(mlflow_span: LiveSpan, sk_attributes: dict[str, Any]) -> None:
    if value := sk_attributes.get(model_gen_ai_attributes.INPUT_TOKENS):
        mlflow_span.set_attribute(TokenUsageKey.INPUT_TOKENS, value)
    if value := sk_attributes.get(model_gen_ai_attributes.OUTPUT_TOKENS):
        mlflow_span.set_attribute(TokenUsageKey.OUTPUT_TOKENS, value)

    if (input_tokens := sk_attributes.get(model_gen_ai_attributes.INPUT_TOKENS)) and (
        output_tokens := sk_attributes.get(model_gen_ai_attributes.OUTPUT_TOKENS)
    ):
        mlflow_span.set_attribute(TokenUsageKey.TOTAL_TOKENS, input_tokens + output_tokens)


def _semantic_kernel_chat_completion_input_wrapper(original, *args, **kwargs) -> None:
    # NB: Semantic Kernel logs chat completions, so we need to extract it and add it to the span.
    try:
        prompt = args[1] if len(args) > 1 else kwargs.get("prompt")

        if isinstance(prompt, ChatHistory):
            prompt_value = [msg.to_dict() for msg in prompt.messages]
        elif not isinstance(prompt, list):
            prompt_value = [prompt]
        else:
            prompt_value = prompt

        prompt_value_with_message = {"messages": prompt_value}

        otel_span_id = get_current_span().get_span_context().span_id

        if mlflow_span := _get_live_span_from_otel_span_id(otel_span_id):
            mlflow_span.set_span_type(SpanType.CHAT_MODEL)
            mlflow_span.set_inputs(prompt_value_with_message)
        else:
            _logger.debug(
                "Span is not found or recording. Skipping registering chat "
                f"completion attributes to {SpanAttributeKey.INPUTS}."
            )

    except Exception as e:
        _logger.warning(f"Failed to set inputs attribute: {e}")

    return original(*args, **kwargs)


def _semantic_kernel_chat_completion_response_wrapper(original, *args, **kwargs) -> None:
    # NB: Semantic Kernel logs chat completions, so we need to extract it and add it to the span.
    try:
        current_span = (args[0] if args else kwargs.get("current_span")) or get_current_span()
        completions = (args[1] if len(args) > 1 else kwargs.get("completions")) or []

        otel_span_id = current_span.get_span_context().span_id
        mlflow_span = _get_live_span_from_otel_span_id(otel_span_id)
        if not mlflow_span:
            _logger.debug(
                "Span is not found or recording. Skipping registering chat "
                f"completion attributes to {SpanAttributeKey.OUTPUTS}."
            )
            return original(*args, **kwargs)

        if are_sensitive_events_enabled():
            full_responses = []
            for completion in completions:
                full_response: dict[str, Any] = {
                    "message": completion.to_dict(),
                }

                if isinstance(completion, ChatMessageContent):
                    full_response["finish_reason"] = completion.finish_reason.value
                if isinstance(completion, StreamingContentMixin):
                    full_response["index"] = completion.choice_index

                full_responses.append(full_response)

            mlflow_span.set_outputs(full_responses)
            mlflow_span.set_attribute(SpanAttributeKey.CHAT_MESSAGES, full_responses)

    except Exception as e:
        _logger.warning(f"Failed to set outputs attribute: {e}")


async def _trace_wrapper(original, *args, **kwargs):
    span = get_current_span()
    if span and span.is_recording():

        otel_span_id = get_current_span().get_span_context().span_id
        mlflow_span = _get_live_span_from_otel_span_id(otel_span_id)

        inputs = _extract_inputs(args, kwargs)

        print("MLFLOW SPAN", mlflow_span)
        
        # Check if span already has inputs and append to them if present
        if mlflow_span and inputs:
            existing_inputs = getattr(mlflow_span, '_inputs', None)
            if existing_inputs:
                try:
                    # Parse existing inputs if they're in JSON format
                    if isinstance(existing_inputs, str):
                        existing_data = json.loads(existing_inputs)
                    else:
                        existing_data = existing_inputs
                    
                    # Merge existing and new inputs
                    if isinstance(existing_data, dict) and isinstance(inputs, dict):
                        merged_inputs = {**existing_data, **inputs}
                        mlflow_span.set_inputs(merged_inputs)
                    else:
                        mlflow_span.set_inputs(inputs)
                except (json.JSONDecodeError, TypeError):
                    # If we can't parse existing inputs, just set new ones
                    mlflow_span.set_inputs(inputs)
            else:
                mlflow_span.set_inputs(inputs)
            
            mlflow_span.set_attribute(SpanAttributeKey.FUNCTION_NAME, original.__qualname__)
            # print(f"Setting  {original.__qualname__} inputs:", inputs)

    try:
        result = await original(*args, **kwargs)
        if span and span.is_recording():
            output_payload ={}
            otel_span_id = get_current_span().get_span_context().span_id
            mlflow_span = _get_live_span_from_otel_span_id(otel_span_id)
            print("result: ",result)
                # Extract chat messages from value
            output_payload, extra_attrs = _extract_chat_and_attributes(result)
            print("output payload: ",output_payload)
            mlflow_span.set_outputs(output_payload)
        return result
    except Exception as e:
        if span and span.is_recording():
            span.set_attribute(SpanAttributeKey.OUTPUTS, json.dumps(f"Error: {e!s}"))
        raise


def _semantic_kernel_chat_completion_error_wrapper(original, *args, **kwargs) -> None:
    current_span = (args[0] if args else kwargs.get("current_span")) or get_current_span()
    error = args[1] if len(args) > 1 else kwargs.get("error")

    otel_span_id = current_span.get_span_context().span_id
    mlflow_span = _get_live_span_from_otel_span_id(otel_span_id)

    mlflow_span.add_event(SpanEvent.from_exception(error))
    mlflow_span.set_status(SpanStatusCode.ERROR)

    with InMemoryTraceManager.get_instance().get_trace(mlflow_span.trace_id) as t:
        t.info.status = TraceStatus.ERROR

    return original(*args, **kwargs)


async def _semantic_kernel_invoke_trace_wrapper(original, *args, **kwargs):
    """Wrapper for Kernel.invoke / invoke_* methods.

    1. Open an MLflow span (which in turn opens an OpenTelemetry span and makes it the
       *current* span in the OTEL context).
    2. Record function name, inputs, outputs, errors.
    3. Because the span is put in the OTEL context, every span that Semantic Kernel
       starts internally will automatically be a child of this span, ensuring a
       well-formed trace tree.
    """

    # Create a new span – if another span is already active this will automatically
    # become its child, otherwise it becomes the root of a new trace.
    with start_span(name=f"{original.__qualname__}", span_type=SpanType.CHAIN) as span:
        # Set up span context and record inputs
        _setup_span_context(span, original, args, kwargs)
        
        # Execute the actual Semantic Kernel call
        try:
            result = await original(*args, **kwargs)
            _record_span_outputs(span, result)
            return result
        except Exception as err:
            _handle_span_error(span, err)
            raise


def _setup_span_context(span: LiveSpan, original, args, kwargs) -> None:
    """Set up span context and record inputs."""
    try:
        inputs = _extract_inputs(args, kwargs)
        span.set_attribute(SpanAttributeKey.INPUTS, inputs)
        
        # Set up OTel span mapping
        token = set_span_in_context(span)
        otel_span_id = getattr(span, "_span", None).get_span_context().span_id  # type: ignore[attr-defined]
        _OTEL_SPAN_ID_TO_MLFLOW_SPAN_AND_TOKEN[otel_span_id] = (span, token)
        
    except Exception as e:
        span.set_attribute(SpanAttributeKey.INPUTS, f"Failed to serialize inputs: {e!s}")


def _extract_inputs(args, kwargs) -> dict:
    """Extract and serialize inputs from function arguments."""
    inputs = {}
    
    # Iterate through args tuple and serialize each argument
    if args:
        for i, arg in enumerate(args):
            # print(f"Processing arg {i}: {arg}")
            # Try different serialization methods
            if hasattr(arg, "dict"):
                serialized_arg = arg.dict()
            elif hasattr(arg, "model_dump"):
                serialized_arg = arg.model_dump()
            elif isinstance(arg, dict):
                serialized_arg = arg
            elif hasattr(arg, "items"):
                serialized_arg = arg.items()
            else:
                serialized_arg = str(arg)
            
            # print(f"Serialized arg {i}: {serialized_arg}")
            
            # Update inputs dict with serialized argument
            if isinstance(serialized_arg, dict):
                inputs.update(serialized_arg)
            else:
                inputs[f"arg{i}"] = serialized_arg

    # Iterate through kwargs dictionary and serialize each value
    if kwargs:
        for key, value in kwargs.items():
            if key not in ['user_input','chat_history', "argument"]:
                continue
            # print(f"Processing kwarg {key}: {value}")
            # Try different serialization methods
            if hasattr(value, "dict"):
                serialized_value = value.dict()
            elif hasattr(value, "model_dump"):
                serialized_value = value.model_dump()
            elif isinstance(value, dict):
                serialized_value = value
            elif hasattr(value, "items"):
                serialized_value = value.items()
            else:
                serialized_value = str(value)
            
            # print(f"Serialized kwarg {key}: {serialized_value}")
            
            # Update inputs dict with serialized value
            if isinstance(serialized_value, dict):
                # For dict values, we could either update directly or nest under the key
                # Nesting under the key preserves the parameter name
                inputs.update(serialized_value)
            else:
                inputs[key] = serialized_value

        # print("Inputs :",inputs)        
    
    return inputs


def _record_span_outputs(span: LiveSpan, result) -> None:
    """Record outputs and attributes from the function result."""
    output_payload, extra_attrs = _extract_chat_and_attributes(result)

    if output_payload:
        span.set_outputs(output_payload)
        # Set chat messages attribute for backwards compatibility
        if "messages" in output_payload:
            extra_attrs[SpanAttributeKey.CHAT_MESSAGES] = output_payload["messages"]
    else:
        # Fallback: log entire result as outputs
        try:
            from mlflow.tracing.utils import TraceJSONEncoder
            span.set_outputs(json.dumps(result, cls=TraceJSONEncoder))
        except Exception:
            span.set_outputs(str(result))

    # Attach remaining attributes
    span.set_attributes(extra_attrs)


def _handle_span_error(span: LiveSpan, error: Exception) -> None:
    """Handle errors in span execution."""
    span.set_attribute(SpanAttributeKey.OUTPUTS, f"Error: {error!s}")
    span.set_status(SpanStatusCode.ERROR)
    span.add_event(SpanEvent.from_exception(error))


def _extract_chat_and_attributes(result: Any) -> tuple[Optional[dict], dict[str, Any]]:
    """Parse the result returned by ``Kernel.invoke``.

    Returns:
        tuple: (output_payload, extra_attrs) where output_payload is ready for set_outputs
               and extra_attrs contains additional span attributes.
    """
    try:
        if hasattr(result, "function") and hasattr(result, "value"):
            return _extract_function_result_data(result)
        else:
            return {"output": result}, {}
    except Exception as err:
        _logger.debug(
            "Failed to extract chat / attributes from invoke result: %s", err, exc_info=True
        )
        return None, {"invoke_result": str(result)}


def _extract_function_result_data(result) -> tuple[dict, dict[str, Any]]:
    """Extract data from a Semantic Kernel FunctionResult object."""
    output_payload: dict[str, Any] = {}
    extra_attrs: dict[str, Any] = {}

    # Extract function information
    _extract_function_info(result, extra_attrs)
    
    # Extract rendered prompt
    if rendered_prompt := getattr(result, "rendered_prompt", None):
        output_payload["rendered_prompt"] = rendered_prompt

    # Extract metadata and messages
    if metadata := getattr(result, "metadata", None):
        _extract_metadata_info(metadata, output_payload, extra_attrs)

    # Extract chat messages from value
    _extract_value_messages(result, output_payload)

    return output_payload, extra_attrs


def _extract_function_info(result, extra_attrs: dict) -> None:
    """Extract function information into extra attributes."""
    try:
        fn_obj = getattr(result, "function", None)
        if fn_obj is not None and hasattr(fn_obj, "dict"):
            extra_attrs["function"] = fn_obj.dict()
        else:
            extra_attrs["function"] = str(fn_obj)
    except Exception:
        extra_attrs["function"] = str(getattr(result, "function", None))


def _extract_metadata_info(metadata, output_payload: dict, extra_attrs: dict) -> None:
    """Extract metadata information and any embedded messages."""
    try:
        meta_dict = metadata if isinstance(metadata, dict) else getattr(metadata, "__dict__", {})
    except Exception:
        meta_dict = {}

    # Extract messages from metadata if present
    if meta_dict and isinstance(meta_dict, dict):
        sk_messages = meta_dict.get("messages")
        if sk_messages and hasattr(sk_messages, "dict"):
            sk_messages = sk_messages.dict().get("messages", [])
            
            if sk_messages:
                chat_messages = _parse_message_list(sk_messages)
                if chat_messages:
                    _merge_messages_into_payload(chat_messages, output_payload)

        # Store metadata (excluding large message objects) as attribute
        reduced_meta = {k: v for k, v in meta_dict.items() if k != "messages"}
        if reduced_meta:
            extra_attrs["metadata"] = reduced_meta


def _extract_value_messages(result, output_payload: dict) -> None:
    """Extract chat messages from the result value."""
    value = getattr(result, "value", None)
    if value is None:
        return

    chat_messages = []
    if isinstance(value, list):
        for item in value:
            if parsed_msg := _parse_message_like(item):
                chat_messages.append(parsed_msg)
    else:
        if parsed_msg := _parse_message_like(value):
            chat_messages.append(parsed_msg)

    if chat_messages:
        _merge_messages_into_payload(chat_messages, output_payload)
    else:
        # If we couldn't turn the value into messages, keep it as-is
        output_payload["value"] = value


def _parse_message_list(messages: list) -> list[ChatMessage]:
    """Parse a list of message-like objects into ChatMessage objects."""
    chat_messages = []
    for msg in messages:
        if parsed_msg := _parse_message_like(msg):
            chat_messages.append(parsed_msg)
    return chat_messages


def _merge_messages_into_payload(chat_messages: list[ChatMessage], output_payload: dict) -> None:
    """Merge chat messages into the output payload."""
    chat_dicts = [m.model_dump_compat() for m in chat_messages]
    if "messages" in output_payload:
        output_payload["messages"].extend(chat_dicts)
    else:
        output_payload["messages"] = chat_dicts


def _parse_message_like(message_like: Any) -> Optional[ChatMessage]:
    """Parse a Semantic Kernel message-like object into MLflow ChatMessage format."""
    try:
        if isinstance(message_like, BaseModel):
            message_like = message_like.model_dump()
        
        role_enum = message_like.get("role")
        role = getattr(role_enum, "value", str(role_enum))  # handle AuthorRole enums or strings

        content_parts = []
        refusal = None
        tool_calls = []
        tool_result = None
        tool_call_id = None

        for item in message_like.get("items", []):
            item_type = item.get("content_type")

            if item_type == "text":
                content_parts.append(TextContentPart(type="text", text=item.get("text", "")))
            elif item_type == "refusal":
                refusal = item.get("refusal")
            elif item_type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=item.get("id"),
                        function=Function(
                            name=item.get("function_name"),
                            arguments=item.get("arguments"),
                        ),
                    )
                )
            elif item_type == "function_result":
                tool_result = item.get("result")
                tool_call_id = item.get("id")
            else:
                _logger.debug(f"Unknown item content_type: {item_type} in item: {item}")

        return _create_chat_message(role, content_parts, refusal, tool_calls, tool_result, tool_call_id)
    
    except Exception as e:
        _logger.debug(f"Failed to parse message-like object: {e}")
        return None


def _create_chat_message(role: str, content_parts: list, refusal: Optional[str], 
                        tool_calls: list, tool_result: Optional[str], 
                        tool_call_id: Optional[str]) -> ChatMessage:
    """Create a ChatMessage based on the parsed components."""
    if tool_calls:
        return ChatMessage(role=role, content="", tool_calls=tool_calls)
    elif tool_result is not None:
        return ChatMessage(role=role, content=tool_result, tool_call_id=tool_call_id)
    else:
        return ChatMessage(role=role, content=content_parts, refusal=refusal)

