import json
import logging
import os
from typing import Any, Optional, Union

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace import get_current_span, set_tracer_provider
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.streaming_content_mixin import StreamingContentMixin
from semantic_kernel.utils.telemetry.model_diagnostics.decorators import (
    CHAT_COMPLETION_OPERATION,
    CHAT_STREAMING_COMPLETION_OPERATION,
    TEXT_COMPLETION_OPERATION,
    TEXT_STREAMING_COMPLETION_OPERATION,
    are_sensitive_events_enabled,
)
from semantic_kernel.utils.telemetry.model_diagnostics.gen_ai_attributes import (
    ERROR_TYPE,
    OPERATION,
)

import mlflow
import mlflow.tracing.provider
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.assessment import MlflowClient
from mlflow.tracing.constant import (
    SpanAttributeKey,
    TraceMetadataKey,
)
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import deduplicate_span_names_in_place

_logger = logging.getLogger(__name__)


def _get_span_type(span: OTelSpan) -> str:
    span_type = None
    print("get_span_type CALLED")

    # Parse gen_ai.operation.name
    if hasattr(span, "attributes") and (operation := span.attributes.get(OPERATION)):
        span_map = {
            CHAT_COMPLETION_OPERATION: SpanType.CHAT_MODEL,
            CHAT_STREAMING_COMPLETION_OPERATION: SpanType.CHAT_MODEL,
            TEXT_COMPLETION_OPERATION: SpanType.LLM,
            TEXT_STREAMING_COMPLETION_OPERATION: SpanType.LLM,
        }
        span_type = span_map.get(operation)
        print(f"spantypefound: {span_type}")

    return span_type or SpanType.UNKNOWN


class SemanticKernelSpanProcessor(SimpleSpanProcessor):
    def __init__(self, base_span_processor: Union[SimpleSpanProcessor, BatchSpanProcessor]):
        self._base_span_processor = base_span_processor
        self.span_exporter = getattr(base_span_processor, "span_exporter", None)

        if not self.span_exporter:
            raise ValueError("Span exporter not found in base span processor.")

    def on_start(self, span: OTelSpan, parent_context: Optional[Context] = None):
        print("Wrapper on start!!!!!!!!")
        print("0-934" * 20)
        # print(span.to_json())
        print("0-934" * 20)
        self._base_span_processor.on_start(span, parent_context)

        tm = InMemoryTraceManager.get_instance()
        request_id = tm.get_request_id_from_trace_id(span.context.trace_id)
        span_type = _get_span_type(span)

        if span.parent and not request_id:
            _logger.debug(
                "Received a non-root span but the request ID is not found. "
                "The trace has likely been halted due to a timeout expiration."
            )
            return

        if active_run := mlflow.active_run():
            tm.set_request_metadata(
                span.request_id, TraceMetadataKey.SOURCE_RUN, active_run.info.run_id
            )

        live_span = LiveSpan(span, request_id, span_type)
        tm.register_span(live_span)

    def on_end(self, span: OTelReadableSpan) -> None:
        print("Wrapper on end!!!!!!!!")
        print("0-914" * 20)
        # print(span.to_json())
        print("0-914" * 20)

        if span._parent is not None:
            tm = InMemoryTraceManager.get_instance()
            request_id = tm.get_request_id_from_trace_id(span.context.trace_id)

            with tm.get_trace(request_id) as trace:
                if trace is None:
                    _logger.debug(f"Trace data with request ID {request_id} not found.")
                    return

                self._base_span_processor._update_trace_info(trace, span)
                deduplicate_span_names_in_place(list(trace.span_dict.values()))

            if hasattr(self._base_span_processor, "_call_super_on_end"):
                self._base_span_processor._call_super_on_end(span)
            else:
                _logger.debug("Base span processor does not have _call_super_on_end method.")

            print("SUCCESSFUL On end")
            # print(json.dumps(dump_all_traces(InMemoryTraceManager.get_instance()), cls=TraceJSONEncoder))
        self._base_span_processor.on_end(span)


def _semantic_kernel_chat_completion_input_wrapper(original, *args, **kwargs) -> None:
    print("_semantic_kernel_chat_completion_input_wrapper")
    try:
        prompt = args[1] if len(args) > 1 else kwargs.get("prompt")

        if isinstance(prompt, ChatHistory):
            prompt_value = [msg.to_dict() for msg in prompt.messages]
        elif not isinstance(prompt, list):
            prompt_value = [prompt]
        else:
            prompt_value = prompt

        prompt_value_with_message = {"messages": prompt_value}

        span = get_current_span()
        if span and span.is_recording():
            span.set_attribute(SpanAttributeKey.SPAN_TYPE, SpanType.CHAT_MODEL)
            span.set_attribute(SpanAttributeKey.INPUTS, json.dumps(prompt_value_with_message))
        else:
            _logger.debug(
                "Span is not found or recording. Skipping registering chat "
                f"completion attributes to {SpanAttributeKey.INPUTS}."
            )
    except Exception as e:
        _logger.warning(f"Failed to set inputs attribute: {e}")

    return original(*args, **kwargs)


def _semantic_kernel_chat_completion_response_wrapper(original, *args, **kwargs) -> None:
    print("_semantic_kernel_chat_completion_response_wrapper")
    try:
        current_span = args[0] if args else kwargs.get("current_span")
        completions = args[1] if len(args) > 1 else kwargs.get("completions")

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

            if current_span.is_recording():
                current_span.set_attribute(SpanAttributeKey.OUTPUTS, json.dumps(full_responses))
            else:
                _logger.debug(
                    "Span is not found or recording. Skipping registering chat "
                    f"completion attributes to {SpanAttributeKey.OUTPUTS}."
                )

    except Exception as e:
        _logger.warning(f"Failed to set outputs attribute: {e}")

    return original(*args, **kwargs)


def _set_completion_error_wrapper(original, *args, **kwargs) -> None:
    """Set an error for a text or chat completion ."""
    print("ERROR HANDLER CALLEDJK:w")
    span = args[0] if args else kwargs.get("span")
    error = args[1] if len(args) > 1 else kwargs.get("error")

    try:
        if span.is_recording():
            span.add_event(SpanEvent.from_exception(error).json())

        tm = InMemoryTraceManager.get_instance()
        request_id = tm.get_request_id_from_trace_id(span.context.trace_id)
        MlflowClient().end_span(
            request_id, span.get_span_context().span_id, status=SpanStatusCode.ERROR
        )
    except Exception as inner_e:
        _logger.warning(f"Encountered unexpected error when ending trace: {inner_e}")

    span.set_attribute(ERROR_TYPE, str(type(error)))

    return original(*args, **kwargs)


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


def _wrap_mlflow_processor_with_semantic_kernel_processor():
    print("wrap_mlflow_processor_with_semantic_kernel_processor CALLED!!!!")
    from mlflow.tracing.provider import _MLFLOW_TRACER_PROVIDER

    if not _MLFLOW_TRACER_PROVIDER or not hasattr(
        _MLFLOW_TRACER_PROVIDER, "_active_span_processor"
    ):
        mlflow.tracing.provider._setup_tracer_provider()

    from mlflow.tracing.provider import _MLFLOW_TRACER_PROVIDER, _MLFLOW_TRACER_PROVIDER_INITIALIZED

    multi_processor = _MLFLOW_TRACER_PROVIDER._active_span_processor
    current_span_processor = multi_processor._span_processors[0]
    wrapped = SemanticKernelSpanProcessor(current_span_processor)
    multi_processor._span_processors = (wrapped,)

    # NB: we must set this value to True to avoid re-initializing the provider
    # in the _get_tracer method, which is called when starting a detached span.
    _MLFLOW_TRACER_PROVIDER_INITIALIZED.done = True


def _set_tracer_provider_with_mlflow_tracer_provider():
    from mlflow.tracing.provider import _MLFLOW_TRACER_PROVIDER

    if _MLFLOW_TRACER_PROVIDER:
        set_tracer_provider(_MLFLOW_TRACER_PROVIDER)
    else:
        raise ValueError("MLflow tracer provider not initialized. Please initialize it first.")


def setup_semantic_kernel_tracing():
    _set_logging_env_variables()
    _wrap_mlflow_processor_with_semantic_kernel_processor()
    _set_tracer_provider_with_mlflow_tracer_provider()


# TODO: remove this util
# def dump_all_traces(self) -> dict[str, dict]:
#     """
#     Return a dictionary of all traces currently in memory, formatted for debugging.
#     """
#     with self._lock:
#         return {
#             request_id: {
#                 "trace_info": trace.info,
#                 "span_ids": list(trace.span_dict.keys()),
#                 "root_span_id": trace.get_root_span().span_id if trace.get_root_span() else None,
#                 "num_spans": len(trace.span_dict),
#             }
#             for request_id, trace in self._traces.items()
#         }
