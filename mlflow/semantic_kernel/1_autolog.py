import inspect
import logging
from typing import Any, Optional, Union

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

import mlflow
import mlflow.semantic_kernel
import mlflow.tracing.provider
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.assessment import MlflowClient
from mlflow.tracing.utils import (
    end_client_span_or_trace,
)
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


class SemanticKernelSpanProcessor(SimpleSpanProcessor):
    def __init__(self, base_span_processor: Union[SimpleSpanProcessor, BatchSpanProcessor]):
        self._base_span_processor = base_span_processor
        self.span_exporter = getattr(base_span_processor, "span_exporter", None)

        if not self.span_exporter:
            raise ValueError("Span exporter not found in base span processor.")

        self.span_exporter._active_span_processor = self

        # TODO: remove
        assert self.span_exporter
        print(self.span_exporter)
        print("SemanticKernelSpanProcessor initialized!!!!!!!!!!!!")

    def on_start(self, span: OTelSpan, parent_context: Optional[Context] = None):
        print("Wrapper on start!!!!!!!!")
        print("0-934" * 20)
        print(span.to_json())
        print("0-934" * 20)
        self._base_span_processor.on_start(span, parent_context)

    def on_end(self, span: OTelReadableSpan) -> None:
        # do some attribute conversion from genai.xyz -> mlflow format
        print("Wrapper on end!!!!!!!!")
        print("0-914" * 20)
        print(span.to_json())
        print("0-914" * 20)

        self._base_span_processor.on_end(span)


def wrap_mlflow_processor_with_semantic_kernel_processor():
    # print("wrap_mlflow_processor_with_semantic_kernel_processor CALLED!!!!")
    # from mlflow.tracing.provider import _MLFLOW_TRACER_PROVIDER

    # if not _MLFLOW_TRACER_PROVIDER:
    #     mlflow.tracing.provider._setup_tracer_provider()

    # multi_processor = _MLFLOW_TRACER_PROVIDER._active_span_processor
    # base_processor = multi_processor._span_processors[0]
    # wrapped = SemanticKernelSpanProcessor(base_processor)
    # multi_processor._span_processors = (wrapped,)
    pass


def _get_span_type(instance) -> str:
    from semantic_kernel.kernel import Kernel

    try:
        if isinstance(instance, Kernel):
            return SpanType.Agent

    except AttributeError as e:
        _logger.warning("An exception happens when resolving the span type. Exception: %s", e)

    return SpanType.UNKNOWN


def construct_full_inputs(func, *args, **kwargs):
    signature = inspect.signature(func)
    # this does not create copy. So values should not be mutated directly
    arguments = signature.bind_partial(*args, **kwargs).arguments

    if "self" in arguments:
        arguments.pop("self")

    return arguments


def _start_span(tracer, instance: Any, inputs: dict[str, Any], run_id: str):
    # Record input parameters to attributes
    attributes = {k: v for k, v in inputs.items() if k != "messages"}

    from mlflow.tracing.provider import _MLFLOW_TRACER_PROVIDER

    tracer = _MLFLOW_TRACER_PROVIDER.get_tracer(__name__)

    # TODO remove
    # assert _MLFLOW_TRACER_PROVIDER._active_span_processor._span_processors[0].__class__.__name__.endswith("SemanticKernelSpanProcessor")

    with tracer.start_as_current_span("test") as span:
        print("TESTTTTTTTTTTTTTTTTTT")
        print(span)
        print(type(span))
        span.set_attribute("test", "test")
        # Optional: store cm to call __exit__ later if needed
        print("BERK!!!!!!!!!")
        print(span)
        print(type(span))
        return span
    # cm = tracer.start_as_current_span("Kernel")
    # span = cm.__enter__()
    # # Optional: store cm to call __exit__ later if needed
    # print("BERK!!!!!!!!!")
    # print(span)
    # print(type(span))
    # return span, cm

    # If there is an active span, create a child span under it, otherwise create a new trace
    # span = tracer.start_as_current_span(
    #     name=instance.__class__.__name__,
    #     span_type=_get_span_type(instance.__class__),
    #     inputs=inputs,
    #     attributes=attributes,
    # )

    # Associate run ID to the trace manually, because if a new run is created by
    # autologging, it is not set as the active run thus not automatically
    # associated with the trace.
    # if run_id is not None:
    #     tm = InMemoryTraceManager().get_instance()
    #     tm.set_request_metadata(span.request_id, TraceMetadataKey.SOURCE_RUN, run_id)

    # return span


def _end_span_on_success(
    mlflow_client: MlflowClient, span: LiveSpan, inputs: dict[str, Any], raw_result: Any
):
    print("END SPAN ON SUCCESS")
    result = raw_result
    end_client_span_or_trace(mlflow_client, span, outputs=result)
    # TODO: add streaming support and set attributes


def _end_span_on_exception(mlflow_client: MlflowClient, span: LiveSpan, e: Exception):
    try:
        span.add_event(SpanEvent.from_exception(e))
        mlflow_client.end_span(span.request_id, span.span_id, status=SpanStatusCode.ERROR)
    except Exception as inner_e:
        _logger.warning(f"Encountered unexpected error when ending trace: {inner_e}")


def _get_autolog_run_id(instance, active_run):
    """
    Get the run ID to use for logging artifacts and associate with the trace.

    The run ID is determined as follows:
    - If there is an active run (created by a user), use its run ID.
    - If the model has a `_mlflow_run_id` attribute, use it. This is the run ID created
        by autologging in a previous call to the same model.
    """
    return active_run.info.run_id if active_run else getattr(instance, "_mlflow_run_id", None)


async def async_patched_class_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.semantic_kernel.FLAVOR_NAME)
    active_run = mlflow.active_run()
    run_id = _get_autolog_run_id(self, active_run)
    mlflow_client = mlflow.MlflowClient()

    span = None
    if config.log_traces:
        inputs = construct_full_inputs(original, self, *args, **kwargs)
        span, cm = _start_span(mlflow_client, self, inputs, run_id)

    try:
        raw_result = await original(self, *args, **kwargs)
    except Exception as e:
        if config.log_traces and span:
            _end_span_on_exception(mlflow_client, span, e)
            cm.__exit__(type(e), e, e.__traceback__)
        raise

    if config.log_traces and span:
        _end_span_on_success(mlflow_client, span, kwargs, raw_result)
        cm.__exit__(None, None, None)

    if run_id and (active_run is None or active_run.info.run_id != run_id):
        mlflow_client.set_terminated(run_id)

    return raw_result


# async def async_patched_class_call(original, self, *args, **kwargs):
#     config = AutoLoggingConfig.init(flavor_name=mlflow.gemini.FLAVOR_NAME)
#     print("ajsdl;gjasflkjsd")

#     if config.log_traces:
#         fullname = f"{self.__class__.__name__}.{original.__name__}"
#         span_type = _get_span_type(self)

#         with mlflow.start_span(name=fullname, span_type=span_type) as span:
#             try:
#                 result = await original(self, *args, **kwargs)
#             except Exception as e:
#                 span.record_exception(e)
#                 span.set_status("error")
#                 raise

#             outputs = result.__dict__ if hasattr(result, "__dict__") else result
#             span.set_outputs(outputs)
#             return result

#     return await original(self, *args, **kwargs)
