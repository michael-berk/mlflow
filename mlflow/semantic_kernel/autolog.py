import inspect
import logging

import mlflow
import mlflow.semantic_kernel

# from mlflow.anthropic.chat import convert_message_to_mlflow_chat, convert_tool_to_mlflow_chat_tool
from mlflow.entities import SpanType
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def _get_span_type(task_name: str) -> str:
    span_type_mapping = {
        "create": SpanType.CHAT_MODEL,
    }
    return span_type_mapping.get(task_name, SpanType.UNKNOWN)


def construct_full_inputs(func, *args, **kwargs):
    signature = inspect.signature(func)
    # this does not create copy. So values should not be mutated directly
    arguments = signature.bind_partial(*args, **kwargs).arguments

    if "self" in arguments:
        arguments.pop("self")

    return arguments


# TODO: this needs to be async
# NOTE: async support was just added for MLflow safe_patch
def patched_class_call(original, self, *args, **kwargs):
    # print("PATCHED CLASS CALL!!!!!")
    config = AutoLoggingConfig.init(flavor_name=mlflow.anthropic.FLAVOR_NAME)

    # TODO: set mlflow provider as tracer_provider (implementation 4)

    if config.log_traces:
        # NOTE: context manager blocks the thread, and there is no async support
        # Need to use client API to start/end span instead of context manager (check OpenAI)
        with mlflow.start_span(
            name=f"{self.__class__.__name__}.{original.__name__}",
            span_type=_get_span_type(original.__name__),
        ) as span:
            inputs = construct_full_inputs(original, self, *args, **kwargs)
            # NOTE: input/output/span type (LLM, ChatModel, Retriever, Function, Tool) are MVP
            span.set_inputs(inputs)
            try:
                outputs = original(self, *args, **kwargs)
                # NOTE: as of this point, all semantic kernnel spans will be here in memory
                span.set_outputs(outputs)
            finally:
                # Set message attribute once at the end to avoid multiple JSON serialization
                # try:
                #     messages.append(convert_message_to_mlflow_chat(outputs))
                #     set_span_chat_messages(span, messages)
                # except Exception as e:
                #     _logger.debug(f"Failed to set chat messages for {span}. Error: {e}")
                pass

            return outputs

            # if (tools := inputs.get("tools")) is not None:
            #     try:
            #         tools = [convert_tool_to_mlflow_chat_tool(tool) for tool in tools]
            #         set_span_chat_tools(span, tools)
            #     except Exception as e:
            #         _logger.debug(f"Failed to set tools for {span}. Error: {e}")

            # messages = [convert_message_to_mlflow_chat(msg) for msg in inputs.get("messages", [])]
            # try:
            #     outputs = original(self, *args, **kwargs)
            #     span.set_outputs(outputs)
            # finally:
            #     # Set message attribute once at the end to avoid multiple JSON serialization
            #     try:
            #         messages.append(convert_message_to_mlflow_chat(outputs))
            #         set_span_chat_messages(span, messages)
            #     except Exception as e:
            #         _logger.debug(f"Failed to set chat messages for {span}. Error: {e}")

            # return outputs
