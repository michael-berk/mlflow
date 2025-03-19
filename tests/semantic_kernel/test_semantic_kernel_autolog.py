import openai
import pytest
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import FunctionResult, KernelArguments

import mlflow.semantic_kernel

from tests.tracing.helper import get_traces


def test_mock_openai_completion(mock_openai):
    # TODO remove
    client = openai.OpenAI(base_url=mock_openai)

    assert client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )


async def create_and_invoke_kernel(mock_openai):
    kernel = Kernel()
    openai_client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)

    kernel.add_service(
        OpenAIChatCompletion(
            service_id="chat-gpt",
            ai_model_id="gpt-3.5-turbo",
            async_client=openai_client,
        )
    )

    chat_function = kernel.add_function(
        plugin_name="ChatBot",
        function_name="Chat",
        prompt="{{$chat_history}}{{$user_input}}",
        template_format="semantic-kernel",
    )

    chat_history = ChatHistory(system_message="You are a chatbot that is super super unhelpful.")
    user_input = "How do genetic algorithms work?"
    return await kernel.invoke(
        chat_function, KernelArguments(user_input=user_input, chat_history=chat_history)
    )


@pytest.mark.asyncio
async def test_autolog_tracing(mock_openai):
    mlflow.semantic_kernel.autolog()
    response = await create_and_invoke_kernel(mock_openai)

    assert isinstance(response, FunctionResult)
    assert response.function.name == "Chat"

    # print("========================")
    # print("========================")
    # print("========================")
    traces = get_traces()
    # print(traces[0].to_dict())
    assert len(traces) == 1
    assert traces[0].data.spans[0].name == "Kernel.invoke"


"""
Steps
1. Get patching to work with OpenAI server - just create a semantic kernel kernel fixture
    - Note that you can use `MOCK_TOOLS` for tools
2. Create tests for basic span
3. Cover semantic kernel key cases
4. Implement async patched call - copy openai implmementation
5. Implement translation layer similar to CrewAI
"""
