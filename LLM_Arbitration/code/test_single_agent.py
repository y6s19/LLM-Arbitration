import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily


async def main() -> None:
    # Point this at your local Ollama server instead of the paid API.
    # api_key can be any non-empty string -- Ollama doesn't check it.
    model_client = OpenAIChatCompletionClient(
        model="llama3.1:8b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
    )

    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful, concise assistant.",
    )

    result = await agent.run(task="Say hello and tell me one fun fact about octopuses.")
    print(result.messages[-1].content)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
