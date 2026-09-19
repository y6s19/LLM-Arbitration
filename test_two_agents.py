import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily


def make_ollama_client() -> OpenAIChatCompletionClient:
    """One shared helper so both agents talk to the same local model."""
    return OpenAIChatCompletionClient(
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


async def main() -> None:
    # Two agents with distinct roles -- the seed of your "Agent A vs Agent B" setup.
    model_client_a = make_ollama_client()
    model_client_b = make_ollama_client()

    agent_a = AssistantAgent(
        name="Optimist",
        model_client=model_client_a,
        system_message=(
            "You are Optimist, an agent who argues FOR the benefits of remote work. "
            "Keep replies to 2-3 sentences. Respond directly to what the other agent just said."
        ),
    )

    agent_b = AssistantAgent(
        name="Skeptic",
        model_client=model_client_b,
        system_message=(
            "You are Skeptic, an agent who argues AGAINST remote work, favouring in-office work. "
            "Keep replies to 2-3 sentences. Respond directly to what the other agent just said."
        ),
    )

    # Stop after 6 total messages (3 turns each) so it doesn't run forever.
    termination = MaxMessageTermination(max_messages=6)

    team = RoundRobinGroupChat(
        [agent_a, agent_b],
        termination_condition=termination,
    )

    # Console() streams each message to the terminal as it's generated.
    await Console(team.run_stream(task="Is remote work better than working in an office?"))

    await model_client_a.close()
    await model_client_b.close()


if __name__ == "__main__":
    asyncio.run(main())
