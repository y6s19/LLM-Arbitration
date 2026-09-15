import asyncio
from datasets import load_dataset
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily


def make_ollama_client() -> OpenAIChatCompletionClient:
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


def describe_priorities(value2issue: dict, value2reason: dict) -> str:
    """Turn the dataset's priority fields into a readable sentence for the system prompt."""
    lines = []
    for level in ["High", "Medium", "Low"]:
        item = value2issue.get(level, "?")
        reason = value2reason.get(level, "")
        lines.append(f"- {level} priority: {item} (reason: \"{reason}\")")
    return "\n".join(lines)


async def main() -> None:
    # Load the real CaSiNo negotiation dataset (downloads once, then cached locally).
    dataset = load_dataset("kchawla123/casino", split="train")
    scenario = dataset[0]  # first real scenario; change the index to try others

    p1 = scenario["participant_info"]["mturk_agent_1"]
    p2 = scenario["participant_info"]["mturk_agent_2"]

    p1_priorities = describe_priorities(p1["value2issue"], p1["value2reason"])
    p2_priorities = describe_priorities(p2["value2issue"], p2["value2reason"])

    print("=== Real CaSiNo scenario loaded ===")
    print("Camper A priorities:\n", p1_priorities)
    print("\nCamper B priorities:\n", p2_priorities)
    print("=" * 40, "\n")

    model_client_a = make_ollama_client()
    model_client_b = make_ollama_client()

    camper_a = AssistantAgent(
        name="Camper_A",
        model_client=model_client_a,
        system_message=(
            "You are Camper A, negotiating with your campsite neighbour over extra packages "
            "of Food, Water, and Firewood (3 of each are available in total to split between you).\n"
            f"Your real priorities:\n{p1_priorities}\n"
            "Negotiate honestly but firmly based on these priorities. Keep replies short (2-3 sentences). "
            "Try to reach a fair split with your neighbour."
        ),
    )

    camper_b = AssistantAgent(
        name="Camper_B",
        model_client=model_client_b,
        system_message=(
            "You are Camper B, negotiating with your campsite neighbour over extra packages "
            "of Food, Water, and Firewood (3 of each are available in total to split between you).\n"
            f"Your real priorities:\n{p2_priorities}\n"
            "Negotiate honestly but firmly based on these priorities. Keep replies short (2-3 sentences). "
            "Try to reach a fair split with your neighbour."
        ),
    )

    termination = MaxMessageTermination(max_messages=10)
    team = RoundRobinGroupChat([camper_a, camper_b], termination_condition=termination)

    await Console(team.run_stream(
        task="Negotiate how to split the 3 Food, 3 Water, and 3 Firewood packages between you."
    ))

    await model_client_a.close()
    await model_client_b.close()


if __name__ == "__main__":
    asyncio.run(main())
