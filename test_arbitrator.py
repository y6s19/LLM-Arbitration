import asyncio
from datasets import load_dataset
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
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
    lines = []
    for level in ["High", "Medium", "Low"]:
        item = value2issue.get(level, "?")
        reason = value2reason.get(level, "")
        lines.append(f"- {level} priority: {item} (reason: \"{reason}\")")
    return "\n".join(lines)


def transcript_to_text(messages) -> str:
    """Turn the negotiation's message list into plain text for the Arbitrator to read."""
    lines = []
    for m in messages:
        # Skip the very first task message from 'user'
        if getattr(m, "source", None) in (None, "user"):
            continue
        lines.append(f"{m.source}: {m.content}")
    return "\n".join(lines)


async def main() -> None:
    dataset = load_dataset("kchawla123/casino", split="train")
    scenario = dataset[0]

    p1 = scenario["participant_info"]["mturk_agent_1"]
    p2 = scenario["participant_info"]["mturk_agent_2"]
    p1_priorities = describe_priorities(p1["value2issue"], p1["value2reason"])
    p2_priorities = describe_priorities(p2["value2issue"], p2["value2reason"])

    model_client_a = make_ollama_client()
    model_client_b = make_ollama_client()
    model_client_arb = make_ollama_client()

    camper_a = AssistantAgent(
        name="Camper_A",
        model_client=model_client_a,
        system_message=(
            "You are Camper A, negotiating with your campsite neighbour over extra packages "
            "of Food, Water, and Firewood (3 of each are available in total to split between you).\n"
            f"Your real priorities:\n{p1_priorities}\n"
            "Negotiate honestly but firmly based on these priorities. Keep replies short (2-3 sentences). "
            "Packages are whole/indivisible units -- never propose half or fractional packages. "
            "Try to reach a fair, valid split with your neighbour."
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
            "Packages are whole/indivisible units -- never propose half or fractional packages. "
            "Try to reach a fair, valid split with your neighbour."
        ),
    )

    termination = MaxMessageTermination(max_messages=10)
    team = RoundRobinGroupChat([camper_a, camper_b], termination_condition=termination)

    print("=== Running negotiation ===\n")
    result = await team.run(
        task="Negotiate how to split the 3 Food, 3 Water, and 3 Firewood packages between you."
    )

    for m in result.messages:
        if getattr(m, "source", None) not in (None, "user"):
            print(f"{m.source}: {m.content}\n")

    transcript = transcript_to_text(result.messages)

    # ---------------- ARBITRATOR ----------------
    arbitrator = AssistantAgent(
        name="Arbitrator",
        model_client=model_client_arb,
        system_message=(
            "You are a neutral Arbitrator reviewing a completed negotiation between two campers "
            "over 3 Food, 3 Water, and 3 Firewood packages total.\n"
            "Check the final agreed split for VALIDITY:\n"
            "1. Each item's split must use whole numbers only (no fractions/halves).\n"
            "2. Each item's split must sum to exactly 3 (not more, not less).\n"
            "3. Both parties must have clearly agreed to the same numbers.\n\n"
            "Respond in this exact format:\n"
            "VERDICT: APPROVED or REJECTED\n"
            "REASON: <one or two sentences>\n"
            "IF REJECTED, SUGGESTED FIX: <a valid whole-number split that respects both parties' stated priorities>"
        ),
    )

    print("=== Arbitrator review ===\n")
    arb_result = await arbitrator.run(
        task=f"Here is the negotiation transcript:\n\n{transcript}\n\nReview the final agreed split."
    )
    print(arb_result.messages[-1].content)

    await model_client_a.close()
    await model_client_b.close()
    await model_client_arb.close()


if __name__ == "__main__":
    asyncio.run(main())
