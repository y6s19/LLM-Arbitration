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
    lines = []
    for m in messages:
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

    # ---------------- IMPROVED ARBITRATOR ----------------
    # Forces explicit extraction + arithmetic BEFORE any verdict, instead of
    # letting the model jump straight to a plausible-sounding conclusion.
    arbitrator = AssistantAgent(
        name="Arbitrator",
        model_client=model_client_arb,
        system_message=(
            "You are a strict, methodical Arbitrator reviewing a negotiation transcript between two "
            "campers splitting 3 Food, 3 Water, and 3 Firewood packages in total.\n\n"
            "You MUST follow this exact procedure, in order, and show every step:\n\n"
            "STEP 1 -- EXTRACT: Go through the transcript and write down the FINAL number of each item "
            "each camper ends up with, as stated in their last agreement on that item. "
            "Format as a table:\n"
            "  Food:     Camper_A = ?, Camper_B = ?\n"
            "  Water:    Camper_A = ?, Camper_B = ?\n"
            "  Firewood: Camper_A = ?, Camper_B = ?\n"
            "If the transcript is contradictory or unclear about the final number for any item, write "
            "'UNCLEAR' instead of guessing.\n\n"
            "STEP 2 -- SUM: For each item, add the two numbers together and write the total explicitly, "
            "e.g. 'Food total = 2 + 1 = 3'.\n\n"
            "STEP 3 -- CHECK: For each item, state whether the total equals exactly 3 (YES/NO), and "
            "whether both numbers are whole numbers (YES/NO).\n\n"
            "STEP 4 -- VERDICT: Only after completing Steps 1-3, give:\n"
            "VERDICT: APPROVED (only if every item passed every check in Step 3) or REJECTED\n"
            "REASON: reference the specific item(s) and numbers that failed, if any\n"
            "IF REJECTED, SUGGESTED FIX: a valid whole-number split for the failing item(s)\n\n"
            "Do not skip steps. Do not approve anything you marked UNCLEAR or that failed a check."
        ),
    )

    print("=== Arbitrator review (v2: forced arithmetic) ===\n")
    arb_result = await arbitrator.run(
        task=f"Here is the negotiation transcript:\n\n{transcript}\n\nReview the final agreed split."
    )
    print(arb_result.messages[-1].content)

    await model_client_a.close()
    await model_client_b.close()
    await model_client_arb.close()


if __name__ == "__main__":
    asyncio.run(main())
