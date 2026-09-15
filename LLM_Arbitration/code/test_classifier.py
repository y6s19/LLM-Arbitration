import asyncio
from datasets import load_dataset
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from experiment_utils import make_ollama_client, describe_priorities, transcript_to_text, save_log

VALID_LABELS = {"agree", "disagree", "stall", "escalate"}


async def classify_turn(classifier: AssistantAgent, prior_context: str, speaker: str, text: str) -> str:
    """Ask the classifier for exactly one label for a single turn, given a bit of context."""
    prompt = (
        f"Conversation so far:\n{prior_context if prior_context else '(this is the first message)'}\n\n"
        f"New message from {speaker}: \"{text}\"\n\n"
        "Classify this NEW message as exactly one of: agree, disagree, stall, escalate.\n\n"
        "Definitions with examples:\n"
        "- agree: clearly accepts a specific proposal with no pushback. "
        "Example: \"Yes, that split works for me.\"\n"
        "- disagree: rejects, counters, proposes a swap/different split, or questions a proposal -- "
        "even if phrased politely or wrapped in cooperative language. "
        "Example: \"Could we instead swap X for Y?\" or \"Don't you think that's a bit much?\" "
        "These are DISAGREE, not agree, even though they sound friendly.\n"
        "- stall: avoids committing, repeats prior points, or delays without proposing anything new.\n"
        "- escalate: hardens their position, adds tough new demands, or raises stakes/pressure.\n\n"
        "IMPORTANT: A message that sounds polite or cooperative can still be a DISAGREE if it changes "
        "the terms, questions the other side, or proposes an alternative instead of accepting outright. "
        "Only use 'agree' if the speaker accepts the specific numbers already on the table with no changes.\n\n"
        "Reply with ONLY the single label word, nothing else."
    )
    result = await classifier.run(task=prompt)
    label = result.messages[-1].content.strip().lower()
    for v in VALID_LABELS:
        if v in label:
            return v
    return "unclear"  # classifier didn't return a clean label -- worth tracking separately


async def main() -> None:
    dataset = load_dataset("kchawla123/casino", split="train")
    scenario = dataset[0]
    p1 = scenario["participant_info"]["mturk_agent_1"]
    p2 = scenario["participant_info"]["mturk_agent_2"]
    p1_priorities = describe_priorities(p1["value2issue"], p1["value2reason"])
    p2_priorities = describe_priorities(p2["value2issue"], p2["value2reason"])

    model_client_a = make_ollama_client()
    model_client_b = make_ollama_client()
    model_client_clf = make_ollama_client()

    camper_a = AssistantAgent(
        name="Camper_A", model_client=model_client_a,
        system_message=(
            "You are Camper A, negotiating over 3 Food, 3 Water, and 3 Firewood packages.\n"
            f"Your real priorities:\n{p1_priorities}\n"
            "Keep replies short (2-3 sentences). Packages are whole units only."
        ),
    )
    camper_b = AssistantAgent(
        name="Camper_B", model_client=model_client_b,
        system_message=(
            "You are Camper B, negotiating over 3 Food, 3 Water, and 3 Firewood packages.\n"
            f"Your real priorities:\n{p2_priorities}\n"
            "Keep replies short (2-3 sentences). Packages are whole units only."
        ),
    )
    classifier = AssistantAgent(
        name="Classifier",
        model_client=model_client_clf,
        system_message="You are a precise conversation-turn classifier. You only ever reply with one label word.",
    )

    termination = MaxMessageTermination(max_messages=10)
    team = RoundRobinGroupChat([camper_a, camper_b], termination_condition=termination)

    print("=== Running negotiation ===\n")
    result = await team.run(
        task="Negotiate how to split the 3 Food, 3 Water, and 3 Firewood packages between you."
    )

    turns = [m for m in result.messages if getattr(m, "source", None) not in (None, "user")]

    print("=== Classifying each turn ===\n")
    output_lines = []
    running_context = ""
    label_counts = {"agree": 0, "disagree": 0, "stall": 0, "escalate": 0, "unclear": 0}

    for m in turns:
        label = await classify_turn(classifier, running_context, m.source, m.content)
        label_counts[label] = label_counts.get(label, 0) + 1
        line = f"[{label.upper():9}] {m.source}: {m.content}"
        print(line)
        output_lines.append(line)
        running_context += f"{m.source}: {m.content}\n"

    summary = (
        "\n=== Summary ===\n"
        f"Total turns classified: {len(turns)}\n"
        + "\n".join(f"{k}: {v}" for k, v in label_counts.items())
    )
    print(summary)
    output_lines.append(summary)

    full_log = "=== NEGOTIATION TRANSCRIPT ===\n" + transcript_to_text(result.messages) + \
               "\n\n=== CLASSIFIED TURNS ===\n" + "\n".join(output_lines)
    save_log(full_log, prefix="classifier_v2_run")

    await model_client_a.close()
    await model_client_b.close()
    await model_client_clf.close()


if __name__ == "__main__":
    asyncio.run(main())
