import asyncio
import re
from datasets import load_dataset
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from experiment_utils import make_ollama_client, describe_priorities, transcript_to_text, save_log

VALID_LABELS = ["agree", "disagree", "stall", "escalate"]


async def classify_turn(classifier: AssistantAgent, prior_context: str, speaker: str, text: str, is_first: bool) -> tuple[str, str]:
    """Ask the classifier to reason step-by-step, THEN commit to a label on its own line.
    Returns (label, full_reasoning) so we can log the reasoning too."""
    opener_note = (
        "\nNOTE: This is the FIRST substantive message -- there is no prior proposal to react to. "
        "If so, treat it as an opening move: label it 'agree' ONLY if the scheme forces a choice, "
        "but prefer 'escalate' if it stakes out a maximalist/aggressive opening position, "
        "otherwise pick whichever of the four labels is the least misleading opening description.\n"
        if is_first else ""
    )
    prompt = (
        f"Conversation so far:\n{prior_context if prior_context else '(this is the first message)'}\n"
        f"{opener_note}\n"
        f"New message from {speaker}: \"{text}\"\n\n"
        "Follow these steps IN ORDER, showing your work:\n\n"
        "STEP 1: What exact terms/numbers (if any) were on the table BEFORE this new message?\n"
        "STEP 2: What does this NEW message actually do? Choose one: "
        "(a) accepts those exact terms unchanged, (b) changes/rejects/counters the terms or questions them, "
        "(c) repeats/delays without proposing anything new, (d) hardens position or adds a tough new demand.\n"
        "STEP 3: Based on Step 2 ONLY (not on how politely it's phrased), map to a label:\n"
        "  (a) -> agree   (b) -> disagree   (c) -> stall   (d) -> escalate\n"
        "Remember: a polite or cooperative TONE does not make something 'agree' -- "
        "a counter-offer or new request is still 'disagree' even if phrased nicely.\n\n"
        "End your reply with a final line in EXACTLY this format (nothing after it):\n"
        "LABEL: <one word>"
    )
    result = await classifier.run(task=prompt)
    full_text = result.messages[-1].content.strip()

    # 1) Try to find an explicit "LABEL:" line first (most reliable when it works).
    match = re.search(r"LABEL[:\s]+\**\s*(\w+)", full_text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().lower()
        if candidate in VALID_LABELS:
            return candidate, full_text

    # 2) Fallback: check the LAST non-empty line only (not the whole reasoning,
    # which will mention other label words while explaining what something ISN'T).
    last_line = [ln for ln in full_text.splitlines() if ln.strip()][-1].lower()
    # Check longer/more specific words first -- "agree" is a substring of
    # "disagree", so checking "agree" first would wrongly match "disagree".
    for label in ["disagree", "escalate", "stall", "agree"]:
        if label in last_line:
            return label, full_text

    # 3) Still nothing found -- print the raw text so we can see why, instead
    # of silently losing information.
    print(f"    [parse failed -- raw last line: \"{last_line[:150]}\"]")
    return "unclear", full_text


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
        system_message="You are a precise, methodical conversation-turn classifier. You always show your reasoning steps before giving a final LABEL line.",
    )

    termination = MaxMessageTermination(max_messages=10)
    team = RoundRobinGroupChat([camper_a, camper_b], termination_condition=termination)

    print("=== Running negotiation ===\n")
    result = await team.run(
        task="Negotiate how to split the 3 Food, 3 Water, and 3 Firewood packages between you."
    )
    turns = [m for m in result.messages if getattr(m, "source", None) not in (None, "user")]

    print("=== Classifying each turn (v3: forced reasoning) ===\n")
    output_lines = []
    running_context = ""
    label_counts = {"agree": 0, "disagree": 0, "stall": 0, "escalate": 0, "unclear": 0}

    for i, m in enumerate(turns):
        label, reasoning = await classify_turn(classifier, running_context, m.source, m.content, is_first=(i == 0))
        label_counts[label] = label_counts.get(label, 0) + 1
        print(f"[{label.upper():9}] {m.source}: {m.content}")
        output_lines.append(f"[{label.upper():9}] {m.source}: {m.content}\n  reasoning: {reasoning}\n")
        running_context += f"{m.source}: {m.content}\n"

    summary = (
        "\n=== Summary ===\n"
        f"Total turns classified: {len(turns)}\n"
        + "\n".join(f"{k}: {v}" for k, v in label_counts.items())
    )
    print(summary)
    output_lines.append(summary)

    full_log = "=== NEGOTIATION TRANSCRIPT ===\n" + transcript_to_text(result.messages) + \
               "\n\n=== CLASSIFIED TURNS (with reasoning) ===\n" + "\n".join(output_lines)
    save_log(full_log, prefix="classifier_v3_run")

    await model_client_a.close()
    await model_client_b.close()
    await model_client_clf.close()


if __name__ == "__main__":
    asyncio.run(main())
