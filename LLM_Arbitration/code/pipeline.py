"""
One negotiation -> classify every turn -> run ALL THREE arbitration
mechanisms against that SAME transcript, so the comparison is fair
(only the arbitration method differs, nothing else).

Later (Step 2), this whole function gets called in a loop across multiple
scenarios/repeated trials to reintroduce controlled randomness and check
how consistent each mechanism is across many different negotiations.
"""
import asyncio
import re
from datasets import load_dataset
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from experiment_utils import make_ollama_client, describe_priorities, transcript_to_text, save_log

VALID_LABELS = ["agree", "disagree", "stall", "escalate"]


async def call_with_retry(agent: AssistantAgent, task: str, max_retries: int = 4, base_delay: int = 8):
    """Wraps agent.run() with automatic retry + wait when we hit a rate limit.
    Groq's free tier has a small tokens-per-minute cap, so this is essential
    for anything beyond a single short call."""
    for attempt in range(max_retries):
        try:
            return await agent.run(task=task)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                wait = base_delay * (attempt + 1)
                print(f"    [Rate limited -- waiting {wait}s, retry {attempt + 1}/{max_retries}]")
                await asyncio.sleep(wait)
                continue
            raise  # not a rate-limit error -- don't hide real bugs
    raise RuntimeError(f"Still rate-limited after {max_retries} retries.")


def truncate_context(context: str, max_chars: int = 1200) -> str:
    """Keep only the most recent part of the conversation as context.
    Without this, prompts grow with every turn and eventually blow past
    Groq's free-tier per-minute token limit on longer negotiations."""
    if len(context) <= max_chars:
        return context
    return "...(earlier turns omitted for length)...\n" + context[-max_chars:]


async def classify_turn(classifier: AssistantAgent, prior_context: str, speaker: str, text: str, is_first: bool) -> tuple[str, str]:
    prior_context = truncate_context(prior_context)
    opener_note = (
        "\nNOTE: This is the FIRST substantive message -- there is no prior proposal to react to. "
        "Pick whichever of the four labels is the least misleading opening description.\n"
        if is_first else ""
    )
    prompt = (
        f"Conversation so far:\n{prior_context if prior_context else '(this is the first message)'}\n"
        f"{opener_note}\n"
        f"New message from {speaker}: \"{text}\"\n\n"
        "Follow these steps IN ORDER, showing your work:\n\n"
        "STEP 1: What exact terms/numbers (if any) were on the table BEFORE this new message?\n"
        "STEP 2: Does the message contain a clear ACCEPTANCE PHRASE committing to a deal? "
        "Examples: \"I agree\", \"I'll accept\", \"alright, deal\", \"sounds good\", \"let's finalize\", "
        "even if reluctant or accompanied by joking/teasing asides.\n"
        "STEP 3: If YES to Step 2, check: does the message ALSO attach a genuinely NEW, "
        "previously-undiscussed condition (not just reworded terms or banter)?\n"
        "  - No new condition -> 'agree'.  - New condition attached -> 'disagree' (conditional acceptance).\n"
        "STEP 4: If NO to Step 2, classify normally: counters/rejects/questions -> disagree; "
        "repeats/delays with nothing new -> stall; hardens position/tough new demand, no acceptance -> escalate.\n\n"
        "Tone (joking, sighing, teasing) is NOT a substantive signal.\n\n"
        "End your reply with a final line in EXACTLY this format (nothing after it):\n"
        "LABEL: <one word>"
    )
    result = await call_with_retry(classifier, prompt)
    full_text = result.messages[-1].content.strip()

    if not full_text:
        # Empty response from the API (rare, but happens) -- don't crash, just
        # record it as unclear so the run continues.
        return "unclear", "(empty response from model)"

    match = re.search(r"LABEL[:\s]+\**\s*(\w+)", full_text, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().lower()
        if candidate in VALID_LABELS:
            return candidate, full_text

    non_empty_lines = [ln for ln in full_text.splitlines() if ln.strip()]
    if not non_empty_lines:
        return "unclear", full_text
    last_line = non_empty_lines[-1].lower()
    for label in ["disagree", "escalate", "stall", "agree"]:
        if label in last_line:
            return label, full_text
    return "unclear", full_text


async def run_voting(camper_a: AssistantAgent, camper_b: AssistantAgent, transcript: str) -> tuple[str, str]:
    transcript = truncate_context(transcript, max_chars=2000)
    question = (
        f"Here is the negotiation transcript:\n\n{transcript}\n\n"
        "Do you personally accept the final deal reached above, exactly as stated? "
        "Reply with ONLY one word: YES or NO."
    )
    result_a = await call_with_retry(camper_a, question)
    await asyncio.sleep(2)
    result_b = await call_with_retry(camper_b, question)
    vote_a = "yes" in result_a.messages[-1].content.strip().lower()
    vote_b = "yes" in result_b.messages[-1].content.strip().lower()
    verdict = "APPROVED" if (vote_a and vote_b) else "REJECTED"
    detail = f"Camper_A vote: {'YES' if vote_a else 'NO'}\nCamper_B vote: {'YES' if vote_b else 'NO'}\nVERDICT: {verdict}"
    return verdict, detail


async def run_structured(camper_a: AssistantAgent, camper_b: AssistantAgent, transcript: str) -> tuple[str, str]:
    """Structured turn-taking protocol: each party states a reasoned argument
    (in fixed order) BEFORE voting, unlike plain voting which asks immediately.
    This tests whether forcing deliberation changes the outcome."""
    transcript = truncate_context(transcript, max_chars=1500)
    # Round 1: fixed-order arguments
    arg_prompt_a = (
        f"Here is the negotiation transcript:\n\n{transcript}\n\n"
        "State your argument for whether the final deal is fair to you, in 2-3 sentences. "
        "This is a reasoned statement, not a yes/no answer yet."
    )
    result_arg_a = await call_with_retry(camper_a, arg_prompt_a)
    argument_a = result_arg_a.messages[-1].content.strip()
    await asyncio.sleep(2)

    arg_prompt_b = (
        f"Here is the negotiation transcript:\n\n{transcript}\n\n"
        f"Camper_A's stated view on the deal: \"{argument_a}\"\n\n"
        "Now state YOUR argument for whether the final deal is fair to you, in 2-3 sentences."
    )
    result_arg_b = await call_with_retry(camper_b, arg_prompt_b)
    argument_b = result_arg_b.messages[-1].content.strip()
    await asyncio.sleep(2)

    # Round 2: final vote, now with both arguments visible to both parties
    vote_prompt = (
        f"Negotiation transcript:\n\n{transcript}\n\n"
        f"Camper_A's argument: \"{argument_a}\"\n"
        f"Camper_B's argument: \"{argument_b}\"\n\n"
        "Given both arguments, do YOU vote to accept the final deal? Reply with ONLY one word: YES or NO."
    )
    result_vote_a = await call_with_retry(camper_a, vote_prompt)
    await asyncio.sleep(2)
    result_vote_b = await call_with_retry(camper_b, vote_prompt)
    vote_a = "yes" in result_vote_a.messages[-1].content.strip().lower()
    vote_b = "yes" in result_vote_b.messages[-1].content.strip().lower()
    verdict = "APPROVED" if (vote_a and vote_b) else "REJECTED"
    detail = (
        f"Camper_A argument: {argument_a}\n"
        f"Camper_B argument: {argument_b}\n"
        f"Camper_A final vote: {'YES' if vote_a else 'NO'}\n"
        f"Camper_B final vote: {'YES' if vote_b else 'NO'}\n"
        f"VERDICT: {verdict}"
    )
    return verdict, detail


async def run_judge(arbitrator: AssistantAgent, transcript: str) -> tuple[str, str]:
    transcript = truncate_context(transcript, max_chars=2000)
    result = await call_with_retry(
        arbitrator, f"Here is the negotiation transcript:\n\n{transcript}\n\nReview the final agreed split."
    )
    full_text = result.messages[-1].content
    verdict = "APPROVED" if "VERDICT: APPROVED" in full_text else "REJECTED"
    return verdict, full_text


async def run_trial(scenario_index: int = 0) -> dict:
    """One full trial: one negotiation, classified once, then judged by ALL
    THREE arbitration mechanisms against that same fixed transcript."""
    dataset = load_dataset("kchawla123/casino", split="train")
    scenario = dataset[scenario_index]
    p1 = scenario["participant_info"]["mturk_agent_1"]
    p2 = scenario["participant_info"]["mturk_agent_2"]
    p1_priorities = describe_priorities(p1["value2issue"], p1["value2reason"])
    p2_priorities = describe_priorities(p2["value2issue"], p2["value2reason"])

    model_client_a = make_ollama_client()
    model_client_b = make_ollama_client()
    model_client_clf = make_ollama_client()
    model_client_arb = make_ollama_client()

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
        name="Classifier", model_client=model_client_clf,
        system_message="You are a precise, methodical conversation-turn classifier. You always show your reasoning before a final LABEL line.",
    )
    arbitrator = AssistantAgent(
        name="Arbitrator", model_client=model_client_arb,
        system_message=(
            "You are a strict, methodical Arbitrator reviewing a negotiation transcript between two "
            "campers splitting 3 Food, 3 Water, and 3 Firewood packages in total.\n\n"
            "You MUST follow this exact procedure, in order, and show every step:\n\n"
            "STEP 1 -- EXTRACT: Write the FINAL number of each item each camper ends up with. Format:\n"
            "  Food:     Camper_A = ?, Camper_B = ?\n"
            "  Water:    Camper_A = ?, Camper_B = ?\n"
            "  Firewood: Camper_A = ?, Camper_B = ?\n"
            "If unclear or contradictory, write 'UNCLEAR' instead of guessing.\n\n"
            "STEP 2 -- SUM: For each item, add the two numbers and show the total explicitly.\n\n"
            "STEP 3 -- CHECK: For each item, state whether the total equals exactly 3 (YES/NO) and "
            "whether both numbers are whole numbers (YES/NO).\n\n"
            "STEP 4 -- VERDICT: Only after Steps 1-3:\n"
            "VERDICT: APPROVED (only if every item passed every check) or REJECTED\n"
            "REASON: reference the specific item(s)/numbers that failed, if any\n"
            "IF REJECTED, SUGGESTED FIX: a valid whole-number split for the failing item(s)\n\n"
            "Do not skip steps. Do not approve anything marked UNCLEAR or that failed a check."
        ),
    )

    # ---- Stage 1: ONE negotiation (this is the fixed transcript all 3 modes will judge) ----
    termination = MaxMessageTermination(max_messages=10)
    team = RoundRobinGroupChat([camper_a, camper_b], termination_condition=termination)
    print(f"=== [Scenario {scenario_index}] Running negotiation (once) ===\n")
    result = await team.run(
        task="Negotiate how to split the 3 Food, 3 Water, and 3 Firewood packages between you."
    )
    turns = [m for m in result.messages if getattr(m, "source", None) not in (None, "user")]
    for m in turns:
        print(f"{m.source}: {m.content}\n")
    transcript = transcript_to_text(result.messages)

    # ---- Stage 2: Classify every turn ONCE ----
    print("=== Classifying each turn ===\n")
    output_lines = []
    running_context = ""
    label_counts = {"agree": 0, "disagree": 0, "stall": 0, "escalate": 0, "unclear": 0}
    for i, m in enumerate(turns):
        label, reasoning = await classify_turn(classifier, running_context, m.source, m.content, is_first=(i == 0))
        label_counts[label] = label_counts.get(label, 0) + 1
        print(f"[{label.upper():9}] {m.source}: {m.content}")
        output_lines.append(f"[{label.upper():9}] {m.source}: {m.content}\n  reasoning: {reasoning}\n")
        running_context += f"{m.source}: {m.content}\n"
        await asyncio.sleep(2)  # pace requests to stay under Groq's free-tier rate limit

    # ---- Stage 3: run ALL FOUR arbitration mechanisms against the SAME transcript ----
    print("\n=== Running all 4 arbitration mechanisms on the SAME transcript ===\n")

    verdict_none = "N/A (no arbitration)"
    detail_none = "Baseline condition -- no review performed. Deal stands exactly as negotiated."
    print(f"[NONE]   {verdict_none}")

    verdict_voting, detail_voting = await run_voting(camper_a, camper_b, transcript)
    print(f"[VOTING] {verdict_voting}\n{detail_voting}\n")
    await asyncio.sleep(3)

    verdict_structured, detail_structured = await run_structured(camper_a, camper_b, transcript)
    print(f"[STRUCTURED] {verdict_structured}\n{detail_structured}\n")
    await asyncio.sleep(3)

    verdict_judge, detail_judge = await run_judge(arbitrator, transcript)
    print(f"[JUDGE]  {verdict_judge}\n{detail_judge}\n")

    # ---- Save everything in one combined log ----
    summary = (
        "\n=== Turn-label summary ===\n"
        f"Total turns: {len(turns)}\n" + "\n".join(f"{k}: {v}" for k, v in label_counts.items())
    )
    arbitration_block = (
        "\n=== ARBITRATION COMPARISON (same transcript, 4 mechanisms) ===\n"
        f"[NONE]   {verdict_none}\n{detail_none}\n\n"
        f"[VOTING] {verdict_voting}\n{detail_voting}\n\n"
        f"[STRUCTURED] {verdict_structured}\n{detail_structured}\n\n"
        f"[JUDGE]  {verdict_judge}\n{detail_judge}\n"
    )
    full_log = (
        f"=== SCENARIO INDEX: {scenario_index} ===\n\n"
        "=== NEGOTIATION TRANSCRIPT (fixed, used by all 3 modes) ===\n" + transcript + "\n\n"
        "=== CLASSIFIED TURNS ===\n" + "\n".join(output_lines) + "\n" + summary + "\n"
        + arbitration_block
    )
    log_path = save_log(full_log, prefix=f"trial_s{scenario_index}_all_modes")

    await model_client_a.close()
    await model_client_b.close()
    await model_client_clf.close()
    await model_client_arb.close()

    return {
        "scenario_index": scenario_index,
        "label_counts": label_counts,
        "verdicts": {"none": verdict_none, "voting": verdict_voting, "structured": verdict_structured, "judge": verdict_judge},
        "log_path": log_path,
    }


if __name__ == "__main__":
    outcome = asyncio.run(run_trial(scenario_index=0))
    print("\n=== Trial result ===")
    print(outcome)
