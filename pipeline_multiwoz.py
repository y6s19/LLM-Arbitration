"""
Medium-conflict task tier: real MultiWOZ trip-planning preferences.
Two travelers, each with a REAL stated preference (taken verbatim from a
different real MultiWOZ dialogue's opening user turn), must agree on ONE
shared choice (e.g. a restaurant) that works for both of them.

Unlike CaSiNo (numeric split) or GSM8K (objectively correct number), here
"validity" is qualitative: does the final agreed choice actually address
BOTH travelers' stated real constraints? The Judge checks this narratively.
"""
import asyncio
from datasets import load_dataset
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from experiment_utils import make_ollama_client, transcript_to_text, save_log
from pipeline import classify_turn, call_with_retry, truncate_context


DOMAIN_KEYWORDS = {
    "restaurant": ["restaurant", "eat", "dine", "dining", "food", "cuisine", "meal", "lunch", "dinner"],
}


def get_real_preference(dataset, index: int, domain: str = "restaurant") -> str:
    """Pull a real user utterance that's actually ABOUT the given domain --
    not just the first turn of a dialogue that happens to mention the domain
    somewhere else. Falls back to the first user turn if no keyword match found."""
    keywords = DOMAIN_KEYWORDS.get(domain, [domain])
    tries = 0
    idx = index
    while tries < 50:
        item = dataset[idx % len(dataset)]
        if domain in item["services"]:
            user_turns = [
                item["turns"]["utterance"][i]
                for i, speaker in enumerate(item["turns"]["speaker"])
                if speaker == 0
            ]
            # Prefer a turn that actually mentions the domain
            for turn in user_turns:
                if any(kw in turn.lower() for kw in keywords):
                    return turn
            # Fallback: no keyword match found, but domain is in services --
            # use the first user turn anyway rather than skipping the dialogue.
            if user_turns:
                return user_turns[0]
        idx += 1
        tries += 1
    return f"I need a {domain}, no other preferences stated."  # should rarely trigger


async def run_voting_mwoz(trav_a, trav_b, transcript: str) -> tuple[str, str]:
    q = (
        f"Transcript:\n\n{truncate_context(transcript, 1500)}\n\n"
        "Do you personally accept the final agreed plan above as satisfying your needs? "
        "Reply with ONLY one word: YES or NO."
    )
    r_a = await call_with_retry(trav_a, q)
    await asyncio.sleep(2)
    r_b = await call_with_retry(trav_b, q)
    vote_a = "yes" in r_a.messages[-1].content.strip().lower()
    vote_b = "yes" in r_b.messages[-1].content.strip().lower()
    verdict = "APPROVED" if (vote_a and vote_b) else "REJECTED"
    return verdict, f"Traveler_A vote: {'YES' if vote_a else 'NO'}\nTraveler_B vote: {'YES' if vote_b else 'NO'}\nVERDICT: {verdict}"


async def run_structured_mwoz(trav_a, trav_b, transcript: str) -> tuple[str, str]:
    transcript = truncate_context(transcript, 1200)
    p_a = f"Transcript:\n\n{transcript}\n\nBriefly (2-3 sentences): does the final plan meet YOUR stated needs?"
    r_a = await call_with_retry(trav_a, p_a)
    arg_a = r_a.messages[-1].content.strip()
    await asyncio.sleep(2)
    p_b = f"Transcript:\n\n{transcript}\n\nTraveler_A's view: \"{arg_a}\"\n\nNow state YOUR view (2-3 sentences)."
    r_b = await call_with_retry(trav_b, p_b)
    arg_b = r_b.messages[-1].content.strip()
    await asyncio.sleep(2)
    vote_prompt = f"Traveler_A: {arg_a}\nTraveler_B: {arg_b}\n\nDo you vote to accept the plan? Reply ONLY YES or NO."
    rv_a = await call_with_retry(trav_a, vote_prompt)
    await asyncio.sleep(2)
    rv_b = await call_with_retry(trav_b, vote_prompt)
    vote_a = "yes" in rv_a.messages[-1].content.strip().lower()
    vote_b = "yes" in rv_b.messages[-1].content.strip().lower()
    verdict = "APPROVED" if (vote_a and vote_b) else "REJECTED"
    detail = f"Traveler_A argument: {arg_a}\nTraveler_B argument: {arg_b}\nTraveler_A vote: {'YES' if vote_a else 'NO'}\nTraveler_B vote: {'YES' if vote_b else 'NO'}\nVERDICT: {verdict}"
    return verdict, detail


async def run_judge_mwoz(judge, pref_a: str, pref_b: str, transcript: str) -> tuple[str, str]:
    transcript = truncate_context(transcript, 1500)
    prompt = (
        f"Traveler_A's real stated preference: \"{pref_a}\"\n"
        f"Traveler_B's real stated preference: \"{pref_b}\"\n\n"
        f"Negotiation transcript:\n\n{transcript}\n\n"
        "Does the FINAL agreed plan genuinely address BOTH travelers' stated preferences above? "
        "Check specifically for any preference (price, area, food type, etc.) that was stated but then "
        "ignored or contradicted in the final plan. Explain your check, then end with exactly one line:\n"
        "VERDICT: APPROVED\nor\nVERDICT: REJECTED (missed: <what was ignored>)"
    )
    result = await call_with_retry(judge, prompt)
    full_text = result.messages[-1].content
    verdict = "APPROVED" if "VERDICT: APPROVED" in full_text else "REJECTED"
    return verdict, full_text


async def run_trial_mwoz(pair_index: int = 0) -> dict:
    dataset = load_dataset("tuetschek/multi_woz_v22")["train"]
    pref_a = get_real_preference(dataset, pair_index * 2, domain="restaurant")
    pref_b = get_real_preference(dataset, pair_index * 2 + 1, domain="restaurant")

    model_client_a = make_ollama_client()
    model_client_b = make_ollama_client()
    model_client_clf = make_ollama_client()
    model_client_judge = make_ollama_client()

    trav_a = AssistantAgent(
        name="Traveler_A", model_client=model_client_a,
        system_message=(
            f"You are Traveler_A, planning a trip with a companion. Your real stated need: \"{pref_a}\"\n"
            "Keep replies short (2-3 sentences). Work with your companion to agree on ONE restaurant choice "
            "that works for you both."
        ),
    )
    trav_b = AssistantAgent(
        name="Traveler_B", model_client=model_client_b,
        system_message=(
            f"You are Traveler_B, planning a trip with a companion. Your real stated need: \"{pref_b}\"\n"
            "Keep replies short (2-3 sentences). Work with your companion to agree on ONE restaurant choice "
            "that works for you both."
        ),
    )
    classifier = AssistantAgent(
        name="Classifier", model_client=model_client_clf,
        system_message="You are a precise, methodical conversation-turn classifier. You always show your reasoning before a final LABEL line.",
    )
    judge = AssistantAgent(
        name="Judge", model_client=model_client_judge,
        system_message="You are a careful, independent checker verifying whether agreed plans actually meet people's stated needs.",
    )

    print(f"=== [MultiWOZ pair {pair_index}] ===\nTraveler_A's real preference: {pref_a}\nTraveler_B's real preference: {pref_b}\n")

    termination = MaxMessageTermination(max_messages=8)
    team = RoundRobinGroupChat([trav_a, trav_b], termination_condition=termination)
    result = await team.run(task="Agree on ONE restaurant choice that works for both of you, given your stated needs.")
    turns = [m for m in result.messages if getattr(m, "source", None) not in (None, "user")]
    for m in turns:
        print(f"{m.source}: {m.content}\n")
    transcript = transcript_to_text(result.messages)

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
        await asyncio.sleep(2)

    print("\n=== Running all 4 arbitration mechanisms ===\n")
    verdict_none = "N/A (no arbitration)"
    print(f"[NONE] {verdict_none}")

    verdict_voting, detail_voting = await run_voting_mwoz(trav_a, trav_b, transcript)
    print(f"[VOTING] {verdict_voting}\n{detail_voting}\n")
    await asyncio.sleep(3)

    verdict_structured, detail_structured = await run_structured_mwoz(trav_a, trav_b, transcript)
    print(f"[STRUCTURED] {verdict_structured}\n{detail_structured}\n")
    await asyncio.sleep(3)

    verdict_judge, detail_judge = await run_judge_mwoz(judge, pref_a, pref_b, transcript)
    print(f"[JUDGE] {verdict_judge}\n{detail_judge}\n")

    summary = f"\n=== Turn-label summary ===\nTotal turns: {len(turns)}\n" + "\n".join(f"{k}: {v}" for k, v in label_counts.items())
    full_log = (
        f"=== MULTIWOZ PAIR {pair_index} ===\nTraveler_A preference: {pref_a}\nTraveler_B preference: {pref_b}\n\n"
        "=== TRANSCRIPT ===\n" + transcript + "\n\n"
        "=== CLASSIFIED TURNS ===\n" + "\n".join(output_lines) + summary + "\n\n"
        f"=== ARBITRATION ===\n[NONE] {verdict_none}\n\n[VOTING] {verdict_voting}\n{detail_voting}\n\n"
        f"[STRUCTURED] {verdict_structured}\n{detail_structured}\n\n[JUDGE] {verdict_judge}\n{detail_judge}\n"
    )
    log_path = save_log(full_log, prefix=f"mwoz_pair{pair_index}")

    await model_client_a.close()
    await model_client_b.close()
    await model_client_clf.close()
    await model_client_judge.close()

    return {
        "pair_index": pair_index, "pref_a": pref_a, "pref_b": pref_b,
        "label_counts": label_counts,
        "verdicts": {"none": verdict_none, "voting": verdict_voting, "structured": verdict_structured, "judge": verdict_judge},
        "log_path": log_path,
    }


if __name__ == "__main__":
    outcome = asyncio.run(run_trial_mwoz(pair_index=0))
    print("\n=== Trial result ===")
    print(outcome)
