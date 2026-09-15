"""
Low-conflict task tier: real GSM8K math word problems.
Unlike the CaSiNo negotiation task, this has an OBJECTIVELY CORRECT answer,
so we can measure genuine task accuracy, not just arbitration approval.

Two solvers independently attempt the same problem, discuss if they differ,
and reach a final answer. All 4 arbitration mechanisms are tested against
that same fixed transcript, same as the negotiation pipeline.
"""
import asyncio
import re
from datasets import load_dataset
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from experiment_utils import make_ollama_client, transcript_to_text, save_log
from pipeline import classify_turn, call_with_retry, truncate_context, VALID_LABELS


def extract_ground_truth(answer_field: str) -> str:
    """GSM8K answers end with '#### <number>' -- pull that out as ground truth."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_field)
    return match.group(1).replace(",", "") if match else "UNKNOWN"


def extract_final_answer(text: str) -> str:
    """Find the LAST 'FINAL ANSWER: <number>' in a message, if present.
    Handles an optional currency symbol (e.g. '$18') between the label and the number."""
    matches = re.findall(r"FINAL ANSWER:\s*[$£€]?\s*(-?[\d,]+\.?\d*)", text, re.IGNORECASE)
    return matches[-1].replace(",", "") if matches else None


async def run_voting_gsm8k(solver_a, solver_b, transcript: str, proposed_answer: str) -> tuple[str, str]:
    question = (
        f"Transcript:\n\n{truncate_context(transcript, 1500)}\n\n"
        f"The proposed final answer is {proposed_answer}. Do you personally accept this as correct? "
        "Reply with ONLY one word: YES or NO."
    )
    result_a = await call_with_retry(solver_a, question)
    await asyncio.sleep(2)
    result_b = await call_with_retry(solver_b, question)
    vote_a = "yes" in result_a.messages[-1].content.strip().lower()
    vote_b = "yes" in result_b.messages[-1].content.strip().lower()
    verdict = "APPROVED" if (vote_a and vote_b) else "REJECTED"
    return verdict, f"Solver_A vote: {'YES' if vote_a else 'NO'}\nSolver_B vote: {'YES' if vote_b else 'NO'}\nVERDICT: {verdict}"


async def run_structured_gsm8k(solver_a, solver_b, transcript: str, proposed_answer: str) -> tuple[str, str]:
    transcript = truncate_context(transcript, 1200)
    arg_prompt_a = (
        f"Transcript:\n\n{transcript}\n\nProposed final answer: {proposed_answer}\n\n"
        "Briefly (2-3 sentences) state whether you believe this answer is correct and why."
    )
    r_a = await call_with_retry(solver_a, arg_prompt_a)
    argument_a = r_a.messages[-1].content.strip()
    await asyncio.sleep(2)
    arg_prompt_b = (
        f"Transcript:\n\n{transcript}\n\nProposed final answer: {proposed_answer}\n\n"
        f"Solver_A's view: \"{argument_a}\"\n\nNow state YOUR view (2-3 sentences)."
    )
    r_b = await call_with_retry(solver_b, arg_prompt_b)
    argument_b = r_b.messages[-1].content.strip()
    await asyncio.sleep(2)
    vote_prompt = (
        f"Proposed answer: {proposed_answer}\nSolver_A's view: {argument_a}\nSolver_B's view: {argument_b}\n\n"
        "Given both views, do you vote to accept this as the final answer? Reply with ONLY one word: YES or NO."
    )
    rv_a = await call_with_retry(solver_a, vote_prompt)
    await asyncio.sleep(2)
    rv_b = await call_with_retry(solver_b, vote_prompt)
    vote_a = "yes" in rv_a.messages[-1].content.strip().lower()
    vote_b = "yes" in rv_b.messages[-1].content.strip().lower()
    verdict = "APPROVED" if (vote_a and vote_b) else "REJECTED"
    detail = (
        f"Solver_A argument: {argument_a}\nSolver_B argument: {argument_b}\n"
        f"Solver_A vote: {'YES' if vote_a else 'NO'}\nSolver_B vote: {'YES' if vote_b else 'NO'}\nVERDICT: {verdict}"
    )
    return verdict, detail


async def run_judge_gsm8k(judge: AssistantAgent, question: str, proposed_answer: str) -> tuple[str, str]:
    """The Judge solves the problem INDEPENDENTLY (not just reading the transcript),
    then compares its own answer to what the solvers agreed on."""
    prompt = (
        f"Solve this math problem yourself, step by step:\n\n{question}\n\n"
        f"After solving, state your own answer, then compare it to this proposed answer: {proposed_answer}\n\n"
        "End with exactly one of these two lines:\n"
        "VERDICT: APPROVED\nor\nVERDICT: REJECTED (correct answer: <your number>)"
    )
    result = await call_with_retry(judge, prompt)
    full_text = result.messages[-1].content
    verdict = "APPROVED" if "VERDICT: APPROVED" in full_text else "REJECTED"
    return verdict, full_text


async def run_trial_gsm8k(problem_index: int = 0) -> dict:
    dataset = load_dataset("openai/gsm8k", "main")["test"]
    item = dataset[problem_index]
    question = item["question"]
    ground_truth = extract_ground_truth(item["answer"])

    model_client_a = make_ollama_client()
    model_client_b = make_ollama_client()
    model_client_clf = make_ollama_client()
    model_client_judge = make_ollama_client()

    solver_a = AssistantAgent(
        name="Solver_A", model_client=model_client_a,
        system_message=(
            "You are Solver_A. Solve math word problems step by step, showing your working briefly. "
            "Always end your message with exactly: FINAL ANSWER: <number only, no currency symbols, units, or commas>"
        ),
    )
    solver_b = AssistantAgent(
        name="Solver_B", model_client=model_client_b,
        system_message=(
            "You are Solver_B. Solve math word problems step by step, showing your working briefly. "
            "Always end your message with exactly: FINAL ANSWER: <number only, no currency symbols, units, or commas>"
        ),
    )
    classifier = AssistantAgent(
        name="Classifier", model_client=model_client_clf,
        system_message="You are a precise, methodical conversation-turn classifier. You always show your reasoning before a final LABEL line.",
    )
    judge = AssistantAgent(
        name="Judge", model_client=model_client_judge,
        system_message="You are a careful, independent maths checker. You always solve problems yourself before judging anyone else's answer.",
    )

    print(f"=== [GSM8K problem {problem_index}] Question ===\n{question}\n(Ground truth: {ground_truth})\n")

    termination = MaxMessageTermination(max_messages=6)
    team = RoundRobinGroupChat([solver_a, solver_b], termination_condition=termination)
    result = await team.run(
        task=f"Solve this problem together. Each state your own working and answer, then agree on one final answer:\n\n{question}"
    )
    turns = [m for m in result.messages if getattr(m, "source", None) not in (None, "user")]
    for m in turns:
        print(f"{m.source}: {m.content}\n")
    transcript = transcript_to_text(result.messages)

    # ---- Classification ----
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

    # ---- Determine the proposed final answer (last FINAL ANSWER: stated) ----
    proposed_answer = None
    for m in reversed(turns):
        fa = extract_final_answer(m.content)
        if fa:
            proposed_answer = fa
            break
    if proposed_answer is None:
        proposed_answer = "NO ANSWER STATED"
    task_correct = (proposed_answer == ground_truth)
    print(f"\nProposed final answer: {proposed_answer} | Ground truth: {ground_truth} | Correct: {task_correct}\n")

    # ---- Arbitration (all 4 mechanisms) ----
    print("=== Running all 4 arbitration mechanisms ===\n")
    verdict_none = "N/A (no arbitration)"
    print(f"[NONE] {verdict_none}")

    verdict_voting, detail_voting = await run_voting_gsm8k(solver_a, solver_b, transcript, proposed_answer)
    print(f"[VOTING] {verdict_voting}\n{detail_voting}\n")
    await asyncio.sleep(3)

    verdict_structured, detail_structured = await run_structured_gsm8k(solver_a, solver_b, transcript, proposed_answer)
    print(f"[STRUCTURED] {verdict_structured}\n{detail_structured}\n")
    await asyncio.sleep(3)

    verdict_judge, detail_judge = await run_judge_gsm8k(judge, question, proposed_answer)
    print(f"[JUDGE] {verdict_judge}\n{detail_judge}\n")

    summary = (
        f"\n=== Turn-label summary ===\nTotal turns: {len(turns)}\n"
        + "\n".join(f"{k}: {v}" for k, v in label_counts.items())
        + f"\n\nProposed answer: {proposed_answer} | Ground truth: {ground_truth} | Task correct: {task_correct}"
    )
    full_log = (
        f"=== GSM8K PROBLEM {problem_index} ===\nQuestion: {question}\nGround truth: {ground_truth}\n\n"
        "=== TRANSCRIPT ===\n" + transcript + "\n\n"
        "=== CLASSIFIED TURNS ===\n" + "\n".join(output_lines) + summary + "\n\n"
        f"=== ARBITRATION ===\n[NONE] {verdict_none}\n\n[VOTING] {verdict_voting}\n{detail_voting}\n\n"
        f"[STRUCTURED] {verdict_structured}\n{detail_structured}\n\n[JUDGE] {verdict_judge}\n{detail_judge}\n"
    )
    log_path = save_log(full_log, prefix=f"gsm8k_p{problem_index}")

    await model_client_a.close()
    await model_client_b.close()
    await model_client_clf.close()
    await model_client_judge.close()

    return {
        "problem_index": problem_index,
        "ground_truth": ground_truth,
        "proposed_answer": proposed_answer,
        "task_correct": task_correct,
        "label_counts": label_counts,
        "verdicts": {"none": verdict_none, "voting": verdict_voting, "structured": verdict_structured, "judge": verdict_judge},
        "log_path": log_path,
    }


if __name__ == "__main__":
    outcome = asyncio.run(run_trial_gsm8k(problem_index=0))
    print("\n=== Trial result ===")
    print(outcome)
