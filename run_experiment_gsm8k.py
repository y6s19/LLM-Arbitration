"""
Runs run_trial_gsm8k() across multiple real GSM8K problems, saving a CSV
summary that includes both arbitration verdicts AND actual task accuracy
(since GSM8K has real ground-truth answers, unlike the negotiation task).
"""
import asyncio
import csv
import os
from datetime import datetime

from pipeline_gsm8k import run_trial_gsm8k

PROBLEM_INDICES = [0, 1, 2, 3, 4]


async def main():
    results = []
    for idx in PROBLEM_INDICES:
        print(f"\n{'='*60}\nGSM8K TRIAL: problem {idx}\n{'='*60}\n")
        try:
            outcome = await run_trial_gsm8k(problem_index=idx)
            results.append(outcome)
        except Exception as e:
            print(f"!! Trial for problem {idx} failed: {e}")
            results.append({
                "problem_index": idx, "ground_truth": "", "proposed_answer": "",
                "task_correct": "ERROR", "label_counts": {},
                "verdicts": {"none": "ERROR", "voting": "ERROR", "structured": "ERROR", "judge": "ERROR"},
                "log_path": "",
            })
        print("\n[Pausing 10s between problems...]")
        await asyncio.sleep(10)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(os.path.dirname(__file__), "logs", f"gsm8k_summary_{timestamp}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "problem_index", "ground_truth", "proposed_answer", "task_correct",
            "none", "voting", "structured", "judge",
            "agree", "disagree", "stall", "escalate", "unclear", "log_path"
        ])
        for r in results:
            lc = r.get("label_counts", {})
            v = r.get("verdicts", {})
            writer.writerow([
                r["problem_index"], r.get("ground_truth", ""), r.get("proposed_answer", ""),
                r.get("task_correct", ""), v.get("none", ""), v.get("voting", ""),
                v.get("structured", ""), v.get("judge", ""),
                lc.get("agree", ""), lc.get("disagree", ""), lc.get("stall", ""),
                lc.get("escalate", ""), lc.get("unclear", ""), r.get("log_path", ""),
            ])

    print(f"\n\n=== GSM8K EXPERIMENT COMPLETE ===\nSummary CSV: {csv_path}")

    correct_count = sum(1 for r in results if r.get("task_correct") is True)
    total = len(results)
    print(f"\nTask accuracy: {correct_count}/{total} correct ({100*correct_count/total:.0f}%)")

    print("\n=== Approval rate per mechanism ===")
    for mode in ["none", "voting", "structured", "judge"]:
        verdicts = [r["verdicts"].get(mode, "") for r in results]
        approved = sum(1 for v in verdicts if v == "APPROVED")
        total_scoreable = sum(1 for v in verdicts if v in ("APPROVED", "REJECTED"))
        if total_scoreable > 0:
            print(f"{mode:12} {approved}/{total_scoreable} approved ({100*approved/total_scoreable:.0f}%)")
        else:
            print(f"{mode:12} N/A")


if __name__ == "__main__":
    asyncio.run(main())
