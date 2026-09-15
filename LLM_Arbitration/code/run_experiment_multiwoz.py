import asyncio
import csv
import os
from datetime import datetime

from pipeline_multiwoz import run_trial_mwoz

PAIR_INDICES = [0, 1, 2, 3, 4]


async def main():
    results = []
    for idx in PAIR_INDICES:
        print(f"\n{'='*60}\nMULTIWOZ TRIAL: pair {idx}\n{'='*60}\n")
        try:
            outcome = await run_trial_mwoz(pair_index=idx)
            results.append(outcome)
        except Exception as e:
            print(f"!! Trial for pair {idx} failed: {e}")
            results.append({
                "pair_index": idx, "pref_a": "", "pref_b": "", "label_counts": {},
                "verdicts": {"none": "ERROR", "voting": "ERROR", "structured": "ERROR", "judge": "ERROR"},
                "log_path": "",
            })
        print("\n[Pausing 10s between pairs...]")
        await asyncio.sleep(10)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(os.path.dirname(__file__), "logs", f"multiwoz_summary_{timestamp}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair_index", "pref_a", "pref_b", "none", "voting", "structured", "judge",
            "agree", "disagree", "stall", "escalate", "unclear", "log_path"
        ])
        for r in results:
            lc = r.get("label_counts", {})
            v = r.get("verdicts", {})
            writer.writerow([
                r["pair_index"], r.get("pref_a", ""), r.get("pref_b", ""),
                v.get("none", ""), v.get("voting", ""), v.get("structured", ""), v.get("judge", ""),
                lc.get("agree", ""), lc.get("disagree", ""), lc.get("stall", ""),
                lc.get("escalate", ""), lc.get("unclear", ""), r.get("log_path", ""),
            ])

    print(f"\n\n=== MULTIWOZ EXPERIMENT COMPLETE ===\nSummary CSV: {csv_path}")
    print("\n=== Approval rate per mechanism ===")
    for mode in ["none", "voting", "structured", "judge"]:
        verdicts = [r["verdicts"].get(mode, "") for r in results]
        approved = sum(1 for v in verdicts if v == "APPROVED")
        total = sum(1 for v in verdicts if v in ("APPROVED", "REJECTED"))
        if total > 0:
            print(f"{mode:12} {approved}/{total} approved ({100*approved/total:.0f}%)")
        else:
            print(f"{mode:12} N/A")


if __name__ == "__main__":
    asyncio.run(main())
