"""
Runs run_trial() across multiple CaSiNo scenarios, collecting results into
a CSV summary. This is your actual experiment data collection step.

Each scenario_index gives a genuinely different real negotiation situation
(different people, different priorities) -- this is where controlled
randomness comes back in, on top of the fair same-transcript comparison
already built into run_trial() itself.
"""
import asyncio
import csv
import os
from datetime import datetime

from pipeline import run_trial

# Which real CaSiNo scenarios to use. Increase this list once you've
# confirmed a small batch works cleanly (start small -- each one takes a
# few minutes and uses your free Groq rate limit).
SCENARIO_INDICES = [0, 1, 2, 3, 4]


async def main():
    results = []
    for idx in SCENARIO_INDICES:
        print(f"\n{'='*60}\nTRIAL: scenario {idx}\n{'='*60}\n")
        try:
            outcome = await run_trial(scenario_index=idx)
            results.append(outcome)
        except Exception as e:
            print(f"!! Trial for scenario {idx} failed: {e}")
            results.append({
                "scenario_index": idx,
                "label_counts": {},
                "verdicts": {"none": "ERROR", "voting": "ERROR", "structured": "ERROR", "judge": "ERROR"},
                "log_path": "",
                "error": str(e),
            })
        print("\n[Pausing 10s between scenarios to stay within Groq's free-tier rate limit...]")
        await asyncio.sleep(10)

    # ---- Save a CSV summary: one row per scenario, columns per mechanism ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(os.path.dirname(__file__), "logs", f"experiment_summary_{timestamp}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario_index", "none", "voting", "structured", "judge",
            "agree", "disagree", "stall", "escalate", "unclear", "log_path"
        ])
        for r in results:
            lc = r.get("label_counts", {})
            v = r.get("verdicts", {})
            writer.writerow([
                r["scenario_index"], v.get("none", ""), v.get("voting", ""),
                v.get("structured", ""), v.get("judge", ""),
                lc.get("agree", ""), lc.get("disagree", ""), lc.get("stall", ""),
                lc.get("escalate", ""), lc.get("unclear", ""), r.get("log_path", ""),
            ])

    print(f"\n\n=== EXPERIMENT COMPLETE ===")
    print(f"Summary CSV saved to: {csv_path}")

    # ---- Quick console summary: approval rate per mechanism ----
    print("\n=== Approval rate per mechanism (across all trials) ===")
    for mode in ["none", "voting", "structured", "judge"]:
        verdicts = [r["verdicts"].get(mode, "") for r in results]
        approved = sum(1 for v in verdicts if v == "APPROVED")
        total = sum(1 for v in verdicts if v in ("APPROVED", "REJECTED"))
        if total > 0:
            print(f"{mode:12} {approved}/{total} approved ({100*approved/total:.0f}%)")
        else:
            print(f"{mode:12} N/A (baseline has no approve/reject check)")


if __name__ == "__main__":
    asyncio.run(main())
