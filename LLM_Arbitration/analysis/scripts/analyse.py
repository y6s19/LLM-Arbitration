"""Aggregate analysis of the 15 official trials + the reported re-run."""
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
BASE = str(ROOT / "logs" / "official_experiments") + "/"

cas = pd.read_csv(BASE + "experiment_summary_20260911_005812.csv")
gsm = pd.read_csv(BASE + "gsm8k_summary_20260911_020702.csv")
mwz = pd.read_csv(BASE + "multiwoz_summary_20260911_031840.csv")

cas["tier"] = "CaSiNo"; gsm["tier"] = "GSM8K"; mwz["tier"] = "MultiWOZ"
cols = ["tier", "voting", "structured", "judge", "agree", "disagree", "stall", "escalate", "unclear"]
all_df = pd.concat([cas[cols], gsm[cols], mwz[cols]], ignore_index=True)

print("=" * 70)
print("1. APPROVAL RATE PER MECHANISM (run 1, n=5 per tier)")
print("=" * 70)
tbl = []
for tier, g in all_df.groupby("tier", sort=False):
    row = {"tier": tier, "n": len(g)}
    for m in ["voting", "structured", "judge"]:
        row[m] = f"{(g[m]=='APPROVED').sum()}/{len(g)}"
    tbl.append(row)
row = {"tier": "ALL", "n": len(all_df)}
for m in ["voting", "structured", "judge"]:
    k = (all_df[m] == "APPROVED").sum()
    row[m] = f"{k}/{len(all_df)} ({100*k/len(all_df):.0f}%)"
tbl.append(row)
print(pd.DataFrame(tbl).to_string(index=False))

print()
print("=" * 70)
print("2. McNEMAR EXACT TESTS (paired: same transcript, different mechanism)")
print("=" * 70)
def mcnemar_exact(a, b, na, nb, df=all_df, indent=""):
    A = (df[a] == "APPROVED").values
    B = (df[b] == "APPROVED").values
    b_ = int((A & ~B).sum())
    c_ = int((~A & B).sum())
    n = b_ + c_
    p = 1.0 if n == 0 else binomtest(b_, n, 0.5).pvalue
    print(f"{indent}{na:>10} vs {nb:<11} discordant {b_}/{c_} (n={n})  exact p = {p:.4f}"
          + ("   SIGNIFICANT" if p < 0.05 else ""))
for a, b, na, nb in [("voting","structured","Voting","Structured"),
                     ("voting","judge","Voting","Judge"),
                     ("structured","judge","Structured","Judge")]:
    mcnemar_exact(a, b, na, nb)

print("\nPer tier, Voting vs Judge:")
for tier, g in all_df.groupby("tier", sort=False):
    mcnemar_exact("voting", "judge", tier, "Judge", df=g, indent="   ")

print()
print("=" * 70)
print("3. POOLED ACROSS BOTH BATCHES (run1 + reported re-run, n=30)")
print("=" * 70)
pooled = {"Voting": (26, 30), "Structured": (23, 30), "Judge": (9, 30)}
for m, (k, n) in pooled.items():
    ci = binomtest(k, n, 0.5).proportion_ci(0.95)
    print(f"{m:<12} {k:>2}/{n} = {100*k/n:5.1f}%   95% CI [{100*ci.low:.1f}%, {100*ci.high:.1f}%]")
print("\n(Pooled figures rely on the re-run SUMMARY files; raw re-run logs are absent.)")

print()
print("=" * 70)
print("4. TURN-LABEL DISTRIBUTION (105 classified turns)")
print("=" * 70)
lab = all_df[["tier","agree","disagree","stall","escalate","unclear"]].fillna(0)
per_tier = lab.groupby("tier", sort=False).sum()
per_tier["TOTAL"] = per_tier.sum(axis=1)
print(per_tier.to_string())
tot = per_tier.drop(columns="TOTAL").sum()
print("\nOverall:")
for k, v in tot.items():
    print(f"  {k:<9} {int(v):>3}  ({100*v/tot.sum():5.1f}%)")
print(f"  {'TOTAL':<9} {int(tot.sum()):>3}")

p = (tot / tot.sum()).values
pe = float((p**2).sum())
po = 0.83
print(f"\nObserved raw agreement po = {po:.2f}  (reported, n=30 hand-labelled)")
print(f"Chance agreement pe from marginals = {pe:.3f}")
print(f"=> Cohen's kappa approx = {(po-pe)/(1-pe):.3f}   (substantial: >0.61)")

print()
print("=" * 70)
print("5. CONVERGENCE: STRICT vs TOLERANT DEFINITION")
print("=" * 70)
conv = pd.read_csv(ROOT / "analysis" / "convergence_summary.csv")
def tolerant(seq):
    labs = seq.split(",")
    for i in range(len(labs)):
        tail = labs[i:]
        if tail[0] == "agree" and all(l in ("agree","stall","unclear") for l in tail):
            return i + 1
    return None
rows = []
for _, r in conv.iterrows():
    t = tolerant(r["labels_sequence"])
    rows.append({"tier": r["tier"], "file": r["file"][:24],
                 "strict": r["convergence_turn"],
                 "tolerant": t if t else "DID NOT CONVERGE"})
cdf = pd.DataFrame(rows)
print(cdf.to_string(index=False))
s_nc = (cdf["strict"] == "DID NOT CONVERGE").sum()
t_nc = (cdf["tolerant"] == "DID NOT CONVERGE").sum()
print(f"\nNon-convergence: strict {s_nc}/15 ({100*s_nc/15:.0f}%)"
      f"  ->  tolerant {t_nc}/15 ({100*t_nc/15:.0f}%)")

print()
print("=" * 70)
print("6. TRUNCATION EXPOSURE PER CONDITION (chars of transcript seen)")
print("=" * 70)
lens = {"CaSiNo": [1937,2274,2218,2153,1511],
        "GSM8K":  [1004,1027,10186,1730,3469],
        "MultiWOZ":[1537,1411,1284,1510,841]}
caps = {"CaSiNo": (2000,1500,2000), "GSM8K": (1500,1200,None), "MultiWOZ": (1500,1200,1500)}
for tier, L in lens.items():
    v, s, j = caps[tier]
    nv = sum(1 for x in L if x > v); ns = sum(1 for x in L if x > s)
    nj = "n/a (solves independently)" if j is None else sum(1 for x in L if x > j)
    print(f"{tier:<9} truncated -> Voting {nv}/5 (cap {v}), Structured {ns}/5 (cap {s}), Judge {nj}"
          + ("" if j is None else f" (cap {j})"))
print("\nStructured ALWAYS has the smallest cap -> it systematically sees less")
print("context than Voting/Judge. Mechanism and context length are confounded.")
