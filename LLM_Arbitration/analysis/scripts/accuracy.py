"""Score each mechanism against OBJECTIVE ground truth, including the baseline.

CaSiNo ground truth = does the final allocation satisfy the hard constraint
(each of Food/Water/Firewood sums to exactly 3, whole units)?  Hand-verified
from the five official transcripts:

 s0  A: 2FW,1F,'split' W  | B: 2FW,1F,'half' W  -> FW 2+2=4, F 1+1=2, W fractional  INVALID
 s1  A: 2FW,1F,1W         | B: 1F,2W            -> F 1+1=2, FW 2+0=2                 INVALID
 s2  A: 1W,1F,1FW         | B: 1W,1F,1FW        -> every item sums to 2, not 3       INVALID
 s3  never finalised; A's own statements contradict each other                       INVALID
 s4  A: 1F,1W,3FW         | B: 2F,1W,1FW        -> W 1+1=2, FW 3+1=4                 INVALID

GSM8K ground truth = the dataset's '#### n' answer (already in the summary CSV).
MultiWOZ excluded: no objective ground truth, and 2/5 pairs were degenerate.
"""
import pandas as pd
from scipy.stats import binomtest

# correct verdict for each trial: True = deal/answer is sound, should be APPROVED
casino_sound = {0: False, 1: False, 2: False, 3: False, 4: False}
casino_verdicts = {           # from experiment_summary_20260911_005812.csv
    0: dict(voting="A", structured="A", judge="R"),
    1: dict(voting="A", structured="R", judge="R"),
    2: dict(voting="A", structured="A", judge="R"),
    3: dict(voting="R", structured="R", judge="R"),
    4: dict(voting="A", structured="R", judge="R"),
}
gsm_sound = {0: True, 1: True, 2: False, 3: True, 4: True}
gsm_verdicts = {              # from gsm8k_summary_20260911_020702.csv
    0: dict(voting="A", structured="A", judge="R"),   # judge REJECTED via malformed line
    1: dict(voting="A", structured="A", judge="A"),
    2: dict(voting="R", structured="A", judge="R"),
    3: dict(voting="A", structured="A", judge="A"),
    4: dict(voting="A", structured="A", judge="A"),
}

def score(sound, verdicts, judge_fix=False):
    rows = {}
    for mech in ["none", "voting", "structured", "judge"]:
        correct = []
        for i, ok in sound.items():
            if mech == "none":
                v = "A"                       # baseline never rejects, by construction
            else:
                v = verdicts[i][mech]
                if judge_fix and mech == "judge" and i == 0 and verdicts is gsm_verdicts:
                    v = "A"                   # parser artefact: judge's reasoning agreed
            correct.append((v == "A") == ok)
        rows[mech] = correct
    return rows

cas = score(casino_sound, casino_verdicts)
gsm = score(gsm_sound, gsm_verdicts)
gsm_fixed = score(gsm_sound, gsm_verdicts, judge_fix=True)

print("=" * 72)
print("VERDICT ACCURACY AGAINST OBJECTIVE GROUND TRUTH")
print("=" * 72)
print(f"{'mechanism':<13}{'CaSiNo':>12}{'GSM8K':>12}{'COMBINED':>14}")
print("-" * 72)
for m in ["none", "voting", "structured", "judge"]:
    c, g = sum(cas[m]), sum(gsm[m])
    lbl = "none (baseline)" if m == "none" else m
    print(f"{lbl:<13}{c}/5 ({100*c/5:3.0f}%){'':>2}{g}/5 ({100*g/5:3.0f}%){'':>2}"
          f"{c+g}/10 ({100*(c+g)/10:3.0f}%)")
print("-" * 72)
jf = sum(gsm_fixed["judge"])
print(f"{'judge*':<13}{'5/5 (100%)':>12}{f'{jf}/5 ({100*jf/5:.0f}%)':>12}"
      f"{f'{5+jf}/10 ({100*(5+jf)/10:.0f}%)':>14}")
print("* correcting the one GSM8K p0 verdict-parsing artefact (Judge's own")
print("  working reached 18, matching the proposal, but its final line was malformed)")

print()
print("=" * 72)
print("FALSE APPROVALS (accepting an unsound deal/answer) - the costly error")
print("=" * 72)
unsound_cas = [i for i, ok in casino_sound.items() if not ok]
unsound_gsm = [i for i, ok in gsm_sound.items() if not ok]
for m in ["none", "voting", "structured", "judge"]:
    fa = sum(1 for i in unsound_cas if (casino_verdicts[i][m] if m != "none" else "A") == "A")
    fa += sum(1 for i in unsound_gsm if (gsm_verdicts[i][m] if m != "none" else "A") == "A")
    n = len(unsound_cas) + len(unsound_gsm)
    lbl = "none (baseline)" if m == "none" else m
    print(f"{lbl:<16} {fa}/{n} unsound cases wrongly approved ({100*fa/n:3.0f}%)")

print()
print("=" * 72)
print("McNEMAR: each mechanism vs the NO-ARBITRATION BASELINE (n=10 paired)")
print("=" * 72)
base = cas["none"] + gsm["none"]
for m in ["voting", "structured", "judge"]:
    mm = cas[m] + gsm[m]
    b_ = sum(1 for x, y in zip(mm, base) if x and not y)   # mech right, baseline wrong
    c_ = sum(1 for x, y in zip(mm, base) if not x and y)   # baseline right, mech wrong
    n = b_ + c_
    p = 1.0 if n == 0 else binomtest(b_, n, 0.5).pvalue
    print(f"{m:<12} improves {b_}, worsens {c_}  exact p = {p:.4f}"
          + ("   SIGNIFICANT" if p < 0.05 else ""))
mmj = cas["judge"] + gsm_fixed["judge"]
b_ = sum(1 for x, y in zip(mmj, base) if x and not y)
c_ = sum(1 for x, y in zip(mmj, base) if not x and y)
p = binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
print(f"{'judge*':<12} improves {b_}, worsens {c_}  exact p = {p:.4f}"
      + ("   SIGNIFICANT" if p < 0.05 else ""))
