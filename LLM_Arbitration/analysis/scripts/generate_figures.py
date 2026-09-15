"""Generate every figure for the dissertation as editable SVG.

svg.fonttype='none' keeps text as real text elements, so labels remain editable
in Word, Inkscape or Illustrator rather than being converted to outlines.
Greyscale + hatching so the figures survive black-and-white printing.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT = str(Path(__file__).resolve().parent / "figures") + "/"
os.makedirs(OUT, exist_ok=True)

GREYS = ["#3d3d3d", "#767676", "#a5a5a5", "#cccccc", "#ececec"]
HATCH = ["", "//", "..", "xx", "\\\\"]
LABELS = ["agree", "disagree", "stall", "escalate", "unclear"]


def save(fig, name):
    fig.savefig(OUT + name, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("  ", name)


# ══════════════════════════════════════════════ Figure 4.2 — classifier iterations
versions = ["v1", "v2", "v3\n(first run)", "v3\n(revised)", "v4\n(adopted)"]
counts = {"agree": [9, 8, 0, 2, 7], "disagree": [0, 0, 0, 4, 2],
          "stall": [0, 1, 0, 0, 0], "escalate": [0, 0, 0, 3, 0],
          "unclear": [0, 0, 9, 0, 0]}

fig, ax = plt.subplots(figsize=(6.5, 3.4))
bottom = np.zeros(len(versions))
for lab, c, h in zip(LABELS, GREYS, HATCH):
    v = np.array(counts[lab], float)
    ax.bar(versions, v, bottom=bottom, label=lab, color=c,
           edgecolor="black", linewidth=0.6, hatch=h)
    for i, val in enumerate(v):
        if val:
            ax.text(i, bottom[i] + val / 2, int(val), ha="center", va="center",
                    fontsize=8, color="white" if lab == "agree" else "black")
    bottom += v
ax.set_ylabel("Turns labelled (of 9)")
ax.set_ylim(0, 9.6)
ax.set_title("Classifier label distribution across design iterations\n"
             "(same nine-turn negotiation transcript)")
ax.legend(ncol=5, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, -0.17), fontsize=8)
save(fig, "figure_4_2_classifier_iterations.svg")


# ══════════════════════════════════════════════ Figure 5.1 — label distribution
tiers = ["Factual\n(GSM8K)  n=25", "Planning\n(MultiWOZ)  n=35", "Negotiation\n(CaSiNo)  n=45"]
tally = {"agree": [14, 21, 28], "disagree": [1, 6, 10], "stall": [2, 4, 3],
         "escalate": [0, 0, 2], "unclear": [8, 4, 2]}
totals = np.array([25, 35, 45], float)

fig, ax = plt.subplots(figsize=(6.5, 3.2))
left = np.zeros(3)
for lab, c, h in zip(LABELS, GREYS, HATCH):
    pct = np.array(tally[lab], float) / totals * 100
    ax.barh(tiers, pct, left=left, label=lab, color=c,
            edgecolor="black", linewidth=0.6, hatch=h)
    for i, p in enumerate(pct):
        if p >= 6:
            ax.text(left[i] + p / 2, i, f"{p:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if lab == "agree" else "black")
    left += pct
ax.set_xlabel("Percentage of classified turns")
ax.set_xlim(0, 100)
ax.set_title("Turn-label distribution by task tier\n(tiers ordered by designed conflict potential)")
ax.legend(ncol=5, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, -0.22), fontsize=8)
save(fig, "figure_5_1_label_distribution.svg")


# ══════════════════════════════════════════════ Figure 5.2 — verdict accuracy
mechs = ["None\n(baseline)", "Voting", "Structured", "Judge"]
neg, fac, comb = [0, 20, 60, 100], [80, 100, 80, 80], [40, 60, 70, 90]

x = np.arange(4); w = 0.26
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.bar(x - w, neg,  w, label="Negotiation (n=5)", color=GREYS[0], edgecolor="black", linewidth=0.6)
ax.bar(x,     fac,  w, label="Factual (n=5)",     color=GREYS[2], edgecolor="black", linewidth=0.6, hatch="//")
ax.bar(x + w, comb, w, label="Combined (n=10)",   color=GREYS[4], edgecolor="black", linewidth=0.6, hatch="..")
for xi, vals in zip(x, zip(neg, fac, comb)):
    for off, v in zip([-w, 0, w], vals):
        ax.text(xi + off, v + 2.5, f"{v}%", ha="center", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(mechs)
ax.set_ylabel("Verdict accuracy (%)")
ax.set_ylim(0, 114)
ax.axhline(40, ls=":", lw=0.9, color="black", xmax=0.86)
ax.set_xlim(-0.55, 4.05)
ax.text(3.55, 38.5, "baseline\n(40%)", fontsize=7.5, style="italic", ha="left")
ax.set_title("Verdict accuracy against ground truth,\nby arbitration mechanism and task tier")
ax.legend(frameon=False, fontsize=8, loc="upper left")
save(fig, "figure_5_2_verdict_accuracy.svg")


# ══════════════════════════════════════════════ Figure 5.3 — per-trial verdict matrix
# correct = verdict matched ground truth.  MultiWOZ has no ground truth -> not scored.
trials = ["s0", "s1", "s2", "s3", "s4", "p0", "p1", "p2", "p3", "p4"]
conds = ["None (baseline)", "Voting", "Structured", "Judge"]
#             s0 s1 s2 s3 s4   p0 p1 p2 p3 p4
M = np.array([[0, 0, 0, 0, 0,  1, 1, 0, 1, 1],   # baseline
              [0, 0, 0, 1, 0,  1, 1, 1, 1, 1],   # voting
              [0, 1, 0, 1, 1,  1, 1, 0, 1, 1],   # structured
              [1, 1, 1, 1, 1,  0, 1, 1, 1, 1]])  # judge

fig, ax = plt.subplots(figsize=(6.7, 2.7))
ax.imshow(M, cmap=matplotlib.colors.ListedColormap(["#e0e0e0", "#4a4a4a"]),
          aspect="auto", vmin=0, vmax=1)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        col = "white" if M[i, j] else "#404040"
        if M[i, j]:                                  # tick
            ax.plot([j - 0.17, j - 0.04, j + 0.18], [i + 0.02, i + 0.17, i - 0.17],
                    color=col, lw=1.8, solid_capstyle="round")
        else:                                        # cross
            ax.plot([j - 0.15, j + 0.15], [i - 0.15, i + 0.15], color=col, lw=1.8,
                    solid_capstyle="round")
            ax.plot([j - 0.15, j + 0.15], [i + 0.15, i - 0.15], color=col, lw=1.8,
                    solid_capstyle="round")
ax.set_xticks(range(10)); ax.set_xticklabels(trials)
ax.set_yticks(range(4)); ax.set_yticklabels(conds)
ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)
for s in ax.spines.values():
    s.set_visible(False)
ax.axvline(4.5, color="black", lw=1.1)
ax.text(2.0, -0.92, "Negotiation (CaSiNo)", ha="center", fontsize=8.5, style="italic")
ax.text(7.0, -0.92, "Factual (GSM8K)", ha="center", fontsize=8.5, style="italic")
ax.set_title("Per-trial verdict correctness against ground truth\n"
             "(tick = correct verdict, cross = incorrect; planning tier omitted \u2014 no ground truth)",
             pad=26)
save(fig, "figure_5_3_verdict_matrix.svg")


# ══════════════════════════════════════════════ Figure 3.1 — paired design
fig, ax = plt.subplots(figsize=(6.6, 4.1))
ax.set_xlim(0, 10); ax.set_ylim(0, 7.0); ax.axis("off")

def box(x, y, w, h, text, fill="#f2f2f2", fs=8.5, bold=False, ls="-"):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor="black",
                               linewidth=0.9, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.45)

def arrow(x1, y1, x2, y2, lw=0.9):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=lw,
                                shrinkA=0, shrinkB=0))

# top row
box(0.15, 5.5, 2.5, 1.15, "Task scenario\nCaSiNo / GSM8K /\nMultiWOZ", "#ffffff", fs=8)
box(3.25, 5.5, 2.7, 1.15, "Two agents converse\nONCE\nasymmetric incentives", "#d6d6d6", bold=True, fs=8)
box(6.55, 5.5, 3.3, 1.15, "Fixed transcript\n+ per-turn labels", "#ffffff", fs=8.5)
arrow(2.65, 6.07, 3.25, 6.07)
arrow(5.95, 6.07, 6.55, 6.07)

# fan-out caption, kept clear of the arrows
ax.text(4.55, 4.72, "the same fixed transcript is reviewed four times",
        ha="center", fontsize=8, style="italic")

# four conditions
ys = [3.55, 2.65, 1.75, 0.85]
names = ["Baseline \u2014 no review",
         "Voting \u2014 both disputants assent",
         "Structured \u2014 argue, then vote",
         "Judge \u2014 independent arbitrator"]
fills = ["#ffffff", "#ececec", "#d4d4d4", "#a8a8a8"]
for y, n, f in zip(ys, names, fills):
    box(3.25, y, 4.4, 0.66, n, f, fs=8.5)
    arrow(8.20, 5.5, 7.72, y + 0.33)

# left annotation
box(0.10, 1.50, 2.85, 1.75,
    "Each transcript is\nits own control\n\n\u2192 repeated measures\n\u2192 McNemar exact test",
    "#ffffff", fs=7.4, ls="--")
arrow(2.95, 2.38, 3.25, 2.38)

ax.set_title("Paired same-transcript experimental design", fontsize=10.5, pad=2)
save(fig, "figure_3_1_paired_design.svg")


# ══════════════════════════════════════════════ Figure 4.1 — system architecture
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 8.6); ax.axis("off")

def layer(y, h, title, fill):
    ax.add_patch(plt.Rectangle((0.15, y), 9.7, h, facecolor=fill,
                               edgecolor="black", linewidth=1.0))
    ax.text(0.45, y + h - 0.30, title, fontsize=9, fontweight="bold", va="top")

# runner
layer(6.55, 1.85, "Experiment-runner layer", "#f7f7f7")
for nm, x, w in [("select dataset\nindices", 0.75, 2.6),
                 ("per-trial error\nhandling", 3.75, 2.4),
                 ("CSV summary +\ntrial logs", 6.55, 2.7)]:
    box(x, 6.72, w, 0.85, nm, "#ffffff", fs=8)

# pipelines
layer(2.75, 3.35, "Pipeline layer  \u2014  one module per task tier", "#efefef")
for nm, x in [("pipeline.py\nCaSiNo", 0.75), ("pipeline_gsm8k.py\nGSM8K", 3.75),
              ("pipeline_multiwoz.py\nMultiWOZ", 6.75)]:
    box(x, 4.45, 2.5, 0.85, nm, "#ffffff", fs=8)
box(0.75, 2.95, 8.5, 1.30,
    "run_trial()\n1  generate conversation   \u2192   2  classify every turn\n"
    "\u2192   3  apply all four arbitration conditions",
    "#d6d6d6", fs=8.5, bold=True)

# utilities
layer(0.15, 2.25, "Shared utility layer", "#f7f7f7")
for nm, x, w in [("model client\nOllama, llama3.1:8b", 0.75, 2.7),
                 ("transcript\nserialisation", 3.85, 2.3),
                 ("turn classifier\nv4, forced reasoning", 6.55, 2.7)]:
    box(x, 0.50, w, 1.05, nm, "#ffffff", fs=8)

for x in (2.05, 5.0, 7.9):
    arrow(x, 2.75, x, 2.45)
    arrow(x, 6.55, x, 6.20)

ax.set_title("Experimental harness architecture", fontsize=10.5, pad=2)
save(fig, "figure_4_1_architecture.svg")

print("\nAll figures written to", OUT)
