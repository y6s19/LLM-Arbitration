REPRODUCIBILITY RE-RUN
=======================
This folder contains a full independent re-run of all 15 official trials
(5 CaSiNo, 5 GSM8K, 5 MultiWOZ), executed separately from the original
batch to check whether the arbitration mechanism findings replicate.

KEY RESULT: the core finding replicated closely across both independent
batches (30 trials total):
  Voting:     87% approval in BOTH runs (identical)
  Structured: 80% (run 1) vs 73% (run 2)
  Judge:      33% (run 1) vs 27% (run 2)

NOTABLE FINDINGS FROM THIS RE-RUN:
1. On the GSM8K "house flipping" percentage problem, the Judge calculated
   the exact same INCORRECT answer ($25,000, true answer $70,000) in both
   independent runs, despite the solvers making different mistakes each
   time. This suggests a systematic, reproducible misunderstanding of this
   problem type by the Judge, not random noise.

2. Counter-example to the main finding: in GSM8K trial 3 (this re-run),
   Solver_A introduced an off-topic bookshelf problem mid-conversation.
   Voting and Structured correctly rejected the resulting mismatched
   answer. The Judge, despite visibly confusing itself in its reasoning,
   approved anyway -- it verified the ARITHMETIC was locally consistent
   but did not notice the conversation had drifted off-topic. This shows
   the Judge is specifically strong at numeric consistency checks, not
   universally superior to simpler mechanisms.
