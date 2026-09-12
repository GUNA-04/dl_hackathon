# IAR / JANUS — Continual Learning Hackathon Project

Track 5 (Continual Learning). Interference-Aware Replay: predict which old
classes a new task will damage, and allocate a fixed replay buffer to them
before the damage happens.

## IMPORTANT — read FIXES_READ_ME_FIRST.md before running

This project had three bugs that made every result meaningless (all accuracies
were near 1%, including the joint-training upper bound). They are now fixed in
the code, and the broken embeddings.npy and stale results were removed so the
pipeline regenerates correctly. Details and reasoning are in
FIXES_READ_ME_FIRST.md. Short version:

1. embed.py — now applies ImageNet normalisation + resize to 224, and runs a
   linear-probe sanity gate. The probe must print >= 0.40 (expect ~0.5-0.6).
2. metrics.py — now computes real per-task forgetting (was returning initial
   accuracy).
3. figure_R1_FIXED.py — the corrected headline plot (predicted risk vs. actual
   forgetting). figures.py calls it automatically.

Previous versions are kept as embed_OLD.py and metrics_OLD.py for reference;
they are not used.

## How to run

    pip install -r requirements.txt
    python run_all.py     # embed -> experiments -> figures, one command

Stop and check after the embedding step: if the linear probe is near 1%, the
embeddings are still broken and nothing downstream is valid. It should be well
above chance.

## What "correct" looks like after running

- joint is your HIGHEST final accuracy (it's the upper bound).
- naive forgets the most.
- iar >= herding >= random on retention (the claim under test — if it doesn't
  hold, report that honestly; the R1 correlation is still a result).

## Only add new features after the core numbers are correct

A project with correct core results and no extras beats a project with ten
features built on noise — especially under relative grading.
