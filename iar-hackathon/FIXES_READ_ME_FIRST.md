# READ THIS BEFORE ADDING ANYTHING — Your results are currently broken

I audited the whole project. The **structure is good** — the IAR method, buffer,
task splits, and unlearning gate are correctly written and not wasted. But the
project currently produces **meaningless numbers**, and three bugs are the cause.
Adding new features on top of this would be decorating a house with no
foundation. Fix these first. All three fixes are written for you.

---

## Bug 1 (CRITICAL) — Embeddings are garbage → every accuracy near 1%

`embed.py` fed raw uint8 pixels (0–255) straight into ResNet with **no
normalisation and no resize to 224×224**. ImageNet models need inputs scaled to
[0,1] and normalised by ImageNet mean/std. The result: 73% of every embedding is
exactly zero, 148 of 512 dimensions are dead, values run up to 467, and the
embeddings carry almost no class information.

**Proof it's broken:** your `joint` training (the supposed *upper bound*) scores
**6%**. On 5-way tasks, random guessing is 20%. Every method — naive, joint,
random, herding, IAR — sits between 1% and 6%. The model is learning from noise.

**Fix:** replace `embed.py` with `embed_FIXED.py`, then re-run it. It adds the
missing preprocessing, batches the forward pass (224×224 is too big to hold at
once), and — importantly — runs a **linear probe sanity gate** at the end.

```bash
python embed_FIXED.py
```

**Do not proceed until the linear probe prints ≥ 0.40 (it should reach ~0.50–0.60).**
If it's still near 1%, something else is wrong and no downstream number is valid.

---

## Bug 2 (CRITICAL for your story) — The R1 headline plot shows nothing

Your entire unique claim is *"forgetting is predictable before training."* The
figure that's supposed to prove it (`plot_1_predicted_vs_forgetting` in
`figures.py`) actually plots **softmax temperature τ against forgetting**. τ is
fixed at 1.0 in every run, so this is a "correlation" over a single x value —
pure noise. The predicted risk scores you compute (`results/risk_scores.csv`)
are never joined to forgetting at all.

**Fix:** use `figure_R1_FIXED.py`. It joins each class's predicted risk to the
**real** per-task forgetting from the saved accuracy matrices and reports a
genuine Pearson r. Run it after `experiments.py`:

```bash
python figure_R1_FIXED.py
```

This is the number your whole project rests on. If, after fixing embeddings, r
is strong (say > 0.5), you have a real finding. Report whatever r actually is —
honesty is in the rubric.

---

## Bug 3 — `per_class_forgetting` returned initial accuracy, not forgetting

In the original `metrics.py`, `per_class_forgetting` returned `R[t,t]` (accuracy
right after learning a task) and called it forgetting. It's the wrong quantity,
and it feeds the R1 analysis.

**Fix:** replace `metrics.py` with `metrics_FIXED.py`. It adds a correct
`per_task_forgetting` (best-ever minus final) and rebuilds `per_class_forgetting`
on top of it. `figure_R1_FIXED.py` already imports from `metrics_FIXED`.

---

## Order of operations

```bash
# 1. fix embeddings and PASS the probe gate (do not skip the gate)
mv embed.py embed_OLD.py &&  mv embed_FIXED.py embed.py
python embed.py                 # must print linear probe >= 0.40

# 2. swap in fixed metrics
mv metrics.py metrics_OLD.py &&  mv metrics_FIXED.py metrics.py

# 3. re-run every experiment on the GOOD embeddings
python experiments.py           # regenerates results/results.csv + matrices

# 4. rebuild the headline figure correctly
python figure_R1_FIXED.py       # prints the real Pearson r

# 5. sanity check: joint should now be your HIGHEST accuracy, naive should
#    forget hard, replay methods should sit in between. If not, stop and look.
```

---

## What "done" looks like after the fix

- `joint` final accuracy is your **highest** number (should be well above naive).
- `naive` has the **highest** forgetting.
- `iar` >= `herding` >= `random` on retention (this is the claim you're testing —
  if IAR does NOT win, report that honestly; the R1 correlation is still your
  result).
- `figure_R1_FIXED.py` prints a real r over ~95 classes, not over one τ value.

---

## Only AFTER all of the above passes: add features

Do not add the unlearning spike, the second dataset, or new dashboard panels
until the five steps above produce sensible numbers. A project with correct
core results and no extras beats a project with ten features built on noise —
especially under relative grading, where one judge spotting that "joint = 6%"
collapses trust in everything you present.

The lesson worth carrying: **the plots rendered, so it looked finished — but it
wasn't.** That gap is exactly where AI-assisted projects lose. Your edge over the
other teams is that yours is verified and theirs probably isn't.
