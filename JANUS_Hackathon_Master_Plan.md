# JANUS — Bidirectional Control of Catastrophic Forgetting
### The Complete Hackathon Master Plan

**Track:** 5 — Continual Learning
**Domain:** Continual visual recognition, extended to *controllable memory* (privacy / right-to-be-forgotten)
**Project name:** JANUS *(the two-faced god — looks both ways: prevents forgetting AND induces it on demand)*
**Primary dataset:** CIFAR-100, class-incremental — 20 tasks × 5 classes, seed 42
**Second dataset (generality):** ESC-50 audio — same method, loader-only change

---

## THE ONE-SENTENCE PITCH

> Every other team tries to *stop* forgetting. We discovered forgetting is *predictable before training* — and once you can predict it, you can control it in both directions: protect the knowledge that matters, and cleanly erase the knowledge you're legally or ethically required to remove, using the exact same mechanism.

---

## WHY THIS IS UNIQUE (say this to your teacher)

The other teams are all answering **one** question: *"can we reduce forgetting?"* Their projects will look nearly identical and relative grading will split them on tiny accuracy differences.

JANUS answers a question **no other team is asking**: *"can we control forgetting — both prevent it and cause it on purpose?"*

Three things make it a different category, not a better version:

1. **It reframes forgetting as a tool, not a failure.** Prevention and deliberate erasure are the same problem viewed from two sides. Nobody else will frame it this way, and the framing can't be copied cheaply because copying it means rebuilding their whole story.
2. **It's built on a discovery, not a benchmark.** We first *prove* forgetting is predictable from representational similarity (a measurable law), then exploit that law. Teams that report "we beat ER by X%" have a number; we have a finding — and judges remember findings.
3. **It connects to a real-world problem the others ignore: the right to be forgotten.** GDPR-style data deletion in a model that's still learning is an unsolved, high-value problem. Our "induce forgetting" direction is a toy but honest attempt at it. That real-world hook is a problem-quality and originality spike.

---

## THE SIX RESULTS (each is a plot, a claim, and a demo panel)

| # | Question | What it proves | Risk |
|---|---|---|---|
| R1 | Is forgetting predictable **before** training? | Predicted interference correlates with actual forgetting (report r). **This is the scientific spine — holds even if every method ties.** | Low |
| R2 | Can we **pre-empt** it? | Interference-Aware Replay allocates a fixed buffer by predicted risk → lower forgetting than random/herding ER | Low |
| R3 | Can we **induce** forgetting precisely? *(THE UNIQUE SPIKE)* | Targeted unlearning: remove ONE class to near-chance while keeping the rest intact — using the same interference signal in reverse | **Medium — must be proven, see gate** |
| R4 | Can our cheap predictor match the **expensive truth**? | Prototype-similarity proxy correlates with real gradient-interference | Low-Med |
| R5 | Does the principle **generalize**? | Same method works on images (CIFAR-100) AND audio (ESC-50) | Low |
| R6 | Can the system **explain itself**? | Per-class plain-language explanation of why it was kept/forgotten | Low |

**Golden rule:** R1 + R2 are your guaranteed floor (≈8.4). R3 is what makes you incomparable (≈9.5) — but it is the one result that must PROVE itself or be cut. R4–R6 are depth and demo. If anything wobbles, cut in this order: R4 → R6 → R5 → R3. Never cut R1 or R2.

---

## PART 1 — THE THREE RULES THAT PROTECT YOUR MARKS

Print these. Break one and you lose marks you cannot recover.

1. **Test split is touched ONCE, at the very end.** Never train, tune, or compute any risk/forgetting statistic on it. Carve a validation slice from training data for everything mid-run. *(Guards 15% rigor.)*
2. **Every replay method replays the SAME number of samples per step** — same buffer size AND same replay volume. Otherwise you compare replay *volume*, not replay *strategy*, and the result is meaningless.
3. **seed = 42, set once, globally** (torch, numpy, random, `cudnn.deterministic=True`). Headline number is seed 42; also report mean ± std over 5 seeds — the brief warns against lucky seeds, so error bars are cheap rigor marks.

---

## PART 2 — THE VERIFICATION DISCIPLINE (this is how you win a relative grade)

Using AI to write code does not make results *correct*. AI generates six panels in six turns that quietly contradict each other. Your edge over every other AI-using team is that **your numbers can't contradict each other and every result can be defended out loud.** Enforce three gates:

**Gate A — Single source of truth.** There is exactly ONE `results.csv`. Every plot, every table, every dashboard number reads from it. Nothing is hand-typed or generated ad hoc. If it's not in `results.csv`, it doesn't exist. This makes self-contradiction structurally impossible.

**Gate B — Defend-in-one-sentence.** Before any result is kept, one human must explain it in a single sentence with no notes. Example: *"Leopard was forgotten because lion and tiger sit within 0.1 cosine distance of it and it held only 2 buffer slots."* If nobody can explain it, CUT IT — an undefendable result is a liability a judge exposes in two questions.

**Gate C — Leakage statement.** The report explicitly says: risk/forgetting stats are computed on validation; test is touched once. Judges reward teams that flag their own discipline, and it signals every other number is trustworthy.

---

## PART 3 — REPO STRUCTURE

```
janus/
├── run_all.py              # ONE command reproduces everything
├── requirements.txt
├── README.md               # 3-line run instructions
├── seed.py                 # set_seed(42) — torch/numpy/random/cudnn
├── data.py                 # CIFAR-100 (+ ESC-50) -> images/labels
├── embed.py                # frozen ResNet-18 -> embeddings.npy  (RUN ONCE)
├── tasks.py                # 100 classes -> 20 tasks of 5, deterministic
├── continual.py            # training loop + accuracy matrix R[i][j]
├── metrics.py              # final avg accuracy + avg forgetting from R
├── interference.py         # prototypes, similarity, risk scores  (CORE)
├── buffer.py               # memory: random / herding / IAR allocation
├── unlearn.py              # TARGETED FORGETTING — the unique spike (R3)
├── gradient_check.py       # cheap proxy vs true gradient interference (R4)
├── experiments.py          # runs ALL conditions -> results.csv
├── explain.py              # per-class plain-language explanations (R6)
├── figures.py              # all plots from results.csv
├── app.py                  # Streamlit dashboard
└── results/
    ├── embeddings.npy
    ├── labels.npy
    └── results.csv          # SINGLE SOURCE OF TRUTH
```

### What each file does

- **seed.py** — one `set_seed(42)`, called at the top of everything.
- **data.py** — `torchvision.datasets.CIFAR100(download=True)`; ESC-50 loader added later. Returns raw images + labels only.
- **embed.py** — frozen ImageNet **ResNet-18** with the final layer removed. Resize CIFAR 32→224, push every image through ONCE, save `embeddings.npy` (N×512) + `labels.npy`. **The only slow step. Run once, never again.** Everything after is fast vector math.
- **tasks.py** — deterministically assign 100 classes to 20 ordered tasks of 5. Return per task: class list, and train/val/test index lists.
- **continual.py** — train a small classifier head task by task. After each task, evaluate on all tasks seen so far → fill accuracy matrix `R[i][j]` = accuracy on task *j* after training through task *i*. **Everything downstream comes from this one matrix.**
- **metrics.py** — final avg accuracy = `mean_j R[last][j]`; avg forgetting = `mean_j ( max_i R[i][j] − R[last][j] )`.
- **interference.py** *(CORE)* — class prototypes (mean embedding, L2-normalised); interference matrix `I[old,new] = cosine(proto_old, proto_new)`; per-class risk = max similarity to any incoming class.
- **buffer.py** — fixed capacity N, three modes: `random`, `herding` (nearest to class mean), `iar` (risk-proportional quota + herding within quota).
- **unlearn.py** *(SPIKE)* — see Part 5.
- **gradient_check.py** — for a subset, compute cosine between an old class's and a new class's loss gradients (true interference), correlate with the cheap prototype-similarity proxy.
- **experiments.py** — run every condition × memory size × seed × ablation cell → one tidy `results.csv`.
- **explain.py** — given a class, emit the one-sentence explanation from stored risk + quota + outcome.
- **figures.py** — read `results.csv`, write all figures.
- **app.py** — Streamlit; reads `results.csv`; panels in Part 8.
- **run_all.py** — embed → experiments → figures, so a fresh clone reproduces with `python run_all.py`.

---

## PART 4 — THE CORE METHOD (Interference-Aware Replay, R2)

Before training on task *t*:

```python
# the entire prevention mechanism
P_old = prototypes[old_classes]            # (C_old, 512) L2-normalised
P_new = prototypes[new_classes]            # (C_new, 512) L2-normalised
risk  = (P_old @ P_new.T).max(axis=1)      # (C_old,) predicted interference
quota = floor + (N - floor*C_old) * softmax(risk / tau)   # per-class budget
# keep the top quota[c] samples of class c nearest its prototype (herding)
```

Give every old class a floor so none starves; distribute the rest by risk. **Cap τ tuning at 20 minutes** — try 3 values on validation, pick one, move on.

---

## PART 5 — THE UNIQUE SPIKE: TARGETED FORGETTING (R3)

**The idea:** the same interference signal that tells us what to *protect* tells us what to *attack*. To make the model forget class *k* on request (a deletion request), we do the reverse of replay:

1. Remove all of class *k* from the buffer (stop rehearsing it).
2. Identify the classes most similar to *k* (its interferers) and replay *them* — deliberately using interference to overwrite *k*'s knowledge.
3. Optionally fine-tune the head briefly to push *k*'s logits down while keeping others fixed.

**The claim:** after unlearning, accuracy on *k* drops to near-chance while accuracy on all other classes stays essentially unchanged.

### THE R3 PROOF GATE — keep it or cut it, no middle ground

R3 stays in the project ONLY if, on the **test set**:
- Removed class accuracy falls to **≤ (chance + small margin)** — genuinely gone, and
- Average accuracy of all retained classes drops by **≤ ~2 points** — the rest survived.

Report both numbers as a before/after table. If you cannot show BOTH cleanly, **cut R3 entirely** — a fumbled unlearning claim is the one thing a sharp judge can puncture, and under relative grading a punctured claim loses to an honest modest one. Do not overclaim. An honest "we achieved partial unlearning with X% collateral" is fine; a fake "clean unlearning" is fatal.

---

## PART 6 — EXPERIMENTAL DESIGN

### Main comparison (prevention, R2)

| Condition | Buffer | Role |
|---|---|---|
| Naive sequential fine-tuning | 0 | **Mandatory lower bound** |
| Joint training | ∞ | **Mandatory upper bound** |
| Random ER | N | Standard comparison |
| Herding ER (uniform quota) | N | Strong comparison |
| **JANUS-Prevent (IAR)** | N | **Ours** |

### Ablation — 2×2 (isolate allocation from selection)

| Allocation | Selection | Tests |
|---|---|---|
| Uniform | Random | Plain ER |
| Uniform | Herding | Does selection help? |
| Risk-based | Random | Does allocation help? |
| Risk-based | Herding | Full IAR |

Every cell answers a distinct question — stronger than a cumulative ablation with redundant stages.

### Memory-budget sweep
N = 20, 50, 100. Small-N is where allocation matters most — your narrative: *"with a tiny buffer, ours remembers more."*

### Metrics
- Final average accuracy `A_T` (↑) and average forgetting `F_T` (↓)
- **R1 correlation:** predicted risk vs. measured per-class forgetting (headline)
- **R3 table:** removed-class and retained-class accuracy, before vs. after unlearning
- **R4 correlation:** prototype-similarity proxy vs. gradient-interference truth

### Seeds
Seed 42 = headline; seeds 42–46 = mean ± std. Runs are seconds, so this is nearly free.

---

## PART 7 — EXECUTION TIMELINE

Because AI writes the code, the clock cost shifts to **running, checking, and understanding**. Each block ends in a **GATE**. Fail a gate → take the fallback immediately.

| Time | Do | GATE / fallback |
|---|---|---|
| 0:00–0:20 | Start CIFAR-100 download in background. Repo skeleton, `seed.py`, empty `run_all.py`. Declare track + dataset to faculty. | Download running. |
| 0:20–1:00 | `embed.py` → `embeddings.npy`. | **Linear probe on 100 classes ≥ 50%.** Near chance → embeddings broken, fix NOW; nothing works downstream. |
| 1:00–1:45 | `tasks.py`, `continual.py` (accuracy matrix), then Naive + Joint. | **Naive forgets hard; Joint far above. 20% banked. Save matrices.** Weak forgetting → make tasks harder. |
| 1:45–2:30 | `buffer.py`: Random + Herding ER. Enforce equal replay volume. | Both beat Naive. Save numbers as you go — never re-run to recover. |
| 2:30–3:15 | `interference.py` + JANUS-Prevent (IAR, R2). | IAR runs end-to-end, produces a matrix. Cap τ tuning at 20 min. |
| 3:15–3:45 | **R1:** predicted risk vs. actual forgetting scatter + r. | Scatter shows a trend; report r honestly whatever it is. |
| 3:45–4:45 | **R3 (THE SPIKE):** `unlearn.py` + before/after test table. | **R3 PROOF GATE (Part 5). Passes → keep. Fails → CUT, move on, no debate.** |
| 4:45–5:15 | `experiments.py`: main × 5 seeds, memory sweep, 2×2 ablation → `results.csv`. | One `results.csv`, all numbers in it. |
| 5:15–5:45 | **R4:** `gradient_check.py`. **R6:** `explain.py`. | Both optional — skip either if behind. |
| 5:45–6:30 | `figures.py`: all plots. **R5:** ESC-50 second block IF ahead. | Headline R1 + forgetting + memory plots exist. |
| 6:30–7:15 | `app.py`: Streamlit dashboard (Part 8). | Opens, reads `results.csv`, shows plots. |
| 7:15–... | Report (Part 9), running in parallel from 5:45 by the report owner. | 5 sections present. |
| last 30m | Fresh clone → `pip install` → `python run_all.py` reproduces. Record 5-min demo, one take. Upload. | **Submitted before deadline.** |

**Hard checkpoint:** by roughly the 1:45 mark you MUST have embeddings that pass the probe AND Naive forgetting badly. If not, that's your problem to fix before adding anything.

---

## PART 8 — THE DASHBOARD (`app.py`)

Reads only `results.csv`. Panels, in order of demo impact:

1. **Live Forgetting Monitor** — as tasks advance, a risk gauge climbs and highlights which classes enter the danger zone; buffer reallocates on screen. *The centrepiece — makes an invisible idea watchable.*
2. **Bidirectional Control** — toggle PREVENT vs. INDUCE. Prevent shows a protected class staying high; Induce shows a target class being cleanly erased while others hold. *This is the panel that sells "we control forgetting both ways."*
3. **Why Was This Class Forgotten?** — click a class → the one-sentence explanation (risk, quota, outcome). *Makes the project explain itself.*
4. **Accuracy-matrix heatmaps** — Naive vs. JANUS, side by side.
5. **Interference heatmap** — which new classes threaten which old ones.

Keep interactivity to selectors and toggles. No per-sample inspector rabbit holes — demo is 10% of the grade.

---

## PART 9 — THE REPORT (4 pages max)

1. **Problem** — catastrophic forgetting under a fixed replay budget; existing replay is reactive; and the unaddressed flip side — deliberate, targeted forgetting for data deletion.
2. **Method** — the discovery (R1), prevention via interference-aware allocation (R2), and induction via reverse interference (R3). Include the one-block equation.
3. **Results** — main table (5 conditions, accuracy + forgetting, mean ± std); the R1 correlation figure; the R3 before/after unlearning table; the R4 proxy-vs-truth correlation.
4. **Ablation** — the 2×2, one sentence per cell.
5. **Limitations** (honest — vague ones lose marks):
   - Frozen backbone → prototypes don't drift; a fine-tuned system needs recomputation.
   - Cosine-of-means is a crude interference proxy; multi-modal classes get flattened.
   - Unlearning is evaluated by accuracy only, not by stronger membership-inference tests.
   - Single primary dataset (unless ESC-50 block is present).

**Cite:** CIFAR-100, ESC-50 (Piczak), ResNet/ImageNet weights, PyTorch, and 1–2 references on task-similarity/forgetting and on machine unlearning. **State the leakage discipline explicitly** (Gate C).

---

## PART 10 — TEAM SPLIT (nobody on the same file)

- **A (lead):** `embed.py` → `interference.py` → `unlearn.py` (R3). Owns repo + `run_all.py`.
- **B:** `tasks.py`, `continual.py`, `metrics.py`, Naive + Joint, R1 correlation.
- **C:** `buffer.py`, `experiments.py`, `results.csv`, memory sweep + ablation.
- **D:** `app.py` early with FAKE numbers so it's ready when results land → then owns the **report from ~5:45**, plus `explain.py` (R6).

**D starting the report early is the highest-leverage decision on this list.** Reports are what unfinished teams sacrifice, and it carries most of your rigor and presentation evidence.

---

## PART 11 — HOW THIS MAPS TO THE RUBRIC

| Criterion | Weight | Target | Why |
|---|---|---|---|
| Problem quality | 20% | 19 | Bidirectional control + right-to-be-forgotten hook; sharp and forward-looking |
| Beats baseline | 20% | 19 | Any replay crushes naive fine-tuning — near-zero risk |
| Depth of solution | 20% | 19 | Discovery + prevention + induction + gradient-truth validation |
| Experimental rigor | 15% | 14 | Fixed splits, 5 seeds, leakage stated, unlearning proven not claimed |
| Originality | 15% | 14 | Controlling forgetting in both directions is a different category, not a better version |
| Demo / presentation | 10% | 10 | Live monitor + bidirectional toggle + self-explanation |
| **Total** | | **~95 → 9.5/10** | **IF it all runs, reproduces, and every number is consistent** |

**Honest ceiling.** ~9.5 is about the top of what a one-day student hackathon reaches. Going higher needs a genuinely novel algorithm with theory — a semester, not a day. What caps you now is **execution and verification discipline, not ideas.** Six-plus results is a lot to keep consistent, so your risk is no longer "is it impressive" — it's "does every number line up and reproduce by the deadline."

**The one thing that would sink this:** letting "make it more unique" tempt anyone into a trainable backbone, generative replay, or an RL memory controller. Those don't converge in a day and they destabilise every number. The deal is fixed: **ambitious in questions and framing, boring and reliable in machinery.** Hold that line and JANUS is the best-looking project in the room that actually compiles.

---

## PART 12 — FINAL SUBMISSION CHECKLIST

- [ ] Repo runs end-to-end with ONE command on a fresh clone
- [ ] `seed=42` global; results reproduce
- [ ] Test split used exactly once (Gate C stated in report)
- [ ] All replay methods: equal buffer size AND equal replay volume
- [ ] Main table has both mandatory baselines (Naive, Joint)
- [ ] 2×2 ablation present
- [ ] R1 correlation figure present with r value
- [ ] R3 unlearning: before/after table, proof gate passed (or R3 cut cleanly)
- [ ] Every number in every plot traces to `results.csv` (Gate A)
- [ ] Every kept result defensible in one sentence (Gate B)
- [ ] Report ≤ 4 pages, all 5 sections, citations, leakage statement
- [ ] Dashboard opens and reads `results.csv`
- [ ] Recorded demo ≤ 5 min, one take
- [ ] Everything uploaded before the deadline

---

**The whole strategy in one line:** don't be 5% better than five identical projects — be a different *kind* of project (control forgetting both ways), prove it airtight so no number can be punctured, and make it explain itself. That's how you win a relative grade.
