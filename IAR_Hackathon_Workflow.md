# IAR — Complete Hackathon Workflow

**Track:** 5 — Continual Learning
**Domain:** Continual visual recognition (class-incremental learning)
**Title:** Interference-Aware Replay: Predicting Task Interference to Allocate a Fixed Replay Budget
**Short name:** IAR
**Dataset:** CIFAR-100, class-incremental split — 20 tasks × 5 classes, seed 42

---

## 0. The idea in one paragraph (for your teacher)

When a neural network learns a sequence of tasks, it overwrites what it learned earlier — catastrophic forgetting. The standard fix is experience replay: keep a small memory of past examples and revisit them while learning new tasks. But under a tight memory budget, existing methods allocate that memory *reactively* — they wait until forgetting has been measured, then respond. We argue forgetting is *predictable in advance*, because new classes mainly damage old classes that are representationally similar to them. Before training on an incoming task, we compute a prototype embedding for every class, measure the similarity between incoming and previously-learned classes, and produce a per-class interference risk score. We distribute the fixed replay buffer in proportion to that predicted risk — protecting the classes most at threat before the damage occurs — and select samples within each class quota by herding. We evaluate on a class-incremental split of CIFAR-100 against naive sequential fine-tuning and joint training (the mandatory baselines), plus random and herding-based replay, reporting final average accuracy and average forgetting under matched memory budgets. Our central evidence is the correlation between predicted interference and observed forgetting, which tests the hypothesis directly and independently of whether our allocation strategy wins.

**What is actually new:** not the observation that similar tasks interfere (that is known) — the contribution is turning it into a concrete, cheap allocation rule for a fixed replay budget, and showing the prediction holds empirically.

---

## 1. The three rules that protect your marks

Break any of these and you lose marks you cannot get back. Print them.

1. **The test split is touched exactly ONCE, at the very end.** Never train on it, never tune on it, never compute interference or forgetting statistics on it. Carve a validation slice out of the training data for anything you need to measure mid-run. (Protects 15% rigor.)
2. **Every replay method replays the SAME number of samples per step.** Same buffer size AND same replay volume. Otherwise you are comparing how much you replay, not how well you choose — and your whole result is meaningless.
3. **seed = 42 everywhere, set once, globally.** torch, numpy, random, and `cudnn.deterministic = True`. This is mandatory and it makes your run reproducible. Report seed-42 as the headline number; report mean ± std over 5 seeds as supplementary.

---

## 2. Repo structure

Build the files in this order. Each does one job.

```
iar-hackathon/
├── run_all.py            # ONE command runs everything end-to-end
├── requirements.txt
├── README.md             # how to run, in 3 lines
├── seed.py               # sets seed=42 for torch/numpy/random
├── data.py               # loads CIFAR-100, returns (images, labels)
├── embed.py              # frozen ResNet-18 -> embeddings.npy (run ONCE)
├── tasks.py              # splits 100 classes into 20 tasks of 5
├── continual.py          # training loop + the accuracy matrix R[i][j]
├── metrics.py            # final avg accuracy + avg forgetting from R
├── buffer.py             # the memory: random / herding / IAR selection
├── iar.py                # THE CONTRIBUTION: prototypes, risk, allocation
├── experiments.py        # runs all conditions -> results.csv
├── figures.py            # makes the 4 plots from results.csv
├── app.py                # Streamlit dashboard
└── results/
    ├── embeddings.npy
    ├── labels.npy
    └── results.csv
```

### What each file must do

- **seed.py** — one function `set_seed(42)`. Called at the top of everything.
- **data.py** — `torchvision.datasets.CIFAR100(download=True)`. Return raw images + labels. Nothing clever.
- **embed.py** — frozen ImageNet ResNet-18 (drop the final layer). Push every image through once. Save `embeddings.npy` (N×512) and `labels.npy` (N). **This is the only slow step. Run it once and never again.**
- **tasks.py** — assign the 100 classes to 20 ordered tasks of 5 classes each, deterministically under seed 42. Return, per task: which classes, and train/val/test index lists.
- **continual.py** — trains a small classifier head task by task. After finishing each task, evaluate on every task seen so far and fill in the **accuracy matrix** `R[i][j]` = accuracy on task *j* after training through task *i*. Everything downstream comes from this one matrix.
- **metrics.py** —
  - Final average accuracy: `A = mean over j of R[last][j]`
  - Average forgetting: for each earlier task *j*, `max over i of R[i][j]` minus `R[last][j]`; average those.
- **buffer.py** — a memory of fixed capacity N. Three selection modes:
  - `random` — pick samples at random
  - `herding` — pick samples closest to their class mean
  - `iar` — use the allocation from `iar.py`, then herding within each class quota
- **iar.py** — the contribution (see Section 3).
- **experiments.py** — runs every condition, every memory size, every seed, every ablation cell. Writes one tidy `results.csv`.
- **figures.py** — reads `results.csv`, writes the 4 figures (Section 6).
- **app.py** — Streamlit, shows the figures. Basic.
- **run_all.py** — calls embed → experiments → figures in order, so a fresh clone reproduces everything with `python run_all.py`.

---

## 3. The contribution — `iar.py` (the whole novelty)

Before training on task *t*:

1. **Prototypes.** For every class seen so far AND every incoming class, average its embeddings → one 512-d vector per class. L2-normalise.
2. **Interference matrix.** For each old class, its risk = the *maximum* cosine similarity to any incoming class. High similarity = about to be damaged.
3. **Allocation.** Give every old class a small floor (so none starves), then distribute the remaining budget in proportion to risk (a softmax over risk scores with a temperature τ).
4. **Selection.** Within each class's quota, keep the samples closest to that class's prototype (herding).

```python
# the entire novel mechanism — everything else is standard CL code
P_old = prototypes[old_classes]            # (C_old, 512), L2-normalised
P_new = prototypes[new_classes]            # (C_new, 512), L2-normalised
risk  = (P_old @ P_new.T).max(axis=1)      # (C_old,) predicted interference
quota = floor + (N - floor*C_old) * softmax(risk / tau)   # per-class budget
# then: keep the top `quota[c]` samples of class c nearest its prototype
```

**Do NOT spend more than 20 minutes tuning τ.** Try 3 values on the validation split, pick the best, move on.

---

## 4. Experimental design

### Main comparison (5 conditions)

| Condition | Buffer | Role |
|---|---|---|
| Naive sequential fine-tuning | 0 | **Mandatory lower bound** |
| Joint training | ∞ | **Mandatory upper bound** |
| Random ER | N | Standard comparison |
| Herding ER (uniform quota) | N | Strong comparison |
| **IAR** (risk quota + herding) | N | **Ours** |

### Metrics
- Final average accuracy `A_T` (higher better)
- Average forgetting `F_T` (lower better)
- **The correlation** between predicted risk and measured per-class forgetting — this is your headline

### Ablation — 2×2 (isolate allocation from selection)

| Allocation | Selection | Tests |
|---|---|---|
| Uniform | Random | Plain ER |
| Uniform | Herding | Does better selection help? |
| Risk-based | Random | Does better allocation help? |
| Risk-based | Herding | Full IAR |

This is stronger than a cumulative ablation because every cell answers a distinct question and none are redundant.

### Memory-budget sweep
N = 20, 50, 100. The small-N regime is where allocation matters most — that is your narrative ("with a tiny buffer, ours remembers more").

### Seeds
Seed 42 = headline. Also run seeds 42, 43, 44, 45, 46 and report mean ± std. Runs are seconds each, so this is nearly free and the brief explicitly warns against lucky seeds.

---

## 5. Execution timeline (8 hours, offsets from start)

Each block ends in a **GATE** — a specific thing that must be true. If a gate fails, take the fallback **immediately**, no debate.

| Time | Do | GATE / fallback |
|---|---|---|
| 0:00–0:30 | Start CIFAR-100 download in background. Build repo skeleton, `seed.py`, empty `run_all.py`. Declare track + dataset to faculty. | Dataset downloading. |
| 0:30–1:15 | `embed.py`: frozen ResNet-18 → `embeddings.npy`. | **Linear probe on all 100 classes ≥ 50%.** If near chance, embeddings are broken — fix now, nothing works downstream. |
| 1:15–2:00 | `tasks.py`, `continual.py` (the accuracy matrix), then Naive + Joint. | **Naive forgets badly; Joint far above it. 20% of marks banked. Save the matrices.** If forgetting is weak, make tasks harder. |
| 2:00–2:45 | `buffer.py`: Random ER + Herding ER. | Both beat Naive. Save numbers as you go — never re-run to recover a result. Enforce **equal replay volume**. |
| 2:45–4:00 | `iar.py`: prototypes, risk, allocation, herding. | IAR runs end-to-end and produces a matrix. Cap τ tuning at 20 min. |
| 4:00–4:45 | `experiments.py`: main table × 5 seeds, memory sweep (3), 2×2 ablation → `results.csv`. | One `results.csv` with everything. |
| 4:45–5:30 | `figures.py`: the 4 plots. | Headline scatter exists and shows a trend. |
| 5:30–6:30 | `app.py`: Streamlit, 4 panels, method selector only. **No per-sample inspector.** | Dashboard opens and shows plots. |
| 6:30–7:30 | 4-page report. | All 5 sections present (below). |
| 7:30–8:00 | Fresh clone → `pip install` → `python run_all.py` → reproduces. Record 5-min demo, one take. Upload. | **Submitted before 5pm.** |

### The one hard checkpoint
**By 11am (≈ T+2:00) you MUST have:** embeddings that pass the probe check, and Naive forgetting badly. If not, and you had chosen audio, you'd switch to CIFAR-100 here — but you are already on CIFAR-100, so this is your safety margin, not a switch point.

---

## 6. Figures (priority order — make #1 first)

1. **Predicted interference (x) vs. actual forgetting (y), scatter + correlation r.** Your headline. Proves the hypothesis even if the method doesn't win.
2. **Average forgetting per method** (bar chart). Naive tallest, IAR shortest.
3. **Final accuracy vs. memory budget** (line, N = 20/50/100). IAR above ER, gap widest at small N.
4. **Accuracy-matrix heatmaps, Naive vs. IAR side by side.** Visually shows retained knowledge.

---

## 7. Dashboard (`app.py`) — keep it basic

Four panels, one method-selector dropdown, nothing else:
1. Accuracy-matrix heatmap
2. Interference heatmap (which new classes threaten which old ones)
3. Buffer composition by class (how IAR spent the memory)
4. Forgetting curves per method

Demo is 10% of the grade. Do not let it eat report time. If you run short, screenshots of the 4 figures in a plain Streamlit page is enough.

---

## 8. Report (4 pages max — the structure judges expect)

1. **Problem** — catastrophic forgetting under a fixed replay budget; existing replay is reactive.
2. **Method** — proactive interference prediction from prototype similarity → risk-proportional allocation → herding selection. Include the one-block equation.
3. **Results** — the main table (5 conditions, accuracy + forgetting, mean ± std), and the headline correlation figure.
4. **Ablation** — the 2×2 table, one or two sentences on what each cell tells you.
5. **Limitations** (write these honestly — vague ones lose marks):
   - Single dataset (unless you added ESC-50 as a stretch goal).
   - Frozen backbone → prototypes never drift; a fine-tuned system would need recomputation.
   - Cosine similarity between class means is a crude interference proxy (multi-modal classes get flattened).
   - Prediction made once per task; interference may shift during training.

**Cite:** CIFAR-100, ResNet / ImageNet pretrained weights, PyTorch, and 1–2 continual-learning references on task similarity and forgetting. **State explicitly that risk/forgetting stats are computed on validation, never test.** Judges reward teams that flag their own leakage risk.

---

## 9. Team split (nobody on the same file)

- **A (lead):** `embed.py` → `iar.py`. Owns the repo and `run_all.py`.
- **B:** `tasks.py`, `continual.py`, `metrics.py`, Naive + Joint.
- **C:** `buffer.py` (random + herding), then `experiments.py` + `results.csv`.
- **D:** starts `app.py` early with FAKE numbers so it's ready when results land → then writes the report from T+5:30.

**D starting the report early is the highest-leverage decision here.** Reports are what unfinished teams sacrifice, and it carries most of your rigor and presentation evidence.

---

## 10. Stretch goal (only if main table + report done by T+5:30)

Add **ESC-50** (or any second dataset) as a second results block. The method code does not change — only the loader. This upgrades your claim from "worked on our dataset" to "holds across two modalities," which is a real originality win. **Drop it without hesitation if you're behind.** One complete dataset beats two half-finished ones.

---

## 11. How this maps to the rubric

| Criterion | Weight | Target | How you earn it |
|---|---|---|---|
| Problem quality | 20% | 17 | Sharp, motivated, scoped to the time |
| Beats baseline | 20% | 19 | Any replay crushes naive fine-tuning — near-zero risk |
| Depth of solution | 20% | 16 | Real mechanism + 2×2 ablation + memory sweep |
| Experimental rigor | 15% | 13 | Fixed splits, 5 seeds, leakage handled and stated |
| Originality | 15% | 11 | Proactive vs. reactive; honest that similarity effects are known and your part is the allocation rule |
| Demo / presentation | 10% | 8 | Clean, finished, not over-built |
| **Total** | | **~84 = 8.4/10** | Finishing completely beats a bigger idea half-done |

---

## 12. Final submission checklist (do not skip)

- [ ] Repo runs end-to-end with **one command** on a fresh clone
- [ ] `seed = 42` set globally; results reproduce
- [ ] Test split used exactly once, for final numbers only
- [ ] All replay methods use equal buffer size AND equal replay volume
- [ ] Main results table includes both mandatory baselines (Naive, Joint)
- [ ] Ablation present (2×2)
- [ ] Report ≤ 4 pages, all 5 sections, citations included, leakage statement present
- [ ] 4 figures generated from `results.csv`
- [ ] Recorded demo (≤ 5 min)
- [ ] Everything uploaded **before 5pm**

---

**The whole logic of this plan in one line:** a modest idea executed completely and honestly, finished with time to spare, beats a brilliant idea that's still compiling at 4:45pm.
