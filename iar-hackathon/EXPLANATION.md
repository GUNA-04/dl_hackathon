# Project Explanation — Study Notes

## What this project is about

This hackathon project is about **continual learning**, a field in machine learning where a model learns a sequence of tasks without forgetting what it already learned.

The main problem is **catastrophic forgetting**: when a neural network learns a new task, it often overwrites what it learned from previous tasks.

This project introduces **IAR — Interference-Aware Replay**, which tries to **predict which old classes will be hurt by the new task before the damage happens**, and then gives those vulnerable classes more memory in the replay buffer.

## Why this project matters

- Traditional methods use a fixed replay buffer and replay random old examples.
- IAR is smarter: it allocates the buffer **based on predicted interference**.
- This is better when memory is limited, because it protects the classes that are most likely to be forgotten.

## How the project works

### 1. Embeddings

We first convert CIFAR-100 images into 512-dimensional vectors using a **frozen ResNet-18** model.

### 2. Task splitting

The 100 classes of CIFAR-100 are split into **20 tasks of 5 classes each**.

### 3. Training

The model trains task by task. After each task, we evaluate how well it remembers all previous tasks.

### 4. IAR algorithm

For each new task, we:

1. Compute a **prototype** (average embedding) for each class.
2. Measure how similar the new classes are to the old classes.
3. Give more buffer space to the old classes that are most similar to the new classes.
4. Keep the best examples of each class (closest to its prototype).

### 5. Evaluation

We compare:

- **Naive fine-tuning** (no replay)
- **Joint training** (all data at once)
- **Random replay**
- **Herding replay**
- **IAR replay**

We measure:

- Final average accuracy
- Average forgetting
- Correlation between predicted interference and actual forgetting

## File-by-file explanation

### `seed.py`

Sets the random seed to **42** for:
- Python `random`
- NumPy
- PyTorch
- CUDA

This makes every run reproducible.

### `data.py`

Loads the **CIFAR-100** dataset from torchvision.

### `embed.py`

Uses a **frozen ResNet-18** to turn images into 512-d embeddings.

### `tasks.py`

Splits the 100 classes into 20 tasks of 5 classes each.

### `continual.py`

Trains the classifier head task by task.

### `metrics.py`

Computes:
- Final average accuracy
- Average forgetting

### `buffer.py`

Implements three types of replay buffers:
- Random
- Herding
- IAR

### `iar.py`

The core contribution:
- Computes prototypes
- Computes interference risk
- Allocates buffer budget
- Selects samples by herding

### `experiments.py`

Runs all experiments and writes `results.csv`.

### `figures.py`

Makes 4 plots:
1. Predicted interference vs actual forgetting
2. Average forgetting per method
3. Final accuracy vs memory budget
4. Accuracy matrix heatmaps

### `app.py`

Streamlit dashboard showing the results.

### `run_all.py`

Runs the whole pipeline:
1. Embeddings
2. Experiments
3. Figures

## Why this project wins

- It predicts forgetting before it happens.
- It allocates memory smarter than random replay.
- It is simple, fast, and easy to explain.
- It includes all required baselines and ablations.

## How to explain this to your teacher

IAR is a smarter replay strategy for continual learning.

Instead of randomly choosing old examples to replay, it first predicts which old classes are most likely to be damaged by the new task. It does this by comparing class prototypes using cosine similarity. Then it gives more buffer space to the most vulnerable classes and selects the best examples from each class.

The result is less forgetting and better memory efficiency under a fixed replay budget.