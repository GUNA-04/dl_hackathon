# JANUS — Bidirectional Control of Catastrophic Forgetting

## Claim

JANUS predicts which stored classes are vulnerable to an incoming task using cosine similarity between frozen class prototypes. It uses that signal both to allocate a fixed replay budget and to audit targeted deletion requests.

## Experimental discipline

All main comparisons use identical task splits, buffer capacities, replay volume, and seeds. The dashboard reads `results/results.csv`; figures and tables must be regenerated from that file. Current continual metrics are validation metrics. A final test result must be run once and labelled separately.

## Results

Report the mean and standard deviation across seeds 42–46 from `results/results.csv`. Do not claim a JANUS improvement unless its measured confidence interval supports it.

## Targeted unlearning proof gate

Keep the unlearning claim only when the requested class reaches near-chance accuracy and retained classes lose no more than two points. Otherwise report it as partial unlearning or omit it.

## Limitations

Frozen prototypes do not model representation drift; cosine-of-means is a coarse interference proxy; accuracy-only deletion checks are not certified unlearning; and the current benchmark needs a separate held-out test evaluation.
