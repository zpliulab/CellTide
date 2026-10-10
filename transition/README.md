# Transition detection

`transition.py` provides `detect_transitions`; it is a reusable Python module, not a standalone file-loading CLI. Use the combined example in `examples/` for command-line execution.

```python
from transition.transition import detect_transitions

result = detect_transitions(X, stage_names, bootstrap_n=500, random_seed=42)
print(result.scores)
print(result.best_interval)
```

## Input and output

`X` is a finite `(T, G, D)` array with the same gene order and comparable embedding coordinates in every stage. Stage labels must be unique and ordered. Optional `timepoints` divide displacement by elapsed time. Without times, velocity means displacement per ordered interval.

The result includes interval scores, zero-based `best_interval`, smoothed velocities, bootstrap peak indices, and peak frequencies. A tied maximum or a single available interval returns `best_interval=None`; tied bootstrap replicates are marked `-1`. No peak is invented for a static trajectory.

## Calculation

The score is `z(mean gene speed) + z(velocity entropy)`. Entropy uses the distribution of normalized gene velocity magnitudes, matching the supplied implementation's formula. It is not adjacency entropy.

Optional smoothing requires one explicit nonnegative `(G, G)` adjacency matrix per interval and uses `(1-alpha)*v + alpha*row_normalize(A)@v`. Reproducing the original smoothing requires symmetric cosine-weighted matrices constructed from the same stage embeddings and candidate edges. The example CLI uses no smoothing.

Bootstrap resamples genes after smoothing; it does not resample cells or refit the network. Peak frequency indicates stability under that procedure, not biological confidence. Frequencies can sum to less than one when replicates have tied peaks.

Dependencies: `numpy`, `pandas`, `scipy`. Gene assembly is planned in `embedding_merge/pinjie.py` and is not implemented yet.
