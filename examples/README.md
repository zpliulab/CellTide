# Combined transition and perturbation example

`run_analysis.py` loads an assembled trajectory, runs transition detection and regulator prioritization, and exports results. It does not implement the empty `pinjie.py` step.

From the repository root:

```bash
python -m examples.run_analysis --embeddings data/trajectory_embeddings.npz --network results/networks/network_union_all_stages.csv --weight-col CMI_1 --top-k 10 --output results/analysis
```

The NPZ must contain `embeddings` `(T,G,D)`, `genes`, and `stage_names`; optional `timepoints` are supported. String arrays should use string dtype, not object dtype. The code uses `allow_pickle=False`.

`--weight-col` explicitly chooses the network stage. `--interval` optionally specifies a zero-based interval; otherwise a unique detected peak is used. If no unique peak exists, the script raises an error and requires inspection before manually selecting an interval. `--propagation legacy` selects the original reverse-propagation mode.

Outputs: `transition_scores.csv`, `bootstrap_peaks.csv`, `tf_scores.csv`, `target_scores.csv`, `selected_edges.csv`, and `run_config.json`. The saved configuration records input paths, stage names, selected interval and TFs, propagation mode, and entropy definition.
