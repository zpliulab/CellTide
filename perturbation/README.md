# TF-driven virtual perturbation

`perturbation.py` provides `rank_regulators`. Run it from the repository root as a module/API, or use `examples/run_analysis.py`.

```python
from perturbation.perturbation import rank_regulators

result = rank_regulators(X, genes, edges, interval_index=0,
                         weight_col="CMI_1", top_k=10)
```

## Inputs

- `X`: aligned embeddings `(T, G, D)`.
- `genes`: unique identifiers matching the gene axis.
- `edges`: DataFrame with `TF`, `Gene`, and an explicitly chosen weight column.
- `interval_index`: zero-based adjacent-stage interval.

Network weights must be finite and nonnegative. Missing columns, duplicate edges, or unknown genes are errors by default; `unknown_genes="drop"` explicitly discards unmatched edges and reports their count. Zero-weight edges are ignored. Use `stage_mapping.csv` to select the correct CMI column. Whether to use a source-stage or destination-stage network is an analysis decision.

## Outputs and interpretation

Returns separate TF alignment/effect scores, target execution/support scores, selected TFs, and selected edges. TF scores compare entire response fields; target scores compare individual gene responses, so they are not automatically combined into one calibrated ranking.

For `A[TF, target]`, default propagation is `row_normalize(A).T @ u`, transmitting a TF signal to its targets. `propagation="legacy"` retains the original reverse propagation. Correcting this direction can change rankings and requires real-data review.

The perturbation is a one-step network response to a normalized reference embedding vector. `amplitude` is configurable; a positive perturbation is not automatically a biological knockout. Rankings identify candidate regulatory programs, not experimentally established causal drivers.

Dependencies: `numpy`, `pandas`, `scipy`, plus the sibling transition module for shared validation/normalization.
