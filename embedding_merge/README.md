# Embedding assembly: pinjie.py

This standalone script generalizes the supplied merging program. It retains explicit stage ordering, sorted common genes, and horizontal CSV concatenation. It also exports the trajectory NPZ accepted by the transition/perturbation example.

## Run

From the repository root:

```bash
python embedding_merge/pinjie.py --files output_multitask/emb_fused_D12.csv output_multitask/emb_fused_D6.csv output_multitask/emb_fused_A1.csv output_multitask/emb_fused_I.csv --stages D12 D6 A1 I --output results/merged
```

Each input CSV has gene identifiers in its first column and embedding dimensions in the remaining columns. Both the original `emb_fused_*.csv` files and the current trainer's `embeddings.csv` files are supported, provided dimension names and order match across the selected files. Stage order follows `--files` and `--stages`, not filename sorting. Dependencies: numpy, pandas.

## Outputs

- `merged_embedding_common_genes.csv`: genes in rows, stage-major dimension columns prefixed `emb1_`, `emb2_`, etc., matching the original layout.
- `trajectory_embeddings.npz`: numeric `embeddings` array `(T,G,D)`, string `genes` and `stage_names`, and optional numeric `timepoints`.
- `merge_metadata.json`: stage/file mapping, dimension names, shape, and discarded gene identifiers.

Optional `--timepoints` requires strictly increasing numeric values matching the stages; do not assign fictitious sampling times to pathological categories.

## Downstream use

```bash
python -m examples.run_analysis --embeddings results/merged/trajectory_embeddings.npz --network results/networks/network_union_all_stages.csv --weight-col CMI_1 --output results/analysis
```

Choose the weight column using the intended network stage and `stage_mapping.csv`; `CMI_1` is only an example.

The script rejects empty tables, duplicate/missing gene identifiers, nonfinite values, inconsistent dimension columns, and an empty gene intersection. It does not perform coordinate alignment. Independent stage embeddings must have justified coordinate comparability before displacement analysis.

For the wide CSV, correct conversion is `values.reshape(G,T,D).transpose(1,0,2)`, not direct `reshape(T,G,D)`. The NPZ is assembled directly with `np.stack`, avoiding this reshape ambiguity.
