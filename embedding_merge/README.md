# Embedding assembly: pinjie.py

`pinjie.py` is deliberately a zero-byte placeholder, as requested. It currently performs no computation.

## Planned input

An explicitly ordered collection of stage-specific `embeddings.csv` files exported by `train_scgpt_multitask_recon.py`. Each CSV has gene identifiers in its first column and embedding dimensions in the remaining columns. The stage order must be supplied independently and must not be inferred from alphabetical filenames.

## Planned output

The semantic output is the merged stage embedding used in the original example. For the current reusable `transition.py` API, the canonical representation is an NPZ containing:

- `embeddings`: numeric array of shape `(T, G, D)`.
- `genes`: unique string identifiers in the exact order of the gene axis.
- `stage_names`: ordered unique string labels matching the stage axis.
- Optional `timepoints`: strictly increasing numerical sampling times.

A combined CSV can also be exported for compatibility with the original example, but its column ordering must be explicit. For a gene-row CSV with stage-major dimension columns, conversion is `values.reshape(G, T, D).transpose(1, 0, 2)`, not direct `reshape(T, G, D)`.

## Required implementation checks

Match common genes by identifier, reorder each stage identically, verify embedding dimensions, reject duplicate identifiers/nonfinite values, and record discarded genes and stage ordering. Cross-stage coordinate comparability must be established before calculating displacement. This placeholder does not implement alignment, and stacking alone cannot establish it.
