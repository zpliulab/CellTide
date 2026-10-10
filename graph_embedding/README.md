# Portable graph embedding training

## Program

All data preparation, model definitions, training, checkpoint loading, and output are contained in `train_scgpt_multitask_recon.py`. No local split modules are required.

## Run a single-stage example

Required packages: `torch`, `torch-geometric`, `numpy`, `pandas`, `scipy`, `anndata`. The example adapter also requires a compatible `scgpt` installation and pretrained checkpoint directory containing `vocab.json`, `args.json`, and `best_model.pt`.

```bash
python graph_embedding/train_scgpt_multitask_recon.py --adata data/stage.h5ad --network data/network.csv --model-dir checkpoints/scGPT_human --counts-layer counts --weight-col CondPCC --weight-transform absolute --stage stage_4 --output results/stage_4 --device cpu
```

Choose `--weight-transform absolute` only if using correlation magnitude for topology is appropriate for your analysis. It discards activation/repression signs. The default rejects negative weights instead of modifying them silently. Omit `--weight-col` for unit weights.

Alternatively, omit `--counts-layer` if the AnnData already contains a compatible `X_binned` layer. Use `--binned-layer` for a different layer name. Bins must be integer IDs in `[0, n_bins)`, with cells in rows and genes in columns. No automatic transpose or fallback quantile histogram is applied.

`--counts-layer X`, `raw`, or a named layer explicitly selects the count source. `raw` does not inherently mean raw counts; the adapter checks numerical count properties, and the user must confirm provenance. Scaled negative expression is rejected, not clipped to zero. `--gene-column _index` explicitly restores identifiers when required; duplicated identifiers must be resolved before training.

The checkpoint adapter requires strict architecture compatibility and uses `weights_only=True` loading. Architecture mismatches are errors rather than silent partial initialization. Its scGPT constructor follows the supplied script; alternate checkpoint architectures may require an adapter change. The CLI has not been tested with a real scGPT checkpoint in this environment.

## Preserved scientific implementation

The original script calls `model.encoder(gene_ids)`. This refactor preserves that token-embedding lookup. It does **not** perform an expression-conditioned Transformer forward pass. Stage-specific expression enters through the gene histogram reconstruction objective. Fine-tuning affects the selected token encoder, not the unused Transformer backbone.

The original two-layer weighted GCN, projection, sigmoid gate, KL expression reconstruction, BCE edge reconstruction, Adam optimizer, StepLR schedule and gradient clipping are retained. Dimensions, epochs, device, paths, stage labels and data keys are configurable. The Python training API exposes loss weights and optimizer settings through `TrainConfig`.

PyG edges use source-to-target message passing. GCN adds its own self-loops; explicit reconstruction self-loops and duplicate edges are rejected. Nonnegative graph weights are required for this normalization.

The edge decoder remains symmetric inner product. It cannot distinguish `TF → target` from its reverse. Reverse positive edges are therefore excluded from negative sampling to avoid contradictory supervision. This changes the original sampling protocol and can change results. A directed decoder would be a separate model change.

## Outputs and checkpoint selection

Each run writes `embeddings.csv`, `alpha.csv`, `training_history.csv`, `run_config.json`, `checkpoint.pt`, and `stage_embedding.npz`. The checkpoint stores all participating trainable modules, gene ordering, model parameters, optimizer/scheduler states, and training metadata.

Best-state selection uses post-update training loss evaluated against fixed negatives. This avoids comparing checkpoints using newly sampled negatives each epoch, and fixes the original mismatch between pre-update selection loss and post-update exported embeddings. It is **not held-out validation** and does not establish generalization. The checkpoint is sufficient to restore model parameters and the exported embeddings; exact interrupted-training replay also requires RNG state and a resume workflow, which are not implemented here.

## Connecting to transition analysis

A single-stage NPZ contains `(G, D)` embeddings. Do not simply concatenate independently trained stage embeddings and treat them as aligned coordinates. Separate training can rotate or otherwise change coordinates even with identical gene ordering and initialization.

For a trajectory, use the same gene identifiers and ordering across all stages, and establish coordinate comparability through a shared model/training protocol or a justified alignment procedure. Common initialization or seed alone is insufficient. Only then assemble `(T, G, D)` and pass it to `detect_transitions`. Joint multistage training and cross-stage alignment are not implemented by this single-stage refactor.

## Verification

Synthetic checks cover sparse histogram extraction, negative-weight handling, CPU dual-task training, frozen token embeddings, and checkpoint restoration. Real h5ad/network/scGPT checkpoint inputs are still required for end-to-end validation and comparison with the original manuscript results.
