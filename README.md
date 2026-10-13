# CellTide

**CellTide identifies critical transitions and regulatory programs in dynamic single-cell transcriptomes**

This repository contains stage-specific network inference, gene graph embedding training, transition detection, and TF-driven virtual perturbation analysis. Quality control and Metacell aggregation are performed upstream in R; those R scripts are not included in this bundle.

## Workflow and programs

| Step | Program | Documentation |
| --- | --- | --- |
| 1. Infer stage networks and their union | `network_inference/infer_stage_networks.py` | [README](network_inference/README.md) |
| 2. Train one stage's gene embeddings | `graph_embedding/train_scgpt_multitask_recon.py` | [README](graph_embedding/README.md) |
| 3. Assemble ordered stage embeddings | `embedding_merge/pinjie.py` | [README](embedding_merge/README.md) |
| 4. Detect critical transition intervals | `transition/transition.py` | [README](transition/README.md) |
| 5. Rank candidate TFs and targets | `perturbation/perturbation.py` | [README](perturbation/README.md) |
| Run steps 4–5 and export results | `examples/run_analysis.py` | [README](examples/README.md) |

`pinjie.py` assembles common-gene stage embeddings into CSV and NPZ outputs. Cross-stage coordinate comparability and real-data end-to-end validation remain necessary.

## Upstream data contract

For each pathological stage, R produces a matching h5ad file and expression CSV. Network inference reads the CSV (genes in rows, samples/Metacells in columns) and a TF/Gene prior. Embedding training reads the h5ad and inferred stage network. Keep gene identifiers and stage labels consistent, and document transformations and aggregation levels. The trainer requires counts or compatible pre-binned data, while network inference can use the appropriate R-exported expression representation.

## Installation and commands

Use a Python environment with the dependencies in `requirements.txt`; versions have not been locked. A compatible scGPT installation and pretrained checkpoint are also needed for the training adapter. Large datasets and weights are not included.

Run commands from this repository's root:

```bash
python network_inference/infer_stage_networks.py --expressions data/N.csv data/LGIN.csv data/HGIN.csv --stages N LGIN HGIN --prior data/network.csv --output results/networks

python graph_embedding/train_scgpt_multitask_recon.py --adata data/N.h5ad --network results/networks/network_stage1.csv --model-dir checkpoints/scGPT_human --counts-layer counts --weight-col CondPCC --stage N --output results/embeddings/N
```

Repeat embedding training for each stage. Assemble the ordered stage CSV files first:

```bash
python embedding_merge/pinjie.py --files results/embeddings/N/embeddings.csv results/embeddings/LGIN/embeddings.csv results/embeddings/HGIN/embeddings.csv --stages N LGIN HGIN --output results/merged
```

After confirming cross-stage coordinate comparability, use:

```bash
python -m examples.run_analysis --embeddings results/merged/trajectory_embeddings.npz --network results/networks/network_union_all_stages.csv --weight-col CMI_1 --output results/analysis
```

`CMI_1` is only an example. Choose the network stage explicitly using `stage_mapping.csv`; the program does not automatically decide whether the source or destination network should be used.

## Implementation scope

- The training script uses scGPT's gene-token encoder, not the full expression-conditioned Transformer forward pass. Expression enters through histogram reconstruction.
- Independently trained stage embeddings are not guaranteed to share coordinates. Gene matching and stacking alone are insufficient for dynamic displacement analysis.
- The transition entropy is Shannon entropy of normalized gene velocity magnitudes, not directly computed adjacency entropy.
- Virtual perturbation rankings are computational hypotheses, not experimental evidence of causality.
- Several corrections differ from the supplied originals; consult each program's README before replacing manuscript outputs.

## Validation and publication

Synthetic tests are provided for transition and perturbation analysis. Synthetic CPU training and checkpoint restoration, as well as two-stage network inference/union, were checked during preparation. Real scGPT checkpoint training and full real-data reproduction remain unverified.

```bash
python -m unittest discover -s tests
```

Before publishing, choose an appropriate license, add verified author/citation details, and provide legitimate data/checkpoint acquisition instructions. A license and publication DOI have not been invented for this bundle.
