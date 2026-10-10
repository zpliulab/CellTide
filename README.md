# CellTide

**CellTide identifies critical transitions and regulatory programs in dynamic single-cell transcriptomes**

CellTide (Cell Trajectory in dynamic evolution) is a computational framework for identifying candidate critical transitions and associated regulatory programs from time-resolved or biologically ordered single-cell transcriptomic data. It integrates gene expression semantics with stage-specific gene regulatory topology to characterize changes in cell states and prioritize candidate regulators.

This directory contains manuscript materials, supplementary analysis scripts, regulatory interaction assessments, and figure outputs. This README is based on the local manuscript and available scripts. The directory currently serves as a manuscript analysis workspace and does not provide a unified entry point for CellTide model training and inference.

## Method overview

1. **Preprocessing and network inference.** Perform quality control, normalization, and Metacell aggregation of scRNA-seq data. Infer stage-specific gene regulatory networks using PC-CMI, with candidate interactions constrained by RegNetwork priors.
2. **Joint representation learning.** Encode gene expression semantics using pretrained scGPT and regulatory topology using a two-layer graph convolutional network (GCN). Fuse these representations through gated attention and optimize gene embeddings using expression distribution reconstruction and network edge reconstruction objectives.
3. **Critical transition detection.** Combine system velocity between adjacent stages with network entropy to calculate transition scores and identify candidate critical states or transition intervals.
4. **Regulatory program prioritization.** Apply transcription factor (TF)-driven virtual perturbations and assess the alignment between network propagation directions and observed embedding displacements to rank candidate TFs and responsive target genes.
5. **Structural and functional interpretation.** Examine candidate regulatory subnetworks, functional enrichment, and potential landscape visualizations to interpret state remodeling.

## Applications in the manuscript

| Application | Analysis objective | Key window described in the manuscript |
| --- | --- | --- |
| Esophageal squamous cell carcinoma (ESCC) progression | Identify precancerous transitions and associated regulatory programs | The window associated with high-grade intraepithelial neoplasia (HGIN) |
| Immune remodeling in mouse glioblastoma (GBM) | Characterize the transition from monocytes toward an immunosuppressive macrophage phenotype | 36–48 h after tumor implantation |
| Cell differentiation in human pancreatic slices (HPSs) | Identify transitional cells and regulatory programs along the differentiation trajectory | A critical transitional state during ductal cell differentiation |

The directory also includes comparison outputs under the analysis labels GAC, DIFF, DIFF2, and MOUSE. Refer to the corresponding configuration files and manuscript methods for group definitions, stage ordering, and analysis parameters.

Virtual perturbations generate candidate regulatory hypotheses and do not establish experimentally validated causality. Potential landscapes are computational representations and should not be interpreted as directly measured physical energy.



## Running supplementary analyses

The examples below apply to analysis scripts available in this directory. They do not constitute the complete CellTide model training workflow. Prepare the required input data and replace example paths with your local paths before running the commands.

### Python dependencies

The tipping point comparison and interval plotting scripts use the following packages:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn anndata
```

A version-pinned environment file is not provided in this directory. This dependency list is derived from script imports; other scripts may require additional packages. Some scripts contain local absolute paths, so check their input and output locations before execution.

### Tipping point index comparison

The default input filename pattern is `seurat_{stage}.h5ad`. Stage names must match the `--stages` argument. The script reads `AnnData.raw` by default; use `--layer` to select another expression layer.

```bash
python compare_tipping_points_h5ad.py --h5dir "path/to/h5ad" --stages "N,LGIN,HGIN,StageI,StageII,StageIII,StageIV" --outdir "tipping_point_results" --bootstraps 200 --n-cells 21 --n-genes 100 --seed 20260728
```

The script compares DNB-like indices, shrinkage DNB, mean absolute correlation, covariance-spectrum statistics, correlation entropy, and forward transition distance using balanced sampling across stages and bootstrap resampling. Report the formula-based DNB-like implementations separately from official BioTIP outputs.

### Official BioTIP scores

An R environment with the `BioTIP` package is required. The input CSV must contain genes in rows and cells or samples in columns, with column names formatted as `stage__sample_identifier`.

```bash
Rscript run_official_biotip.R "expression.csv" "official_biotip_scores.csv" "N,LGIN,HGIN,StageI,StageII,StageIII,StageIV"
```

### Interval-aligned visualization

The output directory must already contain `stage_score_summary.csv`, `bootstrap_scores.csv`, and `official_biotip_scores.csv`. To include the CellTide curve, supply a CSV with `interval` and `score` columns through `--celltide-csv`.

```bash
python plot_transition_aligned_tipping.py --outdir "transition_aligned_results/tipping_ESCC" --stages "N,LGIN,HGIN,StageI,StageII,StageIII,StageIV" --celltide-csv "transition_aligned_results/tipping_ESCC/_celltide_rerender.csv"
```

For state-based methods, the destination-state score is assigned to the interval entering that state: `S_transition(i → i+1) = S_state(i+1)`. The first state is excluded from this mapping because it has no preceding observed interval. Native forward transition distance retains its interval definition.

## Output files

Dataset directories under `transition_aligned_results/` contain:

- `run_config.json`: original analysis parameters and stage ordering.
- `data_audit.csv`: input data audit records.
- `bootstrap_scores.csv`, `stage_score_summary.csv`: bootstrap scores and stage-level summaries.
- `official_biotip_scores.csv`: official BioTIP results.
- `predicted_tipping_points.csv`: predicted tipping points.
- `transition_alignment_metadata.json`: rules for mapping state scores to intervals.
- `transition_aligned_scores.csv`: interval-aligned scores.
- `transition_aligned_final.svg / .pdf / .tiff`: vector and high-resolution comparative figures.

When interpreting results, check the input genes, cell counts, stage order, random seed, and alignment rules together. Distinguish critical states from transition intervals when describing candidate windows, and compare peaks using consistent definitions.

## Data sources and citation

The manuscript's Data Availability section lists the following GEO datasets: ESCC (GSE160269), mouse GBM (GSE232040), HPSs (GSE223713), GAC (GSE234129), and ITP (GSE196676). These accession numbers are transcribed from the local manuscript and should be checked against the original studies before downloading data or publishing this documentation.

When using these datasets, pretrained models, or third-party methods, cite their original publications alongside CellTide. The CellTide manuscript title is:

> CellTide identifies critical transitions and regulatory programs in dynamic single-cell transcriptomes

The available materials do not include final journal publication details or a DOI. Add the complete bibliographic reference when it becomes available.
