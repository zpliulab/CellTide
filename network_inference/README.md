# Stage network inference and union (single file)

`infer_stage_networks.py` combines stage-specific inference and network union. It is independent of ESCC stage names, stage count, and local input paths. R quality control and Metacell aggregation remain upstream steps.

## Input

The upstream R workflow exports one h5ad file and one gene expression CSV for each pathological stage. Use the expression CSV as input to `infer_stage_networks.py`, and the matching h5ad as input to the graph embedding training script. They are paired outputs for the same stage, not two separate stages. Network inference currently reads CSV only; it does not require both files in one invocation.

Keep stage labels and gene identifiers consistent between the CSV and h5ad. Use the R-exported expression representation chosen for network inference; the trainer's counts or binned-expression requirements are separate. The two files need not contain numerically identical transformations, but each transformation and any difference in aggregation level should be documented.

Export one expression CSV per stage from R, with genes in rows and samples or Metacells in columns, including gene identifiers in the first column. For example:

```r
write.csv(as.matrix(metacell_expression), "stage_A_expression.csv")
```

Choose a biologically appropriate expression transformation consistently across stages. This script accepts finite numeric values, including negative scaled values; it does not normalize, aggregate Metacells, or claim that scaled expression is raw counts. For sample-by-gene tables, pass `--orientation samples-by-genes`.

Provide a prior CSV containing named `TF` and `Gene` columns. Duplicate prior edges are collapsed. Self-edges and genes absent from the stage data are excluded. Constant genes and all-zero samples are excluded. Different genes with identical expression profiles are retained rather than silently deleting one identifier.

## Run

Run commands from the repository root.

```bash
python -m pip install numpy pandas scipy scikit-learn networkx
python network_inference/infer_stage_networks.py --expressions data/N.csv data/LGIN.csv data/HGIN.csv --stages N LGIN HGIN --prior data/network.csv --output network_results --max-order 1
```

Any number of ordered stages is supported. Expression file order must match stage label order. Filenames use stage numbers; the label mapping is saved separately.

## Outputs

- `network_stage1.csv`, etc.: inferred networks, with `TF`, `Gene`, `CMI`, `CondPCC` columns.
- `network_stage1.gml`, etc.: graphs with inference statistics.
- `network_stage1_filled.csv`, etc.: common union edge table, with absent stage edges filled with zero.
- `network_union_all_stages.csv`: union table with `CMI_1`, `CondPCC_1`, etc.
- `stage_mapping.csv`: stage names, zero-based embedding indices, one-based network numbers, and weight column names.
- `run_config.json`: parameters and per-stage filtering audit.

Stage-specific nonzero networks can feed the single-file graph embedding trainer. The union table can feed perturbation analysis with an explicitly selected `CMI_i` column. A zero in a filled network means the edge was not retained for that stage; it is not a newly measured conditional statistic. Filled zero-weight edges should be removed before graph training.

## Inference behavior and changes from the original

The initial screen uses absolute PCC OR discretized MI. Defaults follow the supplied running example: PCC/MI thresholds 0.3, conditional CMI/PCC thresholds 0.01, maximum order 1.

Default `--candidate-mode prior` applies this screen only to prior edges. `--candidate-mode augment` restores the original candidate policy: keep nonzero-PCC prior edges and add screened TF-to-gene edges beyond the prior. This can be expensive because it examines gene pairs.

All orders through `--max-order` are executed by default. `--stop-when-unchanged` restores the original early-stop policy. Networks may differ when higher orders are now evaluated.

The original code aggregates CMI and absolute conditional PCC with separate maxima across conditioning sets. Default `--aggregation max` preserves that policy. These maxima can come from different conditioning sets; this is not the same as requiring dependence under every conditioning set. `--aggregation all` uses the minimum of each statistic and requires both thresholds for every tested set. Neither mode should be labeled a standard PC algorithm without matching its precise definition in the manuscript.

Conditioning sets consist of common outgoing neighbors that are TFs, following the original graph convention. Endpoint genes are excluded. Conditional Pearson correlation is computed by least-squares residualization rather than requiring pingouin. Gaussian CMI uses regularized covariance log-determinants rather than flooring determinants; this improves numerical behavior but can change values and threshold decisions. It remains a Gaussian covariance approximation, not a general nonlinear conditional MI estimator.

The script imposes a configurable conditioning-combination limit and fails explicitly if exceeded; it does not silently sample or truncate combinations. Thresholds are effect-size criteria, not statistical significance tests. Edge orientation comes from prior TF labels and does not establish causal direction.

The merged version is not guaranteed to reproduce the old output bit for bit, even with `--candidate-mode augment --stop-when-unchanged`, because numerical handling and identifier handling changed. Compare actual datasets before replacing manuscript results.

Synthetic checks covered conditional statistics, prior restriction, full conditioning-order execution, R CSV orientation, union zero filling and empty networks. Real data validation remains pending.
