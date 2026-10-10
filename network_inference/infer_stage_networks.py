"""Standalone PC-CMI stage-network inference and ordered network union.

Input: expression CSV exported from R, and a TF/Gene prior edge CSV.
Dependencies: numpy, pandas, scipy, scikit-learn, networkx.
Use --help for the command-line interface.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import mutual_info_score

MI_BINS = 10

def mutual_info(X, Y, bins=MI_BINS):
    X = X.flatten()
    Y = Y.flatten()
    try:
        x_disc = pd.cut(X, bins=bins, labels=False, duplicates='drop')
        y_disc = pd.cut(Y, bins=bins, labels=False, duplicates='drop')
    except ValueError:
        try:
            x_disc = pd.cut(X, bins=min(len(np.unique(X)), 2), labels=False, duplicates='drop')
            y_disc = pd.cut(Y, bins=min(len(np.unique(Y)), 2), labels=False, duplicates='drop')
        except ValueError:
            return 0.0
    c_xy = np.histogram2d(x_disc, y_disc, bins=bins)[0]
    mi_val = mutual_info_score(None, None, contingency=c_xy)
    return max(mi_val, 0.0)


def conditional_statistics(x, y, z=None, epsilon=1e-6):
    """Regularized Gaussian CMI and absolute residual Pearson correlation."""
    x, y = np.asarray(x).reshape(-1, 1), np.asarray(y).reshape(-1, 1)
    z = np.empty((len(x), 0)) if z is None else np.asarray(z).reshape(len(x), -1)
    if len(x) <= z.shape[1] + 2:
        raise ValueError("Too few samples for the requested conditioning order.")
    def logdet(values):
        if values.shape[1] == 0:
            return 0.
        cov = np.atleast_2d(np.cov(values, rowvar=False))
        sign, value = np.linalg.slogdet(cov + epsilon * np.eye(cov.shape[0]))
        if sign <= 0 or not np.isfinite(value):
            raise ValueError("Invalid regularized covariance matrix.")
        return value
    cmi = max(0., .5 * (logdet(np.hstack([x, z])) + logdet(np.hstack([y, z]))
                         - logdet(z) - logdet(np.hstack([x, y, z]))))
    design = np.column_stack([np.ones(len(x)), z])
    residual_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    residual_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    denom = np.linalg.norm(residual_x) * np.linalg.norm(residual_y)
    correlation = 0. if denom <= 1e-12 else abs(float(np.sum(residual_x * residual_y) / denom))
    return cmi, min(1., correlation)


def load_expression(path, orientation="genes-by-samples", min_variance=1e-8):
    frame = pd.read_csv(path, index_col=0)
    if orientation == "genes-by-samples":
        frame = frame.T
    elif orientation != "samples-by-genes":
        raise ValueError("Unknown expression orientation.")
    frame.columns = frame.columns.astype(str).str.strip()
    if not frame.columns.is_unique or (frame.columns.str.len() == 0).any():
        raise ValueError("Gene identifiers must be unique and nonempty.")
    frame = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Expression contains missing or nonfinite values.")
    frame = frame.loc[(frame != 0).any(axis=1)]
    if len(frame) < 3:
        raise ValueError("At least three nonzero samples/Metacells are required.")
    frame = frame.loc[:, frame.var() > min_variance]
    # Distinct genes with identical expression remain distinct identifiers.
    if frame.shape[1] < 2:
        raise ValueError("Fewer than two variable genes remain.")
    return frame


def infer_network(data, prior, *, mi_threshold=.3, pcc_threshold=.3,
                  cmi_threshold=.01, conditional_pcc_threshold=.01,
                  max_order=1, mi_bins=10, epsilon=1e-6,
                  candidate_mode="prior", stop_when_unchanged=False,
                  aggregation="max", max_condition_sets=100000):
    """Infer prior-oriented associations; edge direction comes from the prior.

    Conditioning uses common outgoing TF neighbors, matching the original.
    max aggregation reproduces separate maxima of CMI and conditional PCC;
    all requires both criteria for every tested conditioning set.
    """
    if candidate_mode not in ("prior", "augment") or aggregation not in ("max", "all"):
        raise ValueError("Unknown candidate/aggregation mode.")
    if max_order < 0 or mi_bins < 2 or epsilon <= 0 or max_condition_sets < 1:
        raise ValueError("Invalid order, bin count, regularization or combination limit.")
    thresholds = [mi_threshold, pcc_threshold, cmi_threshold, conditional_pcc_threshold]
    if not all(np.isfinite(v) and v >= 0 for v in thresholds) or max(pcc_threshold, conditional_pcc_threshold) > 1:
        raise ValueError("Thresholds must be finite/nonnegative; correlation thresholds must be <= 1.")
    if len(data) <= max_order + 2:
        raise ValueError("Sample count must exceed max_order + 2.")
    if not {"TF", "Gene"}.issubset(prior.columns) or prior[["TF", "Gene"]].isna().any().any():
        raise ValueError("Prior requires nonmissing TF and Gene columns.")
    prior = prior[["TF", "Gene"]].copy()
    for column in ["TF", "Gene"]:
        prior[column] = prior[column].astype(str).str.strip()
        if prior[column].eq("").any():
            raise ValueError("Empty prior gene identifier.")
    prior = prior.drop_duplicates()
    genes = set(data.columns)
    tfs = set(prior.TF) & genes
    pairs = set((u, v) for u, v in prior.itertuples(index=False, name=None)
                if u in genes and v in genes and u != v)
    graph = nx.DiGraph()
    graph.add_nodes_from(data.columns)

    def screen(u, v):
        x, y = data[u].to_numpy(), data[v].to_numpy()
        r = abs(float(np.corrcoef(x, y)[0, 1]))
        mi = mutual_info(x, y, bins=mi_bins)
        return r, mi, r > pcc_threshold or mi > mi_threshold

    if candidate_mode == "prior":
        for u, v in sorted(pairs):
            r, mi, passed = screen(u, v)
            if passed:
                graph.add_edge(u, v, PCC=r, MI=mi)
    else:
        # Original behavior: retain prior edges with nonzero PCC, and augment
        # with screened TF-to-gene candidates across all observed genes.
        for u, v in sorted(pairs):
            r, mi, _ = screen(u, v)
            if r > 0:
                graph.add_edge(u, v, PCC=r, MI=mi)
        for u, v in itertools.combinations(sorted(genes), 2):
            if u not in tfs and v not in tfs:
                continue
            r, mi, passed = screen(u, v)
            if passed and r > 0:
                if u in tfs:
                    graph.add_edge(u, v, PCC=r, MI=mi)
                if v in tfs:
                    graph.add_edge(v, u, PCC=r, MI=mi)
    initial_count = graph.number_of_edges()
    order_audit = []
    for order in range(max_order + 1):
        remove = []
        evaluated = 0
        for u, v in list(graph.edges()):
            neighbors = sorted((set(graph.successors(u)) & set(graph.successors(v)) & tfs) - {u, v})
            if len(neighbors) < order:
                continue
            groups = itertools.combinations(neighbors, order) if order else [()]
            values = []
            for i, group in enumerate(groups):
                if i >= max_condition_sets:
                    raise ValueError("Conditioning combination limit exceeded; lower order or explicitly raise --max-condition-sets.")
                z = data[list(group)].to_numpy() if group else None
                values.append(conditional_statistics(data[u].to_numpy(), data[v].to_numpy(), z, epsilon))
            metrics = np.array(values)
            cmi, cpcc = metrics.max(axis=0) if aggregation == "max" else metrics.min(axis=0)
            evaluated += 1
            if cmi < cmi_threshold or cpcc < conditional_pcc_threshold:
                remove.append((u, v))
            else:
                graph[u][v].update(CMI=float(cmi), CondPCC=float(cpcc), ConditioningOrder=order)
        graph.remove_edges_from(remove)
        order_audit.append(dict(order=order, evaluated=evaluated, removed=len(remove), retained=graph.number_of_edges()))
        if stop_when_unchanged and not remove:
            break
    result = pd.DataFrame([(u, v, d["CMI"], d["CondPCC"]) for u, v, d in graph.edges(data=True)],
                          columns=["TF", "Gene", "CMI", "CondPCC"])
    audit = dict(samples=len(data), variable_genes=data.shape[1], prior_candidate_edges=len(pairs),
                 initial_edges=initial_count, final_edges=len(result), orders=order_audit)
    return result, graph, audit


def union_networks(networks, stage_names, output_dir):
    """Union observed edges and fill absent stage edges with zero weights."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_edges = pd.concat([n[["TF", "Gene"]] for n in networks], ignore_index=True).drop_duplicates().sort_values(["TF", "Gene"]).reset_index(drop=True)
    union = all_edges.copy()
    mapping = []
    for number, (name, network) in enumerate(zip(stage_names, networks), start=1):
        filled = all_edges.merge(network, on=["TF", "Gene"], how="left", validate="one_to_one")
        filled[["CMI", "CondPCC"]] = filled[["CMI", "CondPCC"]].astype(float).fillna(0.)
        filled.to_csv(output / f"network_stage{number}_filled.csv", index=False)
        union[f"CMI_{number}"] = filled.CMI.to_numpy()
        union[f"CondPCC_{number}"] = filled.CondPCC.to_numpy()
        mapping.append(dict(stage_index=number - 1, stage_number=number, stage_name=name,
                            cmi_column=f"CMI_{number}", pcc_column=f"CondPCC_{number}"))
    union.to_csv(output / "network_union_all_stages.csv", index=False)
    pd.DataFrame(mapping).to_csv(output / "stage_mapping.csv", index=False)
    return union


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expressions", nargs="+", required=True, help="Ordered stage expression CSV files")
    parser.add_argument("--stages", nargs="+", required=True, help="Names in the same order as expression files")
    parser.add_argument("--prior", required=True, help="CSV with TF and Gene columns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--orientation", choices=["genes-by-samples", "samples-by-genes"], default="genes-by-samples")
    parser.add_argument("--mi-threshold", type=float, default=.3)
    parser.add_argument("--pcc-threshold", type=float, default=.3)
    parser.add_argument("--cmi-threshold", type=float, default=.01)
    parser.add_argument("--conditional-pcc-threshold", type=float, default=.01)
    parser.add_argument("--max-order", type=int, default=1)
    parser.add_argument("--mi-bins", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--candidate-mode", choices=["prior", "augment"], default="prior")
    parser.add_argument("--aggregation", choices=["max", "all"], default="max")
    parser.add_argument("--stop-when-unchanged", action="store_true")
    parser.add_argument("--max-condition-sets", type=int, default=100000)
    args = parser.parse_args()
    if len(args.expressions) != len(args.stages) or len(set(args.stages)) != len(args.stages):
        parser.error("Provide one unique stage name for each expression file.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prior = pd.read_csv(args.prior)
    networks, audits = [], []
    parameters = {key: getattr(args, key) for key in ["mi_threshold", "pcc_threshold", "cmi_threshold",
                   "conditional_pcc_threshold", "max_order", "mi_bins", "epsilon", "candidate_mode",
                   "aggregation", "stop_when_unchanged", "max_condition_sets"]}
    for number, (name, path) in enumerate(zip(args.stages, args.expressions), start=1):
        data = load_expression(path, args.orientation)
        network, graph, audit = infer_network(data, prior, **parameters)
        network.to_csv(output / f"network_stage{number}.csv", index=False)
        nx.write_gml(graph, output / f"network_stage{number}.gml")
        networks.append(network)
        audits.append(dict(stage=name, expression_file=path, **audit))
        print(f"{name}: {len(data)} samples, {data.shape[1]} variable genes, {len(network)} edges")
    union = union_networks(networks, args.stages, output)
    (output / "run_config.json").write_text(json.dumps(dict(arguments=vars(args), audit=audits), indent=2), encoding="utf-8")
    print(f"Saved {len(union)} union edges to {output}")


if __name__ == "__main__":
    main()


