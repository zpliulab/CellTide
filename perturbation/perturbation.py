"""One-step TF perturbation and alignment scoring on an explicit network."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse
from transition.transition import row_normalize, validate_embeddings


@dataclass
class PerturbationResult:
    tf_scores: pd.DataFrame
    target_scores: pd.DataFrame
    selected_tfs: list
    selected_edges: pd.DataFrame
    excluded_edge_count: int


def rank_regulators(embeddings, genes, edges, interval_index, *,
                    source_col="TF", target_col="Gene", weight_col="weight",
                    top_fraction=0.05, top_k=None, amplitude=1.,
                    propagation="outgoing", unknown_genes="error"):
    """Rank TF perturbations against raw observed embedding displacement.

    The supplied edge table/weight column explicitly selects the network stage.
    outgoing: A[source, target], P = row_normalize(A).T; response = P @ u.
    legacy: P = row_normalize(A), reproducing the original reverse propagation.
    TF and target alignments have different scopes and are returned separately.
    """
    x = validate_embeddings(embeddings)
    genes = list(genes)
    if len(genes) != x.shape[1] or len(set(genes)) != len(genes):
        raise ValueError("genes must uniquely label the embedding gene axis.")
    if not isinstance(interval_index, (int, np.integer)) or not 0 <= interval_index < len(x) - 1:
        raise ValueError("interval_index must select an observed adjacent-stage interval.")
    if not np.isfinite(amplitude) or amplitude == 0:
        raise ValueError("amplitude must be finite and nonzero.")
    if not np.isfinite(top_fraction) or not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")
    if top_k is not None and (not isinstance(top_k, (int, np.integer)) or top_k < 1):
        raise ValueError("top_k must be a positive integer.")
    if propagation not in ("outgoing", "legacy") or unknown_genes not in ("error", "drop"):
        raise ValueError("Invalid propagation or unknown_genes policy.")
    required = [source_col, target_col, weight_col]
    if not set(required).issubset(edges.columns):
        raise ValueError("Missing required network columns: " + str(required))
    network = edges[required].copy()
    weights = pd.to_numeric(network[weight_col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite and nonnegative.")
    valid = network[source_col].isin(genes) & network[target_col].isin(genes)
    excluded = int((~valid).sum())
    if excluded and unknown_genes == "error":
        raise ValueError("Network contains genes absent from embeddings; align genes or set unknown_genes='drop'.")
    network = network.loc[valid & (weights > 0)].copy()
    if network.duplicated([source_col, target_col]).any():
        raise ValueError("Duplicate regulatory edges: aggregate explicitly before analysis.")
    lookup = {gene: i for i, gene in enumerate(genes)}
    rows = network[source_col].map(lookup).to_numpy(dtype=int)
    cols = network[target_col].map(lookup).to_numpy(dtype=int)
    adjacency = sparse.csr_matrix((network[weight_col].to_numpy(dtype=float), (rows, cols)), shape=(len(genes), len(genes)))
    normalized = row_normalize(adjacency)
    operator = normalized.T.tocsr() if propagation == "outgoing" else normalized
    reference = x[interval_index]
    displacement = x[interval_index + 1] - reference
    displacement_norm = np.linalg.norm(displacement)

    def response(tf):
        idx = lookup[tf]
        norm = np.linalg.norm(reference[idx])
        direction = amplitude * reference[idx] / (norm + 1e-8)
        return operator.getcol(idx).toarray() * direction[None, :]

    records = []
    for tf in sorted(network[source_col].unique()):
        delta = response(tf)
        effect = np.linalg.norm(delta)
        records.append((tf, np.sum(delta * displacement) / (effect * displacement_norm + 1e-8), effect))
    tf_scores = pd.DataFrame(records, columns=["TF", "AlignScore", "EffectNorm"])
    tf_scores = tf_scores.sort_values("AlignScore", ascending=False, kind="stable").reset_index(drop=True)
    count = min(len(tf_scores), top_k if top_k is not None else max(1, int(len(tf_scores) * top_fraction)))
    selected = tf_scores.head(count)["TF"].tolist()
    # Compute each selected TF response once, rather than once for every gene.
    sums = np.zeros(len(genes))
    squares = np.zeros(len(genes))
    support = np.zeros(len(genes), dtype=int)
    real_norm = np.linalg.norm(displacement, axis=1)
    for tf in selected:
        delta = response(tf)
        effect = np.linalg.norm(delta, axis=1)
        mask = (effect >= 1e-8) & (real_norm >= 1e-8)
        align = np.sum(delta * displacement, axis=1) / (effect * real_norm + 1e-8)
        sums[mask] += align[mask]
        squares[mask] += align[mask] ** 2
        support[mask] += 1
    records = []
    for i, gene in enumerate(genes):
        if support[i] and gene not in selected:
            mean = sums[i] / support[i]
            sd = np.sqrt(max(0., squares[i] / support[i] - mean ** 2))
            records.append((gene, mean, sd, support[i]))
    targets = pd.DataFrame(records, columns=["Gene", "ExecScore", "AlignStd", "SupportTFs"])
    targets = targets.sort_values("ExecScore", ascending=False, kind="stable").reset_index(drop=True)
    nodes = set(selected) | set(targets["Gene"])
    selected_edges = network.loc[network[source_col].isin(selected) & network[target_col].isin(nodes)].copy()
    return PerturbationResult(tf_scores, targets, selected, selected_edges, excluded)
