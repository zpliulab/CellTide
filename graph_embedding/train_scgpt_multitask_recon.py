"""Standalone CellTide gene graph embedding training.

Example:
    python train_scgpt_multitask_recon.py --adata data/stage.h5ad --network data/network.csv --model-dir save/scGPT_human --counts-layer counts --stage stage4 --output output/stage4

Dependencies: torch, torch-geometric, scgpt, anndata, numpy, pandas, scipy.
Uses scGPT gene-token embeddings, not the full expression-conditioned Transformer.
Independent stage training does not guarantee cross-stage coordinate alignment.
"""
import numpy as np
import pandas as pd
from scipy import sparse


def prepare_graph_data(adata, edges, vocabulary, *, binned_layer="X_binned",
                       source_col="TF", target_col="Gene", weight_col=None,
                       weight_transform="nonnegative", n_bins=51):
    if not adata.var_names.is_unique:
        raise ValueError("Gene identifiers must be unique; aggregate duplicates explicitly.")
    if adata.n_obs == 0 or binned_layer not in adata.layers:
        raise ValueError("A nonempty cells-by-genes binned layer is required.")
    if not {source_col, target_col}.issubset(edges.columns):
        raise ValueError("Missing network source/target columns.")
    if edges[[source_col, target_col]].isna().any().any():
        raise ValueError("Network gene identifiers cannot be missing.")
    candidates = set(edges[source_col]) | set(edges[target_col])
    genes = [g for g in adata.var_names if g in candidates and g in vocabulary]
    if len(genes) < 2:
        raise ValueError("Fewer than two genes shared by expression, vocabulary and network.")
    selected = adata[:, genes]
    binned = selected.layers[binned_layer]
    if sparse.issparse(binned):
        binned = binned.tocsc()
    histogram = np.zeros((len(genes), n_bins))
    for i in range(len(genes)):
        values = binned[:, i].toarray().ravel() if sparse.issparse(binned) else np.asarray(binned[:, i]).ravel()
        if not np.isfinite(values).all() or (values < 0).any() or (values >= n_bins).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Binned values must be integers in [0, n_bins).")
        histogram[i] = np.bincount(values.astype(int), minlength=n_bins) / len(values)
    valid = edges[source_col].isin(genes) & edges[target_col].isin(genes)
    network = edges.loc[valid].copy()
    if network.duplicated([source_col, target_col]).any():
        raise ValueError("Aggregate duplicate edges explicitly.")
    if weight_col is None:
        weights = np.ones(len(network))
    else:
        if weight_col not in network:
            raise ValueError("Missing weight column: " + weight_col)
        weights = pd.to_numeric(network[weight_col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError("Network weights must be finite.")
    if weight_transform == "absolute":
        weights = np.abs(weights)
    elif weight_transform != "nonnegative":
        raise ValueError("Unknown weight transform.")
    if (weights < 0).any():
        raise ValueError("Signed weights are incompatible with this GCN normalization. Choose absolute explicitly or provide nonnegative weights.")
    keep = (weights > 0) & (network[source_col].to_numpy() != network[target_col].to_numpy())
    network = network.loc[keep]
    weights = weights[keep]
    lookup = {g: i for i, g in enumerate(genes)}
    edge_index = np.array([network[source_col].map(lookup), network[target_col].map(lookup)], dtype=np.int64)
    if edge_index.shape[1] == 0:
        raise ValueError("No positive non-self regulatory edges remain.")
    audit = dict(input_gene_count=adata.n_vars, selected_gene_count=len(genes), input_edges=len(edges),
                 unmatched_edges=int((~valid).sum()), retained_edges=len(network))
    return genes, np.array([vocabulary[g] for g in genes]), histogram, edge_index, weights, audit

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphEmbeddingModel(nn.Module):
    """Uses the scGPT token encoder, not contextual Transformer forward.

    edge_index[0] is source and edge_index[1] is target (PyG convention).
    The inner-product decoder is symmetric; it cannot recover edge direction.
    """
    def __init__(self, gene_encoder, embedding_dim, n_bins=51, gcn_hidden=512,
                 gcn_layers=2, fusion_hidden=512, finetune_embeddings=True):
        super().__init__()
        if min(embedding_dim, n_bins, gcn_hidden, gcn_layers, fusion_hidden) < 1:
            raise ValueError("Model dimensions and layer counts must be positive.")
        self.gene_encoder = gene_encoder
        for parameter in self.gene_encoder.parameters():
            parameter.requires_grad_(finetune_embeddings)
        self.convs = nn.ModuleList([GCNConv(embedding_dim if i == 0 else gcn_hidden, gcn_hidden)
                                    for i in range(gcn_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(gcn_hidden) for _ in range(gcn_layers - 1)])
        self.projection = nn.Linear(gcn_hidden, embedding_dim)
        self.gate = nn.Sequential(nn.Linear(2 * embedding_dim, fusion_hidden), nn.ReLU(),
                                  nn.LayerNorm(fusion_hidden), nn.Linear(fusion_hidden, 1))
        self.expression_decoder = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
                                                nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, n_bins))

    def forward(self, gene_ids, edge_index, edge_weight=None):
        expression = self.gene_encoder(gene_ids)
        network = expression
        for i, convolution in enumerate(self.convs):
            network = convolution(network, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                network = self.norms[i](F.relu(network))
        network = self.projection(network)
        alpha = torch.sigmoid(self.gate(torch.cat([expression, network], dim=-1)))
        fused = alpha * expression + (1 - alpha) * network
        return fused, alpha, F.log_softmax(self.expression_decoder(fused), dim=-1)

    @staticmethod
    def decode_edges(embeddings, edge_index):
        return (embeddings[edge_index[0]] * embeddings[edge_index[1]]).sum(dim=-1)

import copy
import random
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


@dataclass
class TrainConfig:
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lambda_expression: float = 1.
    lambda_network: float = 1.
    gradient_clip: float = 5.
    scheduler_step: int = 20
    scheduler_gamma: float = .5
    negative_ratio: float = 1.
    seed: int = 42
    device: str = "cpu"


@dataclass
class TrainingResult:
    embeddings: np.ndarray
    alpha: np.ndarray
    history: list
    checkpoint: dict


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_graph_embedding(model, gene_ids, gene_histograms, edge_index,
                          edge_weight=None, config=None):
    """Train a supplied model; no file paths or dataset names are assumed.

    Seed before constructing the model for reproducible initialization.
    Returns the best evaluated post-update training state, not validation best.
    """
    cfg = config or TrainConfig()
    if cfg.epochs < 1 or cfg.scheduler_step < 1:
        raise ValueError("epochs and scheduler_step must be positive.")
    values = [cfg.learning_rate, cfg.gradient_clip, cfg.negative_ratio, cfg.scheduler_gamma]
    if not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError("Learning rate, clipping, negative ratio and scheduler gamma must be positive and finite.")
    if not all(np.isfinite(v) and v >= 0 for v in [cfg.weight_decay, cfg.lambda_expression, cfg.lambda_network]) or cfg.lambda_expression + cfg.lambda_network == 0:
        raise ValueError("Loss weights and weight decay must be nonnegative; at least one loss must be active.")
    seed_training(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)
    ids = torch.as_tensor(gene_ids, dtype=torch.long, device=device)
    hist = torch.as_tensor(gene_histograms, dtype=torch.float32, device=device)
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    weights = None if edge_weight is None else torch.as_tensor(edge_weight, dtype=torch.float32, device=device)
    n = ids.numel()
    if ids.ndim != 1 or n < 2 or hist.ndim != 2 or hist.shape[0] != n:
        raise ValueError("Expected gene_ids (G,) and gene_histograms (G, bins), with G >= 2.")
    if not torch.isfinite(hist).all() or (hist < 0).any() or not torch.allclose(hist.sum(1), torch.ones(n, device=device), atol=1e-5):
        raise ValueError("Histograms must be finite probability distributions.")
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] == 0 or edges.min() < 0 or edges.max() >= n:
        raise ValueError("edge_index must contain valid local gene indices with shape (2, E).")
    if (edges[0] == edges[1]).any():
        raise ValueError("Remove self-loops from reconstruction edges; GCN adds its own self-loops.")
    if torch.unique(edges.T, dim=0).shape[0] != edges.shape[1]:
        raise ValueError("Duplicate edges must be aggregated explicitly.")
    if weights is not None and (weights.shape != (edges.shape[1],) or not torch.isfinite(weights).all() or (weights < 0).any()):
        raise ValueError("GCN weights must be finite, nonnegative and match edges.")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma)
    # Prevent reverse positive edges from becoming negatives for a symmetric decoder.
    exclusion = torch.unique(torch.cat([edges, edges.flip(0)], dim=1).T, dim=0).T.contiguous()
    evaluation_negatives = negative_sampling(exclusion, num_nodes=n,
                                             num_neg_samples=max(1, int(edges.shape[1] * cfg.negative_ratio)))
    if cfg.lambda_network > 0 and evaluation_negatives.shape[1] == 0:
        raise ValueError("No nonedges available for the symmetric decoder; use a sparser graph or expression-only loss.")

    def objective(negative_edges):
        z, alpha, logp = model(ids, edges, weights)
        if logp.shape != hist.shape:
            raise ValueError("Model bin count does not match histogram bins.")
        expression_loss = F.kl_div(logp, hist, reduction="batchmean")
        network_loss = z.sum() * 0
        if cfg.lambda_network > 0:
            positive = model.decode_edges(z, edges)
            negative = model.decode_edges(z, negative_edges)
            scores = torch.cat([positive, negative])
            labels = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)])
            network_loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss = cfg.lambda_expression * expression_loss + cfg.lambda_network * network_loss
        return loss, expression_loss, network_loss, z, alpha

    best = float("inf")
    best_state = None
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        negatives = negative_sampling(exclusion, num_nodes=n,
                                      num_neg_samples=max(1, int(edges.shape[1] * cfg.negative_ratio)))
        loss, _, _, _, _ = objective(negatives)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, cfg.gradient_clip)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            evaluated, expression_loss, network_loss, _, alpha = objective(evaluation_negatives)
        if not torch.isfinite(evaluated):
            raise FloatingPointError("Nonfinite evaluation loss.")
        history.append(dict(epoch=epoch, training_loss=float(loss.detach()), evaluation_training_loss=float(evaluated),
                            expression_loss=float(expression_loss), network_loss=float(network_loss), alpha_mean=float(alpha.mean())))
        if float(evaluated) < best:
            best = float(evaluated)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            best_scheduler = copy.deepcopy(scheduler.state_dict())
            best_epoch = epoch
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embeddings, alpha, _ = model(ids, edges, weights)
    checkpoint = dict(model_state_dict=best_state, optimizer_state_dict=best_optimizer,
                      scheduler_state_dict=best_scheduler, config=asdict(cfg), epoch=best_epoch,
                      best_evaluation_training_loss=best,
                      evaluation_negative_edges=evaluation_negatives.cpu())
    return TrainingResult(embeddings.cpu().numpy(), alpha.cpu().numpy().reshape(-1), history, checkpoint)

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import anndata as ad
from scipy import sparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adata", required=True)
    p.add_argument("--network", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--counts-layer", help="Use 'X', 'raw', or an explicit counts layer")
    p.add_argument("--binned-layer", default="X_binned")
    p.add_argument("--gene-column", help="Explicit var column holding gene identifiers")
    p.add_argument("--weight-col", help="Omit for unit weights")
    p.add_argument("--weight-transform", choices=["nonnegative", "absolute"], default="nonnegative")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--n-bins", type=int, default=51)
    p.add_argument("--gcn-hidden", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--freeze-embeddings", action="store_true")
    args = p.parse_args()
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.model import TransformerModel
    from scgpt.preprocess import Preprocessor
    seed_training(args.seed)
    directory = Path(args.model_dir)
    vocabulary = GeneVocab.from_file(str(directory / "vocab.json"))
    for token in ["<pad>", "<cls>", "<eoc>"]:
        if token not in vocabulary:
            vocabulary.append_token(token)
    architecture = json.loads((directory / "args.json").read_text(encoding="utf-8"))
    transformer = TransformerModel(len(vocabulary), architecture["embsize"], architecture["nheads"],
                                   architecture["d_hid"], architecture["nlayers"], vocab=vocabulary,
                                   pad_value=-2, n_input_bins=args.n_bins)
    # Require a matching checkpoint; no silent partial initialization.
    pretrained = torch.load(directory / "best_model.pt", map_location="cpu", weights_only=True)
    transformer.load_state_dict(pretrained, strict=True)
    data = ad.read_h5ad(args.adata)
    if args.counts_layer:
        if args.counts_layer == "raw":
            if data.raw is None:
                raise ValueError("AnnData.raw is absent.")
            data = data.raw.to_adata()
        elif args.counts_layer != "X":
            data.X = data.layers[args.counts_layer].copy()
        values = data.X.data if sparse.issparse(data.X) else np.asarray(data.X)
        if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values, np.round(values), atol=1e-6, rtol=0):
            raise ValueError("Selected counts must be finite, nonnegative integer counts; scaled data are not clipped.")
        Preprocessor(use_key="X", filter_gene_by_counts=0, filter_cell_by_counts=False,
                     normalize_total=1e4, result_normed_key="X_normed", log1p=False,
                     subset_hvg=False, binning=args.n_bins, result_binned_key=args.binned_layer)(data)
    if args.gene_column:
        names = data.var[args.gene_column]
        if names.isna().any():
            raise ValueError("Missing gene identifiers.")
        data.var_names = pd.Index(names.astype(str).str.strip())
    if any(not str(g).strip() for g in data.var_names):
        raise ValueError("Empty gene identifiers.")
    genes, ids, hist, edges, weights, audit = prepare_graph_data(
        data, pd.read_csv(args.network), vocabulary.get_stoi(), binned_layer=args.binned_layer,
        weight_col=args.weight_col, weight_transform=args.weight_transform, n_bins=args.n_bins)
    model_parameters = dict(embedding_dim=architecture["embsize"], n_bins=args.n_bins,
                            gcn_hidden=args.gcn_hidden, gcn_layers=2, fusion_hidden=512,
                            finetune_embeddings=not args.freeze_embeddings)
    model = GraphEmbeddingModel(transformer.encoder, **model_parameters)
    cfg = TrainConfig(epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed, device=args.device)
    result = train_graph_embedding(model, ids, hist, edges, weights, cfg)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.embeddings, index=genes).to_csv(output / "embeddings.csv")
    pd.DataFrame({"gene": genes, "alpha": result.alpha}).to_csv(output / "alpha.csv", index=False)
    pd.DataFrame(result.history).to_csv(output / "training_history.csv", index=False)
    metadata = dict(vars(args), audit=audit, model_parameters=model_parameters,
                    representation="scGPT gene-token encoder + GCN + gated fusion",
                    checkpoint_selection="post-update training loss with fixed negatives; no held-out validation")
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result.checkpoint.update(genes=genes, gene_ids=ids.tolist(), model_parameters=model_parameters,
                             scgpt_architecture=architecture, metadata=metadata)
    torch.save(result.checkpoint, output / "checkpoint.pt")
    # Single-stage output is intentionally not presented as an aligned trajectory.
    np.savez(output / "stage_embedding.npz", embeddings=result.embeddings,
             genes=np.asarray(genes, dtype=str), stage=np.asarray(args.stage))
    print("Saved embedding, checkpoint and training metadata to", output)


if __name__ == "__main__":
    main()
