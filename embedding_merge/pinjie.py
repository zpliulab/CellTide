"""Merge ordered per-stage gene embeddings into wide CSV and trajectory NPZ."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def merge_embeddings(files, stages, output_dir, timepoints=None):
    if len(files) < 2 or len(files) != len(stages) or len(set(stages)) != len(stages):
        raise ValueError("Provide at least two files and one unique stage label per file.")
    tables = []
    dimensions = None
    for file in files:
        # Read identifiers as strings to preserve numeric-looking gene IDs.
        raw = pd.read_csv(file, dtype=str)
        if raw.shape[1] < 2 or raw.empty:
            raise ValueError(f"Empty embedding table or missing dimensions: {file}")
        identifiers = raw.iloc[:, 0]
        if identifiers.isna().any() or identifiers.str.strip().eq("").any():
            raise ValueError(f"Missing gene identifier: {file}")
        identifiers = identifiers.str.strip()
        if identifiers.duplicated().any():
            raise ValueError(f"Duplicate gene identifiers: {file}")
        table = raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
        table.index = pd.Index(identifiers, name="Gene")
        if not np.isfinite(table.to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite embedding values: {file}")
        if dimensions is None:
            dimensions = table.columns.tolist()
        elif table.columns.tolist() != dimensions:
            raise ValueError("Embedding dimension names/order must match across stages.")
        tables.append(table)
    common = sorted(set.intersection(*(set(table.index) for table in tables)))
    if not common:
        raise ValueError("No common genes across stages.")
    arrays = [table.loc[common].to_numpy(dtype=float) for table in tables]
    embeddings = np.stack(arrays, axis=0)
    bundle = dict(embeddings=embeddings, genes=np.asarray(common, dtype=str),
                  stage_names=np.asarray(stages, dtype=str))
    if timepoints is not None:
        times = np.asarray(timepoints, dtype=float)
        if times.shape != (len(stages),) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("timepoints must match stages and be finite and strictly increasing.")
        bundle["timepoints"] = times
    merged = pd.concat([table.loc[common].add_prefix(f"emb{i+1}_")
                        for i, table in enumerate(tables)], axis=1)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "merged_embedding_common_genes.csv")
    np.savez_compressed(output / "trajectory_embeddings.npz", **bundle)
    audit = dict(stage_names=list(stages), shape=list(embeddings.shape),
                 dimension_columns=dimensions, gene_order="sorted common gene identifiers",
                 coordinate_alignment_performed=False,
                 stages=[dict(stage=name, file=str(file), input_genes=len(table),
                              discarded_genes=sorted(set(table.index) - set(common)))
                         for name, file, table in zip(stages, files, tables)])
    (output / "merge_metadata.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return bundle, merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", required=True, help="CSV files in biological stage order")
    parser.add_argument("--stages", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timepoints", nargs="+", type=float)
    args = parser.parse_args()
    bundle, _ = merge_embeddings(args.files, args.stages, args.output, args.timepoints)
    print("Saved CSV and NPZ; embedding shape:", bundle["embeddings"].shape)


if __name__ == "__main__":
    main()
