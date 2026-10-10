"""Run from the project root: python -m examples.run_analysis --help."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from transition.transition import detect_transitions
from perturbation.perturbation import rank_regulators


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="NPZ containing embeddings (T,G,D), genes, stage_names")
    parser.add_argument("--network", required=True)
    parser.add_argument("--weight-col", required=True, help="Explicit network stage column, e.g. CMI_3")
    parser.add_argument("--interval", type=int, help="Zero-based interval; default: unique detected peak")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--bootstrap-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--propagation", choices=["outgoing", "legacy"], default="outgoing")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.embeddings, allow_pickle=False) as bundle:
        x = bundle["embeddings"]
        genes = bundle["genes"].astype(str).tolist()
        stages = bundle["stage_names"].astype(str).tolist()
        times = bundle["timepoints"] if "timepoints" in bundle else None
    transition = detect_transitions(x, stages, timepoints=times, bootstrap_n=args.bootstrap_n, random_seed=args.seed)
    interval = args.interval if args.interval is not None else transition.best_interval
    if interval is None:
        raise ValueError("No unique transition peak; explicitly select --interval to run perturbation analysis.")
    perturbation = rank_regulators(x, genes, pd.read_csv(args.network), interval,
                                  weight_col=args.weight_col, top_k=args.top_k, propagation=args.propagation)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    transition.scores.to_csv(output / "transition_scores.csv", index=False)
    pd.DataFrame({"interval_index": transition.bootstrap_peaks}).to_csv(output / "bootstrap_peaks.csv", index=False)
    perturbation.tf_scores.to_csv(output / "tf_scores.csv", index=False)
    perturbation.target_scores.to_csv(output / "target_scores.csv", index=False)
    perturbation.selected_edges.to_csv(output / "selected_edges.csv", index=False)
    metadata = dict(vars(args), stage_names=stages, selected_interval=int(interval),
                    detected_interval=transition.best_interval, selected_tfs=perturbation.selected_tfs,
                    excluded_edge_count=perturbation.excluded_edge_count,
                    entropy_definition="Shannon entropy of normalized gene velocity magnitudes")
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Results saved to", output)


if __name__ == "__main__":
    main()
