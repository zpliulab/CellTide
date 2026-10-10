"""Transition detection independent of dataset names and file formats."""
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class TransitionResult:
    scores: pd.DataFrame
    best_interval: Optional[int]
    bootstrap_peaks: np.ndarray
    peak_frequency: np.ndarray
    velocities: np.ndarray


def validate_embeddings(embeddings):
    x = np.asarray(embeddings, dtype=float)
    if x.ndim != 3 or x.shape[0] < 2 or min(x.shape[1:]) < 1:
        raise ValueError("Expected embeddings with shape (stages >= 2, genes >= 1, dimensions >= 1).")
    if not np.isfinite(x).all():
        raise ValueError("Embeddings must contain only finite values.")
    return x


def safe_zscore(values):
    values = np.asarray(values, dtype=float)
    sd = values.std()
    return np.zeros_like(values) if sd <= 1e-12 else (values - values.mean()) / sd


def row_normalize(matrix):
    matrix = sparse.csr_matrix(matrix, dtype=float)
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("Network weights must be finite and nonnegative.")
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1., totals, out=np.zeros_like(totals), where=totals > 0)
    return sparse.diags(inverse) @ matrix


def _metrics(velocities):
    magnitude = np.linalg.norm(velocities, axis=2)
    totals = magnitude.sum(axis=1, keepdims=True)
    p = np.divide(magnitude, totals, out=np.zeros_like(magnitude), where=totals > 0)
    log_p = np.zeros_like(p)
    np.log(p, out=log_p, where=p > 0)
    return magnitude.mean(axis=1), -(p * log_p).sum(axis=1)


def _unique_peak(score):
    candidates = np.flatnonzero(np.isclose(score, score.max(), rtol=1e-10, atol=1e-12))
    return int(candidates[0]) if len(candidates) == 1 else None


def detect_transitions(embeddings, stage_names: Optional[Sequence[str]] = None,
                       adjacency=None, smoothing_alpha=0., timepoints=None,
                       bootstrap_n=500, random_seed=42):
    """Score adjacent stages using z(speed) + z(velocity-magnitude entropy).

    embeddings: (T, G, D), with identical gene order and aligned coordinates.
    adjacency: optional sequence of T-1 matrices, in source-row convention.
        Smoothing averages neighboring velocities; it is not TF propagation.
    timepoints: optional numeric sampling times; velocity is divided by dt.
        Without these, velocity means displacement per ordered interval.
    Bootstrap resamples genes after smoothing, not cells or networks.
    Tied peaks return None; tied bootstrap peaks are recorded as -1.
    """
    x = validate_embeddings(embeddings)
    t, g, _ = x.shape
    names = list(map(str, stage_names)) if stage_names is not None else [str(i) for i in range(t)]
    if len(names) != t or len(set(names)) != t:
        raise ValueError("stage_names must contain one unique name per stage.")
    if not np.isfinite(smoothing_alpha) or not 0 <= smoothing_alpha <= 1:
        raise ValueError("smoothing_alpha must be in [0, 1].")
    if not isinstance(bootstrap_n, (int, np.integer)) or bootstrap_n < 0:
        raise ValueError("bootstrap_n must be a nonnegative integer.")
    dt = np.ones(t - 1)
    if timepoints is not None:
        times = np.asarray(timepoints, dtype=float)
        if times.shape != (t,) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("timepoints must be finite, strictly increasing, and match stages.")
        dt = np.diff(times)
    velocities = np.diff(x, axis=0) / dt[:, None, None]
    if smoothing_alpha > 0 and adjacency is None:
        raise ValueError("adjacency is required when smoothing_alpha > 0.")
    if adjacency is not None:
        if len(adjacency) != t - 1:
            raise ValueError("Provide one smoothing adjacency per interval.")
        for i, matrix in enumerate(adjacency):
            matrix = row_normalize(matrix)
            if matrix.shape != (g, g):
                raise ValueError("Adjacency shape must match the gene axis.")
            velocities[i] = (1 - smoothing_alpha) * velocities[i] + smoothing_alpha * (matrix @ velocities[i])
    speed, entropy = _metrics(velocities)
    score = safe_zscore(speed) + safe_zscore(entropy)
    # A single interval cannot support relative peak detection.
    best = _unique_peak(score) if t > 2 else None
    rng = np.random.default_rng(random_seed)
    peaks = np.full(bootstrap_n, -1, dtype=int)
    for b in range(bootstrap_n):
        indices = rng.integers(0, g, size=g)
        bs, be = _metrics(velocities[:, indices])
        peak = _unique_peak(safe_zscore(bs) + safe_zscore(be)) if t > 2 else None
        if peak is not None:
            peaks[b] = peak
    frequency = np.array([(peaks == i).mean() for i in range(t - 1)]) if bootstrap_n else np.full(t - 1, np.nan)
    frame = pd.DataFrame({"interval_index": np.arange(t - 1), "source_stage": names[:-1],
                          "target_stage": names[1:], "mean_speed": speed,
                          "velocity_entropy": entropy, "tipping_score": score,
                          "bootstrap_peak_frequency": frequency})
    return TransitionResult(frame, best, peaks, frequency, velocities)
