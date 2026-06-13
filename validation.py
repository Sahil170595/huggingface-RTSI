"""Validation utilities for the fixed QuantSafe substrate.

The deployed score is calibrated on model/quantization cells, so row-level
leave-one-out can overstate transfer when sibling checkpoints share a family.
This module provides a stricter leave-one-model-family-out evaluation and a
deterministic stratified bootstrap interval for its ROC AUC.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from rtsi_core import RTSI_FEATURES, fit_weights


def binary_roc_auc(labels: Sequence[bool | int], scores: Sequence[float]) -> float:
    """Compute binary ROC AUC from pairwise positive/negative score ordering."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must align")
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    comparisons = (pos[:, None] > neg[None, :]).astype(np.float64)
    comparisons += 0.5 * (pos[:, None] == neg[None, :])
    return float(comparisons.mean())


def _score_with_training_fold(
    row: Mapping[str, float],
    train_rows: Sequence[Mapping[str, float]],
    weights: Mapping[str, float],
) -> float:
    score = 0.0
    for feature in RTSI_FEATURES:
        train_abs = np.abs(
            np.asarray([float(item[feature]) for item in train_rows], dtype=np.float64)
        )
        lo = float(np.nanmin(train_abs))
        hi = float(np.nanmax(train_abs))
        value = abs(float(row[feature]))
        normalized = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0)) if hi > lo else 0.0
        score += float(weights[feature]) * normalized
    return score


def grouped_cv_scores(
    rows: Sequence[Mapping[str, float]],
    refusal_deltas: Sequence[float],
    groups: Sequence[str],
) -> list[float]:
    """Score every row while holding its entire model family out of fitting."""
    n = len(rows)
    if n != len(refusal_deltas) or n != len(groups):
        raise ValueError("rows, refusal_deltas, and groups must align")
    if len(set(groups)) < 2:
        raise ValueError("grouped validation requires at least two groups")

    scores = [0.0] * n
    for held_group in dict.fromkeys(groups):
        train_indices = [i for i, group in enumerate(groups) if group != held_group]
        test_indices = [i for i, group in enumerate(groups) if group == held_group]
        train_rows = [rows[i] for i in train_indices]
        train_targets = [float(refusal_deltas[i]) for i in train_indices]
        weights = fit_weights(train_rows, train_targets)
        for index in test_indices:
            scores[index] = _score_with_training_fold(rows[index], train_rows, weights)
    return scores


def stratified_bootstrap_auc(
    labels: Sequence[bool | int],
    scores: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 20260613,
) -> dict[str, float | int]:
    """Return a deterministic 95% stratified-bootstrap interval for ROC AUC."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must align")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        return {
            "auc": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_resamples": n_resamples,
            "seed": seed,
        }

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = np.concatenate(
            (
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            )
        )
        samples[i] = binary_roc_auc(y[indices], s[indices])

    return {
        "auc": binary_roc_auc(y, s),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n_resamples": n_resamples,
        "seed": seed,
    }
