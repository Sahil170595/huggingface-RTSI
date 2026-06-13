"""Tests for stricter family-held-out validation and judge gold metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rtsi_core import RTSI_FEATURES
from validation import binary_roc_auc, grouped_cv_scores, stratified_bootstrap_auc


def test_binary_auc_perfect_and_reversed():
    labels = [0, 0, 1, 1]
    assert binary_roc_auc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert binary_roc_auc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_grouped_scores_hold_out_whole_family():
    rows = [
        {feature: float(i + offset) for offset, feature in enumerate(RTSI_FEATURES)}
        for i in range(6)
    ]
    scores = grouped_cv_scores(
        rows,
        refusal_deltas=[0.0, -0.1, 0.0, -0.2, 0.0, -0.3],
        groups=["a", "a", "b", "b", "c", "c"],
    )
    assert len(scores) == 6
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_bootstrap_is_deterministic():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.4, 0.2, 0.7, 0.9, 0.6]
    a = stratified_bootstrap_auc(labels, scores, n_resamples=100, seed=7)
    b = stratified_bootstrap_auc(labels, scores, n_resamples=100, seed=7)
    assert a == b
    assert a["ci_low"] <= a["auc"] <= a["ci_high"]


def test_frozen_family_held_out_report_matches_recomputation():
    frame = pd.read_csv(_ROOT / "substrate" / "rtsi_table.csv", encoding="utf-8")
    rows = [
        {feature: float(record[feature]) for feature in RTSI_FEATURES}
        for _, record in frame.iterrows()
    ]
    deltas = frame["refusal_rate_delta"].astype(float).tolist()
    labels = [value <= -0.05 for value in deltas]
    scores = grouped_cv_scores(rows, deltas, frame["family"].astype(str).tolist())
    report = json.loads(
        (_ROOT / "substrate" / "validation_report.json").read_text(encoding="utf-8")
    )
    assert abs(binary_roc_auc(labels, scores) - report["roc_auc"]["auc"]) < 1e-12
    assert report["n_families"] == 4
    assert report["method"] == "leave-one-model-family-out"
