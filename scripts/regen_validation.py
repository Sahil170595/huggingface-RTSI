"""Regenerate the model-family-held-out validation report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from rtsi_core import RTSI_FEATURES  # noqa: E402
from validation import grouped_cv_scores, stratified_bootstrap_auc  # noqa: E402

CSV_PATH = _ROOT / "substrate" / "rtsi_table.csv"
OUTPUT_PATH = _ROOT / "substrate" / "validation_report.json"
MATERIAL_LOSS_CUTOFF = 0.05


def main() -> int:
    frame = pd.read_csv(CSV_PATH, encoding="utf-8")
    rows = [
        {feature: float(record[feature]) for feature in RTSI_FEATURES}
        for _, record in frame.iterrows()
    ]
    refusal_deltas = [float(value) for value in frame["refusal_rate_delta"]]
    groups = [str(value) for value in frame["family"]]
    danger = [value <= -MATERIAL_LOSS_CUTOFF for value in refusal_deltas]

    scores = grouped_cv_scores(rows, refusal_deltas, groups)
    auc = stratified_bootstrap_auc(danger, scores)
    report = {
        "method": "leave-one-model-family-out",
        "n_cells": len(frame),
        "n_families": len(set(groups)),
        "families": sorted(set(groups)),
        "material_loss_cutoff": MATERIAL_LOSS_CUTOFF,
        "n_material_loss": sum(danger),
        "roc_auc": auc,
        "note": (
            "Every held-out cell is scored with feature weights and normalization "
            "fit without any checkpoint from its model family."
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {OUTPUT_PATH.relative_to(_ROOT)}: "
        f"AUC {auc['auc']:.4f} "
        f"(95% CI {auc['ci_low']:.4f}-{auc['ci_high']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
