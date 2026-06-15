"""Add a hosted MiniCPM4.1-8B cross-check to the external judge benchmark.

The three specialist guard models remain the selective-consensus cohort.
MiniCPM is evaluated separately as a general-reasoning moderation cross-check
on the identical deterministic BeaverTails sample, then appended to the public
per-model table.

Requires ``OPENBMB_API_KEY``. The key and raw model outputs are never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import judges  # noqa: E402
from openbmb_client import (  # noqa: E402
    MINICPM_HF_REPO,
    MINICPM_MODEL_ID,
    batch_chat,
)
from scripts.eval_external_judges import (  # noqa: E402
    DATASET_NAME,
    DATASET_REVISION,
    DATASET_SPLIT,
    OUTPUT_PATH,
    _load_beavertails,
    corpus_sha256,
)

MINICPM_REFERENCE_REVISION = "3a8dfed9c79a45e07dbff95bcd49d792343fa1a3"


def _run_batches(
    corpus: list[dict],
    *,
    batch_size: int,
    retries: int = 3,
) -> tuple[list[str], list[str]]:
    verdicts: list[str] = []
    raw_hash_inputs: list[str] = []
    for start in range(0, len(corpus), batch_size):
        chunk = corpus[start:start + batch_size]
        messages = [
            judges.MINICPM_JUDGE.build_fn(item["prompt"], item["response"])
            for item in chunk
        ]
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                raw_outputs = batch_chat(
                    messages,
                    max_tokens=judges.MINICPM_JUDGE.max_tokens,
                    temperature=0.0,
                )
                break
            except Exception as exc:  # noqa: BLE001 - retry provider failures
                last_error = exc
                if attempt == retries:
                    raise
                time.sleep(attempt)
        else:  # pragma: no cover - defensive; loop either breaks or raises
            raise RuntimeError("MiniCPM batch failed") from last_error

        verdicts.extend(judges.parse_minicpm(raw) for raw in raw_outputs)
        raw_hash_inputs.extend(raw_outputs)
        print(f"MiniCPM items {start + 1}-{start + len(chunk)}/{len(corpus)}")
    return verdicts, raw_hash_inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    corpus = _load_beavertails(args.sample, args.seed)
    expected = [item["expected"] for item in corpus]
    verdicts, raw_outputs = _run_batches(corpus, batch_size=args.batch_size)
    metrics = judges.classification_metrics(expected, verdicts)
    counts = {
        verdict: verdicts.count(verdict)
        for verdict in ("safe", "unsafe", "unclear")
    }

    if not args.output.exists():
        raise FileNotFoundError(
            f"Base specialist benchmark not found: {args.output}"
        )
    result = json.loads(args.output.read_text(encoding="utf-8"))
    if (
        result.get("dataset") != DATASET_NAME
        or result.get("dataset_revision") != DATASET_REVISION
        or result.get("split") != DATASET_SPLIT
        or int(result.get("sample_size", -1)) != len(corpus)
        or int(result.get("seed", -1)) != args.seed
        or result.get("corpus_sha256") != corpus_sha256(corpus)
    ):
        raise ValueError(
            "Existing benchmark metadata does not match the MiniCPM sample."
        )

    entry = {
        "model": MINICPM_HF_REPO,
        "family": "minicpm",
        "role": "general-reasoning moderation cross-check",
        "metrics": metrics,
        "counts": counts,
    }
    per_judge = [
        item
        for item in result.get("per_judge", [])
        if item.get("family") != "minicpm"
    ]
    per_judge.append(entry)
    result["per_judge"] = per_judge
    result.setdefault(
        "cohort_model_ids",
        [judge.model_id for judge in judges.SOTA_JUDGES],
    )
    result["cohort_note"] = (
        "cohort_selective covers the three specialist guard models only; "
        "MiniCPM is reported as a separate reasoning-model cross-check"
    )
    result["minicpm_runtime"] = {
        "provider": "OpenBMB Build Small hosted API",
        "provider_model": MINICPM_MODEL_ID,
        "hub_reference": MINICPM_HF_REPO,
        "hub_reference_revision": MINICPM_REFERENCE_REVISION,
        "provider_revision": "unreported",
        "transport_security": (
            "Sponsor-published shared hackathon endpoint is HTTP-only; "
            "credential is the shared challenge token, not a user token."
        ),
        "temperature": 0.0,
        "max_tokens": judges.MINICPM_JUDGE.max_tokens,
        "thinking": False,
        "raw_outputs_published": False,
        "raw_output_count": len(raw_outputs),
        "raw_output_sha256": hashlib.sha256(
            json.dumps(raw_outputs, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    result["supplemented_at"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"MiniCPM accuracy={metrics['accuracy']:.3f} "
        f"macro_f1={metrics['macro_f1']:.3f} "
        f"coverage={metrics['coverage']:.3f}"
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
