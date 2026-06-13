"""scripts/regen_debate.py — regenerate the cached Constitutional Debate.

Runs the current production debate cohort (app.LIVE_DEBATE_MODELS, three
distinct model families) on the deployed Modal GPU backend and overwrites
substrate/debate_examples.json with the real transcript + consensus.

Why three models: an odd cohort always yields a strict majority, so the
consensus is genuine (2-1 or 3-0) rather than a 1-1 tie resolved by the
safety-first tie-break — the demo story matches the data without an asterisk.

The contested question is reused verbatim from the existing cache (a
well-calibrated MODERATE-band scenario); only the debaters change.

Usage (PowerShell):
    $env:MODAL_ENDPOINT = "https://<workspace>--debate-generate.modal.run"
    $env:MODAL_TOKEN    = "<the quantsafe-auth token>"
    python scripts/regen_debate.py

Writes substrate/debate_examples.json on success. Exits non-zero (leaving the
existing cache untouched) if the Modal backend errors or the result looks
degenerate (every model failed to generate).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make the SPACE root importable regardless of cwd.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import app  # noqa: E402  (source of truth for LIVE_DEBATE_MODELS)
import debate  # noqa: E402

CACHE_PATH = _ROOT / "substrate" / "debate_examples.json"
ROUNDS = 2


def _load_existing_question() -> str:
    """Reuse the existing contested question — it's a calibrated MODERATE case."""
    existing = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    q = existing.get("question", "").strip()
    if not q:
        raise SystemExit("existing cache has no question to reuse; aborting")
    return q


def main() -> int:
    if not os.environ.get("MODAL_ENDPOINT"):
        raise SystemExit(
            "MODAL_ENDPOINT is not set. Deploy modal_app.py and export the "
            "/generate URL first (see modal_app.py runbook)."
        )

    question = _load_existing_question()
    models = list(app.LIVE_DEBATE_MODELS)
    print(f"Regenerating debate: {len(models)} models, {ROUNDS} rounds")
    for m in models:
        print(f"  · {m}")

    result = debate.run_debate(
        question,
        models,
        backend="modal",
        rounds=ROUNDS,
        on_event=lambda ev: print(
            f"  [{ev.get('type')}] "
            + (f"{ev.get('model')} -> {ev.get('stance')}" if ev.get("type") == "model_response" else "")
        ),
    )

    # Degeneracy guard: refuse to cache a transcript where every model errored.
    final = result["rounds"][-1]["responses"]
    errored = sum(1 for r in final if r["text"].startswith("[generation error"))
    if errored == len(final):
        print("ERROR: every model failed to generate; cache left untouched.", file=sys.stderr)
        return 1

    consensus = result["consensus"]
    print(
        f"\nConsensus: {consensus['verdict']} "
        f"@ {consensus['agreement']:.2f} agreement "
        f"({consensus['vote_breakdown']})"
    )
    if result.get("quantization"):
        print(f"Precision disclosed by endpoint: {result['quantization']}")

    CACHE_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {CACHE_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
