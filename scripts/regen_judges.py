"""scripts/regen_judges.py — regenerate the Judge Agreement cache.

Runs the current-generation safety-judge cohort (judges.SOTA_JUDGES:
Qwen3Guard-Gen-0.6B + Granite-Guardian-3.3-8b, two distinct families) over the
bundled 40-prompt corpus via the deployed Modal /judge endpoint, recomputes
Cohen's kappa, and conditionally overwrites substrate/judge_results.json.

SAFETY VALVE: the cache is only replaced if the new cohort lands in the
RELIABLE band. If the SOTA judges disagree more than the 2024 cohort and kappa
drops below RELIABLE, the validated kappa=0.74 cache is KEPT and the script
exits non-zero with a clear message — we never ship a worse headline number.

The /judge endpoint applies each judge's own moderation chat template
server-side (Granite Guardian needs guardian_config), so this script only
POSTs {model, prompt, response} and parses the raw completion with the
judge's parse_fn — keeping verdict parsing unit-testable without a GPU.

Usage (PowerShell):
    $env:MODAL_ENDPOINT = "https://<workspace>--debate-generate.modal.run"
    $env:MODAL_TOKEN    = "<the quantsafe-auth token>"
    # Optional: override the /judge URL if it isn't the /generate URL with the
    # 'generate' suffix swapped for 'judge':
    $env:MODAL_JUDGE_ENDPOINT = "https://<workspace>--debate-judge.modal.run"
    python scripts/regen_judges.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import judges  # noqa: E402

CORPUS_PATH = _ROOT / "substrate" / "judge_corpus.json"
RESULTS_PATH = _ROOT / "substrate" / "judge_results.json"
TIMEOUT_S = 300


def _judge_endpoint() -> str:
    explicit = os.environ.get("MODAL_JUDGE_ENDPOINT")
    if explicit:
        return explicit
    gen = os.environ.get("MODAL_ENDPOINT")
    if not gen:
        raise SystemExit(
            "Set MODAL_ENDPOINT (the /generate URL) or MODAL_JUDGE_ENDPOINT "
            "(the /judge URL). Deploy modal_app.py first."
        )
    # Modal labels become URL suffixes: ...--debate-generate -> ...--debate-judge
    if "generate" in gen:
        return gen.replace("generate", "judge")
    raise SystemExit(
        "Could not derive the /judge URL from MODAL_ENDPOINT; set "
        "MODAL_JUDGE_ENDPOINT explicitly."
    )


def _post_judge(endpoint: str, headers: dict, model: str, prompt: str, response: str) -> str:
    import requests

    payload = {"model": model, "prompt": prompt, "response": response, "max_new_tokens": 48}
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=TIMEOUT_S)
    if not 200 <= resp.status_code < 300:
        try:
            detail = resp.json().get("detail") or resp.text
        except Exception:
            detail = resp.text
        raise RuntimeError(f"/judge error ({resp.status_code}): {detail}")
    return str(resp.json()["text"])


def main() -> int:
    endpoint = _judge_endpoint()
    headers: dict[str, str] = {}
    token = os.environ.get("MODAL_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    corpus = judges.load_judge_corpus(str(CORPUS_PATH))
    if not corpus:
        raise SystemExit(f"empty corpus at {CORPUS_PATH}")
    print(f"Corpus: {len(corpus)} items")
    print(f"/judge endpoint: {endpoint}")

    cohort = judges.SOTA_JUDGES
    expected = [str(item.get("expected", "")) for item in corpus]
    start = time.perf_counter()
    judge_reports: list[dict] = []
    verdicts_by_judge: list[list[str]] = []

    for j in cohort:
        print(f"\nJudge: {j.model_id}")
        verdict_vector: list[str] = []
        counts = {"safe": 0, "unsafe": 0, "unclear": 0}
        for i, item in enumerate(corpus):
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            try:
                raw = _post_judge(endpoint, headers, j.model_id, prompt, response)
                verdict = j.parse_fn(raw)
            except Exception as exc:  # one flaky call -> unclear, pull kappa down honestly
                print(f"  item {i}: ERROR {exc}", file=sys.stderr)
                verdict = "unclear"
            verdict_vector.append(verdict)
            counts[verdict] += 1
            print(f"  item {i:2d}: {verdict}")
        verdicts_by_judge.append(verdict_vector)
        judge_reports.append({
            "model": j.model_id,
            "counts": counts,
            "verdict_vector": verdict_vector,
            "metrics": judges.classification_metrics(expected, verdict_vector),
        })

    agreement = judges.compute_agreement(verdicts_by_judge)
    selective_metrics = judges.selective_consensus_metrics(expected, verdicts_by_judge)
    elapsed_s = time.perf_counter() - start

    print(
        f"\nkappa = {agreement['kappa']:.4f} ({agreement['method']}) "
        f"-> band {agreement['band']}"
    )
    for rep in judge_reports:
        metrics = rep["metrics"]
        print(
            f"  {rep['model']}: {rep['counts']}; "
            f"accuracy={metrics['accuracy']:.3f}, macro-F1={metrics['macro_f1']:.3f}"
        )
    print(
        "  unanimous non-unclear panel: "
        f"coverage={selective_metrics['coverage']:.3f}, "
        f"accuracy={selective_metrics['accuracy']:.3f}"
    )

    if agreement["band"] != judges.BAND_RELIABLE:
        print(
            f"\nNOT RELIABLE (band={agreement['band']}). "
            "Keeping the validated 2024 cache (kappa=0.74) untouched.",
            file=sys.stderr,
        )
        if agreement.get("invalid_reason"):
            print(f"  reason: {agreement['invalid_reason']}", file=sys.stderr)
        return 2

    result = {
        "agreement": agreement,
        "judges": judge_reports,
        # zones describe the CORPUS items (clear_safe / clear_unsafe /
        # borderline), not the judges — carry them straight from the corpus so
        # app._agreement_breakdown buckets disagreements by the right zone.
        "zones": [item.get("zone", "unlabeled") for item in corpus],
        "n_items": len(corpus),
        "certifier_pass": agreement["band"] == judges.BAND_RELIABLE,
        "gold_validation": {
            "label_source": "curated expected labels in judge_corpus.json",
            "selective_consensus": selective_metrics,
        },
        "source": "scripts/regen_judges.py via Modal /judge endpoint (SOTA cohort)",
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH.relative_to(_ROOT)} in {elapsed_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
