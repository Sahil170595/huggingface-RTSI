"""Regenerate judge measurements as immutable, auditable run artifacts.

A normal invocation runs the three-model SOTA_JUDGES cohort through the
authenticated Modal /judge backend and writes one new artifact under
``substrate/judge_runs/``. It never overwrites ``substrate/judge_results.json``
and never suppresses a run because its agreement or accuracy is unfavorable.

Promotion is a separate, explicit operation:

    python scripts/regen_judges.py
    python scripts/regen_judges.py --promote substrate/judge_runs/<run>.json

Promotion validates the artifact against the current corpus, exact pinned model
revisions, expected model set/order, verdict digest, and recomputed metrics
before deterministically projecting it into ``substrate/judge_results.json``.

The artifact contains no bearer token, endpoint URL, prompt text, response text,
or raw model output. Raw outputs are represented by SHA-256 digests and byte
lengths. The backend is identified only by the stable label ``modal-judge``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import judges  # noqa: E402
from model_revisions import model_revision  # noqa: E402

CORPUS_PATH = _ROOT / "substrate" / "judge_corpus.json"
RESULTS_PATH = _ROOT / "substrate" / "judge_results.json"
RUNS_DIR = _ROOT / "substrate" / "judge_runs"
TIMEOUT_S = 300
ARTIFACT_SCHEMA_VERSION = 1
BACKEND_LABEL = "modal-judge"
SOURCE_LABEL = "scripts/regen_judges.py via Modal /judge endpoint (SOTA cohort)"

PostJudge = Callable[[str, dict[str, str], str, str, str, int], tuple[str, str]]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _code_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip().lower()
    if len(value) == 40 and all(ch in "0123456789abcdef" for ch in value):
        return value
    return None


def _corpus_sha256(path: Path = CORPUS_PATH) -> str:
    return _sha256_bytes(path.read_bytes())


def _judge_endpoint() -> str:
    explicit = os.environ.get("MODAL_JUDGE_ENDPOINT")
    if explicit:
        return explicit
    generate_endpoint = os.environ.get("MODAL_ENDPOINT")
    if not generate_endpoint:
        raise SystemExit(
            "Set MODAL_ENDPOINT or MODAL_JUDGE_ENDPOINT after deploying modal_app.py."
        )
    if "generate" in generate_endpoint:
        return generate_endpoint.replace("generate", "judge")
    raise SystemExit(
        "Could not derive the /judge endpoint; set MODAL_JUDGE_ENDPOINT explicitly."
    )


def _post_judge(
    endpoint: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
    response: str,
    max_new_tokens: int,
) -> tuple[str, str]:
    """Return raw output and the backend-reported dtype/quantization label."""
    import requests

    request_body = {
        "model": model,
        "prompt": prompt,
        "response": response,
        "max_new_tokens": max_new_tokens,
    }
    resp = requests.post(
        endpoint,
        json=request_body,
        headers=headers,
        timeout=TIMEOUT_S,
    )
    if not 200 <= resp.status_code < 300:
        try:
            detail = resp.json().get("detail") or resp.text
        except Exception:
            detail = resp.text
        raise RuntimeError(f"/judge error ({resp.status_code}): {detail}")
    payload = resp.json()
    raw = str(payload["text"])
    reported = payload.get("dtype") or payload.get("quantization") or "unreported"
    return raw, str(reported)


def _generation_settings(judge: judges.SafetyJudge) -> dict[str, Any]:
    return {
        "max_new_tokens": judge.max_tokens,
        "do_sample": False,
        "temperature": None,
        "response_field": "text",
    }


def _verdict_digest(judge_reports: list[dict[str, Any]]) -> str:
    digest_input = [
        {
            "model": report["model"],
            "verdict_vector": report["verdict_vector"],
        }
        for report in judge_reports
    ]
    return _sha256_json(digest_input)


def _public_result(
    corpus: list[dict[str, Any]],
    judge_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    verdicts = [report["verdict_vector"] for report in judge_reports]
    expected = [str(item.get("expected", "")) for item in corpus]
    zones = [str(item.get("zone", "unlabeled")) for item in corpus]
    agreement = judges.compute_agreement(verdicts)
    selective = judges.selective_consensus_metrics(
        expected,
        verdicts,
    )
    accuracy_defined = math.isfinite(selective["accuracy"])
    if not accuracy_defined:
        selective = dict(selective)
        selective["accuracy"] = 0.0
        selective["accuracy_ci_low"] = 0.0
        selective["accuracy_ci_high"] = 0.0
    selective["accuracy_defined"] = accuracy_defined
    return {
        "agreement": agreement,
        "statistical_uncertainty": {
            "kappa": judges.stratified_bootstrap_kappa_ci(verdicts, zones),
            "top_two_accuracy": judges.paired_top_two_mcnemar(
                expected, judge_reports
            ),
        },
        "judges": judge_reports,
        "zones": zones,
        "n_items": len(corpus),
        "certifier_pass": agreement["band"] == judges.BAND_RELIABLE,
        "gold_validation": {
            "label_source": "project-curated expected labels in judge_corpus.json",
            "selective_consensus": selective,
        },
        "source": SOURCE_LABEL,
    }


def build_run_artifact(
    *,
    corpus: list[dict[str, Any]],
    corpus_sha256: str,
    endpoint: str,
    headers: dict[str, str],
    post_judge: PostJudge = _post_judge,
    cohort: list[judges.SafetyJudge] | None = None,
    generated_at: datetime | None = None,
    code_sha: str | None = None,
) -> dict[str, Any]:
    """Execute one complete cohort run and return its immutable artifact."""
    cohort = list(cohort if cohort is not None else judges.SOTA_JUDGES)
    generated_at = generated_at or _utc_now()
    expected = [str(item.get("expected", "")) for item in corpus]
    started = time.perf_counter()
    judge_reports: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    model_revisions = {judge.model_id: model_revision(judge.model_id) for judge in cohort}

    for judge in cohort:
        print(f"\nJudge: {judge.model_id}")
        verdict_vector: list[str] = []
        counts = {"safe": 0, "unsafe": 0, "unclear": 0}
        item_observations: list[dict[str, Any]] = []
        reported_values: set[str] = set()

        for index, item in enumerate(corpus):
            raw = ""
            reported = "unreported"
            error_type: str | None = None
            received_output = False
            try:
                raw, reported = post_judge(
                    endpoint,
                    headers,
                    judge.model_id,
                    str(item.get("prompt", "")),
                    str(item.get("response", "")),
                    judge.max_tokens,
                )
                received_output = True
                verdict = judge.parse_fn(raw)
            except Exception as exc:
                error_type = type(exc).__name__
                verdict = "unclear"
                print(
                    f"  item {index}: ERROR {error_type}; recorded as unclear",
                    file=sys.stderr,
                )

            reported_values.add(reported)
            verdict_vector.append(verdict)
            counts[verdict] += 1
            item_observations.append(
                {
                    "index": index,
                    "item_id": str(item.get("id", index)),
                    "verdict": verdict,
                    "received_output": received_output,
                    "raw_output_sha256": _sha256_bytes(raw.encode("utf-8")),
                    "raw_output_bytes": len(raw.encode("utf-8")),
                    "reported_dtype_or_quantization": reported,
                    "error_type": error_type,
                }
            )
            print(f"  item {index:2d}: {verdict}")

        report = {
            "model": judge.model_id,
            "counts": counts,
            "verdict_vector": verdict_vector,
            "metrics": judges.classification_metrics(expected, verdict_vector),
        }
        judge_reports.append(report)
        observations.append(
            {
                "model": judge.model_id,
                "revision": model_revisions[judge.model_id],
                "generation_settings": _generation_settings(judge),
                "reported_dtype_or_quantization": sorted(reported_values),
                "items": item_observations,
            }
        )

    elapsed_s = time.perf_counter() - started
    result = _public_result(corpus, judge_reports)
    verdict_digest = _verdict_digest(judge_reports)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run": {
            "generated_at_utc": _format_utc(generated_at),
            "backend": BACKEND_LABEL,
            "endpoint_label": "judge",
            "code_sha": code_sha,
            "corpus_sha256": corpus_sha256,
            "corpus_path": "substrate/judge_corpus.json",
            "model_revisions": model_revisions,
            "generation_settings": {
                judge.model_id: _generation_settings(judge) for judge in cohort
            },
            "reported_dtype_or_quantization": {
                observation["model"]: observation[
                    "reported_dtype_or_quantization"
                ]
                for observation in observations
            },
            "elapsed_s": elapsed_s,
            "verdict_digest_sha256": verdict_digest,
        },
        "result": result,
        "observations": observations,
    }


def write_run_artifact(
    artifact: dict[str, Any],
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Create a new artifact file without permitting overwrite."""
    generated = artifact["run"]["generated_at_utc"]
    timestamp = generated.replace("-", "").replace(":", "")
    digest_prefix = artifact["run"]["verdict_digest_sha256"][:12]
    runs_dir.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        artifact,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    duplicate = 0
    while True:
        suffix = "" if duplicate == 0 else f"-{duplicate:02d}"
        path = runs_dir / f"judge-run-{timestamp}-{digest_prefix}{suffix}.json"
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
            return path
        except FileExistsError:
            duplicate += 1


def _validate_artifact(
    artifact: dict[str, Any],
    *,
    corpus_path: Path = CORPUS_PATH,
    cohort: list[judges.SafetyJudge] | None = None,
) -> dict[str, Any]:
    cohort = list(cohort if cohort is not None else judges.SOTA_JUDGES)
    expected_models = [judge.model_id for judge in cohort]

    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported judge run artifact schema")
    run = artifact.get("run")
    result = artifact.get("result")
    observations = artifact.get("observations")
    if not isinstance(run, dict) or not isinstance(result, dict):
        raise ValueError("artifact is missing run/result objects")
    if not isinstance(observations, list):
        raise ValueError("artifact is missing observations")
    if run.get("backend") != BACKEND_LABEL:
        raise ValueError("artifact backend label does not match")
    if run.get("endpoint_label") != "judge":
        raise ValueError("artifact endpoint label does not match")
    try:
        generated_at = datetime.fromisoformat(
            str(run.get("generated_at_utc", "")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("artifact UTC timestamp is invalid") from exc
    if generated_at.tzinfo is None or generated_at.utcoffset() != timezone.utc.utcoffset(
        generated_at
    ):
        raise ValueError("artifact timestamp must be UTC")
    code_sha = run.get("code_sha")
    if code_sha is not None and (
        not isinstance(code_sha, str)
        or len(code_sha) != 40
        or any(ch not in "0123456789abcdef" for ch in code_sha)
    ):
        raise ValueError("artifact code SHA is invalid")
    elapsed_s = run.get("elapsed_s")
    if (
        isinstance(elapsed_s, bool)
        or not isinstance(elapsed_s, (int, float))
        or not math.isfinite(elapsed_s)
        or elapsed_s < 0
    ):
        raise ValueError("artifact elapsed time is invalid")
    if run.get("corpus_sha256") != _corpus_sha256(corpus_path):
        raise ValueError("artifact corpus SHA-256 does not match current corpus")

    artifact_revisions = run.get("model_revisions")
    expected_revisions = {model: model_revision(model) for model in expected_models}
    if artifact_revisions != expected_revisions:
        raise ValueError("artifact model revisions do not match pinned revisions")

    judge_reports = result.get("judges")
    if not isinstance(judge_reports, list):
        raise ValueError("artifact result is missing judge reports")
    result_models = [report.get("model") for report in judge_reports]
    observation_models = [observation.get("model") for observation in observations]
    if result_models != expected_models or observation_models != expected_models:
        raise ValueError(
            "artifact model set/order does not match the configured SOTA cohort"
        )
    if run.get("generation_settings") != {
        judge.model_id: _generation_settings(judge) for judge in cohort
    }:
        raise ValueError("artifact generation settings do not match the cohort")

    corpus = judges.load_judge_corpus(str(corpus_path))
    expected = [str(item.get("expected", "")) for item in corpus]
    for judge, report, observation in zip(cohort, judge_reports, observations):
        vector = report.get("verdict_vector")
        if not isinstance(vector, list) or len(vector) != len(corpus):
            raise ValueError(f"invalid verdict vector for {judge.model_id}")
        if any(verdict not in judges.VERDICTS for verdict in vector):
            raise ValueError(f"invalid verdict label for {judge.model_id}")
        counts = {label: vector.count(label) for label in judges.VERDICTS}
        if report.get("counts") != counts:
            raise ValueError(f"verdict counts do not match for {judge.model_id}")
        if report.get("metrics") != judges.classification_metrics(expected, vector):
            raise ValueError(f"classification metrics do not match for {judge.model_id}")
        if observation.get("revision") != expected_revisions[judge.model_id]:
            raise ValueError(f"observation revision mismatch for {judge.model_id}")
        if observation.get("generation_settings") != _generation_settings(judge):
            raise ValueError(
                f"observation generation settings mismatch for {judge.model_id}"
            )
        items = observation.get("items")
        if not isinstance(items, list) or len(items) != len(corpus):
            raise ValueError(f"invalid observations for {judge.model_id}")
        for index, (item, item_observation, verdict) in enumerate(
            zip(corpus, items, vector)
        ):
            if item_observation.get("index") != index:
                raise ValueError(f"observation index mismatch for {judge.model_id}")
            if item_observation.get("item_id") != str(item.get("id", index)):
                raise ValueError(f"observation item id mismatch for {judge.model_id}")
            if item_observation.get("verdict") != verdict:
                raise ValueError(f"observation verdict mismatch for {judge.model_id}")
            received_output = item_observation.get("received_output")
            if not isinstance(received_output, bool):
                raise ValueError(f"invalid output status for {judge.model_id}")
            raw_hash = item_observation.get("raw_output_sha256")
            if (
                not isinstance(raw_hash, str)
                or len(raw_hash) != 64
                or any(ch not in "0123456789abcdef" for ch in raw_hash)
            ):
                raise ValueError(f"invalid raw output hash for {judge.model_id}")
            raw_bytes = item_observation.get("raw_output_bytes")
            if (
                isinstance(raw_bytes, bool)
                or not isinstance(raw_bytes, int)
                or raw_bytes < 0
            ):
                raise ValueError(f"invalid raw output length for {judge.model_id}")
            if not received_output and (
                raw_bytes != 0 or raw_hash != _sha256_bytes(b"")
            ):
                raise ValueError(f"failed output metadata mismatch for {judge.model_id}")
            error_type = item_observation.get("error_type")
            if error_type is not None and not isinstance(error_type, str):
                raise ValueError(f"invalid error metadata for {judge.model_id}")
            if received_output and error_type is not None:
                raise ValueError(f"output/error status mismatch for {judge.model_id}")
            if item_observation.get("reported_dtype_or_quantization") not in (
                observation.get("reported_dtype_or_quantization") or []
            ):
                raise ValueError(
                    f"reported precision mismatch for {judge.model_id}"
                )

    expected_result = _public_result(corpus, judge_reports)
    if result != expected_result:
        raise ValueError("artifact result does not match recomputed metrics")
    if run.get("verdict_digest_sha256") != _verdict_digest(judge_reports):
        raise ValueError("artifact verdict digest does not match verdict vectors")

    reported = {
        observation["model"]: observation["reported_dtype_or_quantization"]
        for observation in observations
    }
    if run.get("reported_dtype_or_quantization") != reported:
        raise ValueError("reported dtype/quantization summary does not match")
    return result


def promotion_payload(
    artifact: dict[str, Any],
    *,
    corpus_path: Path = CORPUS_PATH,
    cohort: list[judges.SafetyJudge] | None = None,
) -> dict[str, Any]:
    """Validate an artifact and return the deterministic display-cache payload."""
    result = dict(
        _validate_artifact(artifact, corpus_path=corpus_path, cohort=cohort)
    )
    run = artifact["run"]
    result["provenance"] = {
        "artifact_schema_version": artifact["schema_version"],
        "generated_at_utc": run["generated_at_utc"],
        "backend": run["backend"],
        "endpoint_label": run["endpoint_label"],
        "code_sha": run["code_sha"],
        "corpus_sha256": run["corpus_sha256"],
        "model_revisions": run["model_revisions"],
        "generation_settings": run["generation_settings"],
        "reported_dtype_or_quantization": run[
            "reported_dtype_or_quantization"
        ],
        "elapsed_s": run["elapsed_s"],
        "verdict_digest_sha256": run["verdict_digest_sha256"],
    }
    return result


def promote_artifact(
    artifact_path: Path,
    *,
    results_path: Path = RESULTS_PATH,
    corpus_path: Path = CORPUS_PATH,
) -> Path:
    """Explicitly validate and atomically promote one immutable run artifact."""
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload = promotion_payload(artifact, corpus_path=corpus_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = results_path.with_name(f".{results_path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, results_path)
    return results_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or explicitly promote auditable judge run artifacts."
    )
    parser.add_argument(
        "--promote",
        type=Path,
        metavar="RUN_ARTIFACT",
        help="validate and promote an existing run; does not call the backend",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.promote is not None:
        promoted = promote_artifact(args.promote.resolve())
        print(f"Promoted {args.promote} -> {promoted.relative_to(_ROOT)}")
        return 0

    endpoint = _judge_endpoint()
    headers: dict[str, str] = {}
    token = os.environ.get("MODAL_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    corpus = judges.load_judge_corpus(str(CORPUS_PATH))
    if not corpus:
        raise SystemExit(f"empty corpus at {CORPUS_PATH}")
    print(f"Corpus: {len(corpus)} items")
    print(f"Backend: {BACKEND_LABEL}")

    artifact = build_run_artifact(
        corpus=corpus,
        corpus_sha256=_corpus_sha256(),
        endpoint=endpoint,
        headers=headers,
        generated_at=_utc_now(),
        code_sha=_code_sha(),
    )
    path = write_run_artifact(artifact)
    agreement = artifact["result"]["agreement"]
    print(
        f"\nkappa = {agreement['kappa']:.4f} ({agreement['method']}) "
        f"-> band {agreement['band']}"
    )
    print(f"Wrote immutable run artifact: {path.relative_to(_ROOT)}")
    print("Display cache unchanged; use --promote with this artifact explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
