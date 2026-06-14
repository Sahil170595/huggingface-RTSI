"""Immutable artifact and evidence identities for QuantSafe certificates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ACTION_FROM_BAND = {
    "LOW": "SCREEN_PASS",
    "MODERATE": "REVIEW",
    "HIGH": "ROUTE",
}

PUBLISHED_QUANT_ARTIFACTS: dict[tuple[str, str], tuple[str, str]] = {
    ("llama3.2-1b", "AWQ"): (
        "Crusadersk/llama3.2-1b-awq-4bit",
        "c2129999243ed403ad4d64ca2cefe6a0aa50bd17",
    ),
    ("llama3.2-1b", "GPTQ"): (
        "Crusadersk/llama3.2-1b-gptq-4bit",
        "24100f72b80283717083f67d72b07ff24a7a9aa0",
    ),
    ("llama3.2-3b", "AWQ"): (
        "Crusadersk/llama3.2-3b-awq-4bit",
        "753dce6b9831a46054c9c5710ea33d533dca50da",
    ),
    ("llama3.2-3b", "GPTQ"): (
        "Crusadersk/llama3.2-3b-gptq-4bit",
        "716a42c9976158c05e46ba1da283f93dbec3aeac",
    ),
    ("mistral-7b", "AWQ"): (
        "Crusadersk/mistral-7b-awq-4bit",
        "3e6529df3aa5f1defa6654cbb2b48b004e9a6b53",
    ),
    ("mistral-7b", "GPTQ"): (
        "Crusadersk/mistral-7b-gptq-4bit",
        "9cd1b969656738f20c0a37022cf5d7b8abb2517f",
    ),
    ("phi-2", "GPTQ"): (
        "Crusadersk/phi-2-gptq-4bit",
        "6385e88d733fe95b67dc6d18f264b83c6462e681",
    ),
    ("qwen2.5-1.5b", "AWQ"): (
        "Crusadersk/qwen2.5-1.5b-awq-4bit",
        "57f8978065b05507e8e4fd98d6a4bbe5ab392900",
    ),
    ("qwen2.5-1.5b", "GPTQ"): (
        "Crusadersk/qwen2.5-1.5b-gptq-4bit",
        "4e1c7d4d78a3fbb82742207baa7ac305bd836cb5",
    ),
    ("qwen2.5-7b", "AWQ"): (
        "Crusadersk/qwen2.5-7b-awq-4bit",
        "2a36e85d77aaf041e4098a445f3849eeac6a7499",
    ),
    ("qwen2.5-7b", "GPTQ"): (
        "Crusadersk/qwen2.5-7b-gptq-4bit",
        "c0c5e827fdd59cfe2a8278edae2925ef8a6e9260",
    ),
}

EVIDENCE_PATHS = (
    "substrate/rtsi_table.csv",
    "substrate/judge_results.json",
    "substrate/validation_report.json",
    "rtsi_core.py",
    "attestation.py",
    "cert_signer.py",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(model: str, quant: str) -> dict[str, Any]:
    """Return the strongest artifact identity available for a measured cell."""
    published = PUBLISHED_QUANT_ARTIFACTS.get((model, quant))
    if published is None:
        return {
            "scope": "legacy-config-only",
            "repo_id": None,
            "revision": None,
            "note": (
                "The legacy GGUF matrix did not retain immutable weight digests; "
                "this certificate binds the config label and frozen evidence only."
            ),
        }

    repo_id, revision = published
    return {
        "scope": "publisher-linked-huggingface-revision",
        "repo_id": repo_id,
        "revision": revision,
        "url": f"https://huggingface.co/{repo_id}/tree/{revision}",
        "provenance_note": (
            "The publisher links this release target to the measured study cell. "
            "The historical study did not retain a cryptographic weight digest, "
            "so this is not proof that the revision generated the measurement."
        ),
    }


def evidence_identity(root: Path) -> dict[str, Any]:
    """Hash every frozen input that determines the signed decision."""
    files = {
        relative: {
            "sha256": sha256_file(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in EVIDENCE_PATHS
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "files": files,
        "manifest_sha256": manifest_sha256,
        "method": {
            "name": "Refusal Template Stability Index",
            "paper": "https://arxiv.org/abs/2606.10154",
        },
        "source_repository": (
            "https://huggingface.co/spaces/"
            "build-small-hackathon/quantsafe-certifier"
        ),
    }


def validate_record_semantics(record: dict[str, Any]) -> list[str]:
    """Validate the v2 schema and cross-field release-gate invariants."""
    errors: list[str] = []
    if record.get("version") != "2":
        errors.append("record version must be 2")

    config = record.get("config")
    if not isinstance(config, dict):
        return errors + ["record config must be an object"]
    model = config.get("model")
    quant = config.get("quant")
    if not isinstance(model, str) or not isinstance(quant, str):
        errors.append("config model and quant must be strings")
    elif record.get("artifact") != artifact_identity(model, quant):
        errors.append("artifact reference does not match the published mapping")

    screen_results = record.get("screen_results")
    refusal = (
        screen_results.get("refusal_stability")
        if isinstance(screen_results, dict)
        else None
    )
    if not isinstance(refusal, dict):
        errors.append("screen_results.refusal_stability must be an object")
    else:
        band = refusal.get("band")
        score = refusal.get("score")
        if band not in ACTION_FROM_BAND:
            errors.append("refusal band must be LOW, MODERATE, or HIGH")
        elif record.get("verdict") != ACTION_FROM_BAND[band]:
            errors.append("release-gate action is inconsistent with refusal band")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            errors.append("refusal score must be finite and between 0 and 1")

    evidence = record.get("evidence")
    files = evidence.get("files") if isinstance(evidence, dict) else None
    if not isinstance(files, dict):
        errors.append("evidence.files must be an object")
    else:
        if set(files) != set(EVIDENCE_PATHS):
            errors.append("evidence file set does not match the v2 policy")
        for relative, file_record in files.items():
            if not isinstance(file_record, dict):
                errors.append(f"evidence entry is malformed: {relative}")
                continue
            digest = file_record.get("sha256")
            size = file_record.get("size_bytes")
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                errors.append(f"invalid evidence digest: {relative}")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                errors.append(f"invalid evidence size: {relative}")
        expected_manifest = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if evidence.get("manifest_sha256") != expected_manifest:
            errors.append("evidence manifest digest is inconsistent")

    return errors


def verify_evidence_files(evidence: dict[str, Any], root: Path) -> list[str]:
    """Return mismatch descriptions for locally available evidence files."""
    mismatches: list[str] = []
    files = evidence.get("files")
    if not isinstance(files, dict):
        return ["certificate has no evidence.files mapping"]

    for relative, record in files.items():
        if not isinstance(relative, str) or not isinstance(record, dict):
            mismatches.append(f"malformed evidence entry: {relative!r}")
            continue
        expected = record.get("sha256")
        path = root / relative
        if not path.is_file():
            mismatches.append(f"missing evidence file: {relative}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            mismatches.append(
                f"evidence digest mismatch for {relative}: {actual} != {expected}"
            )
        expected_size = record.get("size_bytes")
        if path.stat().st_size != expected_size:
            mismatches.append(
                f"evidence size mismatch for {relative}: "
                f"{path.stat().st_size} != {expected_size}"
            )
    return mismatches


__all__ = [
    "ACTION_FROM_BAND",
    "EVIDENCE_PATHS",
    "PUBLISHED_QUANT_ARTIFACTS",
    "artifact_identity",
    "evidence_identity",
    "sha256_file",
    "validate_record_semantics",
    "verify_evidence_files",
]
