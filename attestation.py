"""Immutable artifact and evidence identities for QuantSafe certificates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


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
        "scope": "huggingface-model-revision",
        "repo_id": repo_id,
        "revision": revision,
        "url": f"https://huggingface.co/{repo_id}/tree/{revision}",
    }


def evidence_identity(root: Path) -> dict[str, Any]:
    """Hash every frozen input that determines the signed decision."""
    files = {
        relative: {"sha256": sha256_file(root / relative)}
        for relative in EVIDENCE_PATHS
    }
    return {
        "files": files,
        "method": {
            "name": "Refusal Template Stability Index",
            "paper": "https://arxiv.org/abs/2606.10154",
        },
        "source": (
            "https://huggingface.co/spaces/"
            "build-small-hackathon/quantsafe-certifier/tree/main"
        ),
    }


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
    return mismatches


__all__ = [
    "EVIDENCE_PATHS",
    "PUBLISHED_QUANT_ARTIFACTS",
    "artifact_identity",
    "evidence_identity",
    "sha256_file",
    "verify_evidence_files",
]
