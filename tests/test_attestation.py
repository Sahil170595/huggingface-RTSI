"""Artifact and evidence identity tests."""

from __future__ import annotations

from pathlib import Path

import attestation


ROOT = Path(__file__).resolve().parent.parent


def test_published_gptq_cell_has_immutable_revision():
    identity = attestation.artifact_identity("phi-2", "GPTQ")
    assert identity["scope"] == "huggingface-model-revision"
    assert identity["repo_id"] == "Crusadersk/phi-2-gptq-4bit"
    assert len(identity["revision"]) == 40


def test_legacy_gguf_cell_is_not_overclaimed():
    identity = attestation.artifact_identity("phi-2", "Q4_K_M")
    assert identity["scope"] == "legacy-config-only"
    assert identity["revision"] is None


def test_evidence_manifest_verifies_against_checkout():
    evidence = attestation.evidence_identity(ROOT)
    assert not attestation.verify_evidence_files(evidence, ROOT)


def test_evidence_manifest_detects_a_changed_file(tmp_path: Path):
    evidence = attestation.evidence_identity(ROOT)
    for relative in attestation.EVIDENCE_PATHS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    (tmp_path / "rtsi_core.py").write_text("changed\n", encoding="utf-8")
    mismatches = attestation.verify_evidence_files(evidence, tmp_path)
    assert any("rtsi_core.py" in mismatch for mismatch in mismatches)
