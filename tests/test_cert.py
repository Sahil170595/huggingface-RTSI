"""Tests for cert_signer — Stage 2 Ed25519 signed safety certificate.

All tests are offline (no network, no env-var dependency).
"""

from __future__ import annotations

import copy
import pytest

from cert_signer import (
    SigningKey,
    build_and_sign_cert,
    cert_hash,
    sign_cert,
    verify_cert,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ISSUED_AT = "2026-06-08T00:00:00Z"

_CONFIG = {"model": "llama3-8b", "quant": "q4_K_M"}

_SCREEN_RESULTS = {
    "refusal_stability": {"score": 0.12, "band": "LOW"},
    "judge_agreement": {"kappa": 0.81, "band": "RELIABLE"},
}
_ARTIFACT = {
    "scope": "publisher-linked-huggingface-revision",
    "repo_id": "example/model",
    "revision": "a" * 40,
}
_EVIDENCE = {
    "files": {"substrate/rtsi_table.csv": {"sha256": "b" * 64}},
}


def _make_cert(key: SigningKey | None = None) -> dict:
    k = key or SigningKey.generate()
    return build_and_sign_cert(
        config=_CONFIG,
        screen_results=_SCREEN_RESULTS,
        verdict="SCREEN_PASS",
        issued_at=_ISSUED_AT,
        key=k,
        artifact=_ARTIFACT,
        evidence=_EVIDENCE,
    )


# ---------------------------------------------------------------------------
# 1. Sign → verify roundtrip
# ---------------------------------------------------------------------------


def test_sign_verify_roundtrip():
    key = SigningKey.generate()
    signed = _make_cert(key)
    assert verify_cert(signed) is True


def test_pubkey_hex_is_64_chars():
    key = SigningKey.generate()
    assert len(key.pubkey_hex) == 64


def test_signature_hex_is_128_chars():
    key = SigningKey.generate()
    signed = _make_cert(key)
    assert len(signed["signature_hex"]) == 128


# ---------------------------------------------------------------------------
# 2. TAMPER: modified fields fail verification
# ---------------------------------------------------------------------------


def test_tamper_score_fails():
    signed = _make_cert()
    tampered = copy.deepcopy(signed)
    tampered["screen_results"]["refusal_stability"]["score"] = 0.99
    assert verify_cert(tampered) is False


def test_tamper_verdict_fails():
    signed = _make_cert()
    tampered = copy.deepcopy(signed)
    tampered["verdict"] = "ROUTE"
    assert verify_cert(tampered) is False


def test_tamper_kappa_fails():
    signed = _make_cert()
    tampered = copy.deepcopy(signed)
    tampered["screen_results"]["judge_agreement"]["kappa"] = 0.01
    assert verify_cert(tampered) is False


# ---------------------------------------------------------------------------
# 3. Wrong expected_pubkey_hex → False
# ---------------------------------------------------------------------------


def test_wrong_expected_pubkey():
    key1 = SigningKey.generate()
    key2 = SigningKey.generate()
    signed = sign_cert(
        {
            "cert_id": "abc",
            "version": "1",
            "issued_at": _ISSUED_AT,
            "config": _CONFIG,
            "screen_results": _SCREEN_RESULTS,
            "debate_result": None,
            "verdict": "PASS",
            "prev_cert_hash": None,
        },
        key1,
    )
    # Correct key — passes.
    assert verify_cert(signed, expected_pubkey_hex=key1.pubkey_hex) is True
    # Wrong key — fails.
    assert verify_cert(signed, expected_pubkey_hex=key2.pubkey_hex) is False


# ---------------------------------------------------------------------------
# 4. Unsigned cert → False, no raise
# ---------------------------------------------------------------------------


def test_unsigned_cert_returns_false():
    unsigned = {
        "cert_id": "xyz",
        "version": "1",
        "issued_at": _ISSUED_AT,
        "config": _CONFIG,
        "screen_results": _SCREEN_RESULTS,
        "debate_result": None,
        "verdict": "PASS",
        "prev_cert_hash": None,
    }
    result = verify_cert(unsigned)
    assert result is False


def test_cert_missing_signature_hex_returns_false():
    key = SigningKey.generate()
    signed = _make_cert(key)
    no_sig = {k: v for k, v in signed.items() if k != "signature_hex"}
    assert verify_cert(no_sig) is False


# ---------------------------------------------------------------------------
# 5. from_hex round-trips a generated key
# ---------------------------------------------------------------------------


def test_from_hex_roundtrip():
    key = SigningKey.generate()
    restored = SigningKey.from_hex(key.privkey_hex)
    assert restored.pubkey_hex == key.pubkey_hex


def test_from_hex_produces_valid_signatures():
    key = SigningKey.generate()
    restored = SigningKey.from_hex(key.privkey_hex)
    signed = _make_cert(restored)
    assert verify_cert(signed, expected_pubkey_hex=key.pubkey_hex) is True


# ---------------------------------------------------------------------------
# 6. cert_hash — stable + deterministic; chaining
# ---------------------------------------------------------------------------


def test_cert_hash_is_stable():
    signed = _make_cert()
    h1 = cert_hash(signed)
    h2 = cert_hash(signed)
    assert h1 == h2


def test_cert_hash_is_64_hex_chars():
    signed = _make_cert()
    h = cert_hash(signed)
    assert len(h) == 64
    int(h, 16)  # must be valid hex


def test_cert_hash_changes_on_mutation():
    signed = _make_cert()
    copy1 = copy.deepcopy(signed)
    copy1["verdict"] = "ROUTE"
    assert cert_hash(signed) != cert_hash(copy1)


def test_cert_chaining():
    key = SigningKey.generate()
    first = _make_cert(key)
    first_hash = cert_hash(first)

    second = build_and_sign_cert(
        config=_CONFIG,
        screen_results=_SCREEN_RESULTS,
        verdict="PASS",
        issued_at=_ISSUED_AT,
        key=key,
        prev_cert_hash=first_hash,
    )

    assert second["prev_cert_hash"] == first_hash
    assert verify_cert(second) is True
    assert second["prev_cert_hash"] == cert_hash(first)


# ---------------------------------------------------------------------------
# 7. build_and_sign_cert — schema completeness + valid signature
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {
    "cert_id",
    "version",
    "issued_at",
    "config",
    "artifact",
    "evidence",
    "screen_results",
    "debate_result",
    "verdict",
    "prev_cert_hash",
    "pubkey_hex",
    "signature_hex",
}


def test_build_and_sign_cert_has_all_schema_fields():
    signed = _make_cert()
    assert _REQUIRED_FIELDS <= set(signed.keys())


def test_build_and_sign_cert_version_is_2():
    signed = _make_cert()
    assert signed["version"] == "2"


def test_build_and_sign_cert_cert_id_is_hex():
    signed = _make_cert()
    int(signed["cert_id"], 16)  # must be valid hex


def test_build_and_sign_cert_debate_result_defaults_none():
    signed = _make_cert()
    assert signed["debate_result"] is None


def test_build_and_sign_cert_signature_verifies():
    key = SigningKey.generate()
    signed = _make_cert(key)
    assert verify_cert(signed, expected_pubkey_hex=key.pubkey_hex) is True


def test_build_and_sign_cert_genesis_prev_cert_hash_is_none():
    signed = _make_cert()
    assert signed["prev_cert_hash"] is None


def test_build_and_sign_cert_config_preserved():
    signed = _make_cert()
    assert signed["config"] == _CONFIG


def test_build_and_sign_cert_artifact_and_evidence_preserved():
    signed = _make_cert()
    assert signed["artifact"] == _ARTIFACT
    assert signed["evidence"] == _EVIDENCE


def test_tamper_artifact_revision_fails():
    signed = _make_cert()
    tampered = copy.deepcopy(signed)
    tampered["artifact"]["revision"] = "c" * 40
    assert verify_cert(tampered) is False


def test_tamper_evidence_digest_fails():
    signed = _make_cert()
    tampered = copy.deepcopy(signed)
    tampered["evidence"]["files"]["substrate/rtsi_table.csv"]["sha256"] = "d" * 64
    assert verify_cert(tampered) is False


def test_build_and_sign_cert_screen_results_preserved():
    signed = _make_cert()
    assert signed["screen_results"] == _SCREEN_RESULTS


def test_build_and_sign_cert_verdict_preserved():
    signed = _make_cert()
    assert signed["verdict"] == "SCREEN_PASS"


# ---------------------------------------------------------------------------
# 8. Non-finite scores (NaN / ±Inf) rejected loudly at issuance
# ---------------------------------------------------------------------------


def _screen_results_with_score(score: float) -> dict:
    results = copy.deepcopy(_SCREEN_RESULTS)
    results["refusal_stability"]["score"] = score
    return results


def _issue_with_score(score: float) -> dict:
    return build_and_sign_cert(
        config=_CONFIG,
        screen_results=_screen_results_with_score(score),
        verdict="PASS",
        issued_at=_ISSUED_AT,
        key=SigningKey.generate(),
    )


def test_nan_score_raises_value_error_at_issuance():
    with pytest.raises(ValueError, match="NaN/Infinity"):
        _issue_with_score(float("nan"))


def test_inf_score_raises_value_error_at_issuance():
    with pytest.raises(ValueError, match="NaN/Infinity"):
        _issue_with_score(float("inf"))


def test_negative_inf_score_raises_value_error_at_issuance():
    with pytest.raises(ValueError, match="NaN/Infinity"):
        _issue_with_score(float("-inf"))


def test_non_finite_error_names_the_offending_field():
    with pytest.raises(
        ValueError, match=r"screen_results\.refusal_stability\.score"
    ):
        _issue_with_score(float("nan"))


def test_nan_kappa_raises_value_error_at_issuance():
    results = copy.deepcopy(_SCREEN_RESULTS)
    results["judge_agreement"]["kappa"] = float("nan")
    with pytest.raises(ValueError, match=r"judge_agreement\.kappa"):
        build_and_sign_cert(
            config=_CONFIG,
            screen_results=results,
            verdict="PASS",
            issued_at=_ISSUED_AT,
            key=SigningKey.generate(),
        )


def test_sign_cert_rejects_nan_directly():
    key = SigningKey.generate()
    with pytest.raises(ValueError, match="NaN/Infinity"):
        sign_cert(
            {
                "cert_id": "abc",
                "version": "1",
                "issued_at": _ISSUED_AT,
                "config": _CONFIG,
                "screen_results": _screen_results_with_score(float("nan")),
                "debate_result": None,
                "verdict": "PASS",
                "prev_cert_hash": None,
            },
            key,
        )


def test_verify_cert_with_nan_returns_false_never_raises():
    # A hand-forged cert smuggling NaN past issuance must not crash verify —
    # verify_cert never raises; allow_nan=False surfaces as a caught failure.
    key = SigningKey.generate()
    signed = _make_cert(key)
    forged = copy.deepcopy(signed)
    forged["screen_results"]["refusal_stability"]["score"] = float("nan")
    assert verify_cert(forged) is False


def test_cert_hash_rejects_non_finite():
    signed = _make_cert()
    mutated = copy.deepcopy(signed)
    mutated["screen_results"]["judge_agreement"]["kappa"] = float("inf")
    with pytest.raises(ValueError):
        cert_hash(mutated)


def test_finite_scores_still_sign_and_verify():
    # Regression guard: ordinary finite floats are unaffected by the
    # allow_nan=False tightening.
    signed = _issue_with_score(0.7864)
    assert verify_cert(signed) is True
