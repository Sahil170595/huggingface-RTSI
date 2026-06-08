"""Tests for cert_signer — Stage 2 Ed25519 signed safety certificate.

All tests are offline (no network, no env-var dependency).
"""

from __future__ import annotations

import copy
import json

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


def _make_cert(key: SigningKey | None = None) -> dict:
    k = key or SigningKey.generate()
    return build_and_sign_cert(
        config=_CONFIG,
        screen_results=_SCREEN_RESULTS,
        verdict="PASS",
        issued_at=_ISSUED_AT,
        key=k,
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


def test_build_and_sign_cert_version_is_1():
    signed = _make_cert()
    assert signed["version"] == "1"


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


def test_build_and_sign_cert_screen_results_preserved():
    signed = _make_cert()
    assert signed["screen_results"] == _SCREEN_RESULTS


def test_build_and_sign_cert_verdict_preserved():
    signed = _make_cert()
    assert signed["verdict"] == "PASS"
