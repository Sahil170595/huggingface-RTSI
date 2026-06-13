"""Stage 2 — Ed25519 signed safety certificate.

Cryptographically attests the result of the two safety screens
(refusal-drift + judge-agreement) for a (model, quant) config.

Design mirrors muse/contracts/signing.py (P107.7 precedent):
- Ed25519 via `cryptography` hazmat (NOT pynacl).
- Signed payload = canonical JSON (sorted keys, no whitespace) of the full
  cert dict *excluding* pubkey_hex and signature_hex.
- Key loading: GRADIO_CERT_SIGNING_KEY_HEX env var, or ephemeral generate().
- verify_cert never raises — returns False on any failure.
- cert_hash is stable and deterministic (sorted keys, no whitespace of the
  full signed cert including pubkey_hex + signature_hex); used for chaining.
- Non-finite floats (NaN / ±Infinity) are rejected at issuance with a clear
  ValueError, and all canonical JSON uses allow_nan=False — NaN/Infinity are
  not valid JSON, and a cert carrying them would fail portable verification
  on any strict parser.

Only dependency beyond stdlib: cryptography.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import uuid
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

logger = logging.getLogger(__name__)

ENV_SIGNING_KEY_HEX = "GRADIO_CERT_SIGNING_KEY_HEX"

# Fields excluded from the signed payload (they are the signature itself).
_EXCLUDED_FROM_PAYLOAD = frozenset({"pubkey_hex", "signature_hex"})


# ---------------------------------------------------------------------------
# SigningKey
# ---------------------------------------------------------------------------


class SigningKey:
    """Ed25519 keypair wrapper.

    Typical usage::

        key = SigningKey.from_env_or_generate()
        signed = sign_cert(cert, key)
        assert verify_cert(signed)
    """

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self._private = private_key
        self._public: Ed25519PublicKey = private_key.public_key()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def generate(cls) -> "SigningKey":
        """Generate a fresh in-memory Ed25519 keypair."""
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_hex(cls, hex_key: str) -> "SigningKey":
        """Build from a 64-hex-char (32-byte) private key."""
        raw = bytes.fromhex(hex_key.strip())
        if len(raw) != 32:
            raise ValueError(f"signing key must be 32 bytes, got {len(raw)}")
        return cls(Ed25519PrivateKey.from_private_bytes(raw))

    @classmethod
    def from_env_or_generate(cls) -> "SigningKey":
        """Load from GRADIO_CERT_SIGNING_KEY_HEX or generate ephemeral key.

        When generating, prints the pubkey_hex so operators can pin it.
        """
        hex_key = os.environ.get(ENV_SIGNING_KEY_HEX, "").strip()
        if hex_key:
            try:
                loaded = cls.from_hex(hex_key)
                logger.info(
                    "cert_signer: using key from %s (pubkey=%s)",
                    ENV_SIGNING_KEY_HEX,
                    loaded.pubkey_hex,
                )
                return loaded
            except Exception:
                logger.warning(
                    "Invalid %s — generating ephemeral key",
                    ENV_SIGNING_KEY_HEX,
                    exc_info=True,
                )
        generated = cls.generate()
        print(
            f"cert_signer: ephemeral key generated "
            f"(pubkey={generated.pubkey_hex}) — "
            f"set {ENV_SIGNING_KEY_HEX} to pin"
        )
        return generated

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pubkey_hex(self) -> str:
        """32-byte raw public key as 64 hex chars."""
        return self._public.public_bytes_raw().hex()

    @property
    def privkey_hex(self) -> str:
        """32-byte raw private key as 64 hex chars. Use sparingly."""
        return self._private.private_bytes_raw().hex()

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    def sign(self, payload: bytes) -> bytes:
        """Return the 64-byte Ed25519 signature over payload."""
        return self._private.sign(payload)


# ---------------------------------------------------------------------------
# Payload + signing helpers
# ---------------------------------------------------------------------------


def _validate_finite(value: Any, path: str = "cert") -> None:
    """Raise ValueError if any float anywhere in value is NaN or ±Infinity.

    Walked recursively over dicts / lists / tuples.  Called at issuance
    (sign_cert) so a non-finite score fails loudly with a message naming the
    offending field — instead of surfacing later as a cryptic json.dumps
    error, or worse, as a verification failure on a consumer's machine.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            f"cannot sign certificate: field {path!r} is {value!r} — "
            f"NaN/Infinity are not valid JSON and would break portable "
            f"verification. Fix the score upstream before issuing."
        )
    if isinstance(value, dict):
        for k, v in value.items():
            _validate_finite(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _validate_finite(v, f"{path}[{i}]")


def _canonical_payload(cert: dict) -> bytes:
    """Return the canonical UTF-8 bytes that get signed.

    Excludes pubkey_hex and signature_hex — they are added by sign_cert and
    must not be part of the payload they attest to.

    allow_nan=False: NaN/Infinity serialize to non-standard JSON tokens
    (``NaN``, ``Infinity``) that strict parsers reject, which would make the
    signed payload non-portable.  json.dumps raises ValueError instead.
    """
    stripped = {k: v for k, v in cert.items() if k not in _EXCLUDED_FROM_PAYLOAD}
    return json.dumps(
        stripped, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sign_cert(cert: dict, key: SigningKey) -> dict:
    """Return {**cert, pubkey_hex, signature_hex}.

    The caller must NOT include pubkey_hex / signature_hex in cert yet;
    this function adds them.  Safe to re-call (overwrites old values).

    Raises
    ------
    ValueError
        If any float in cert is NaN or ±Infinity.  Such values produce
        non-standard JSON that portable verifiers reject, so issuance fails
        loudly here rather than verification failing silently later.
    """
    _validate_finite(cert)
    payload = _canonical_payload(cert)
    sig_bytes = key.sign(payload)
    return {
        **cert,
        "pubkey_hex": key.pubkey_hex,
        "signature_hex": sig_bytes.hex(),
    }


def verify_cert(cert: dict, expected_pubkey_hex: str | None = None) -> bool:
    """Verify the Ed25519 signature embedded in a signed cert dict.

    Parameters
    ----------
    cert:
        A signed cert dict (must contain pubkey_hex and signature_hex).
    expected_pubkey_hex:
        If provided, also require cert['pubkey_hex'] == expected_pubkey_hex.

    Returns
    -------
    True if the signature is present and verifies.  False (never raises) on
    any failure — including unsigned certs, malformed hex, wrong key, tampered
    fields.
    """
    try:
        pubkey_hex = cert.get("pubkey_hex")
        sig_hex = cert.get("signature_hex")
        if not pubkey_hex or not sig_hex:
            return False
        if expected_pubkey_hex is not None and pubkey_hex != expected_pubkey_hex:
            return False
        pubkey_bytes = bytes.fromhex(pubkey_hex)
        sig_bytes = bytes.fromhex(sig_hex)
        public_key = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
        payload = _canonical_payload(cert)
        public_key.verify(sig_bytes, payload)
        return True
    except InvalidSignature:
        return False
    except Exception:
        logger.debug("verify_cert: unexpected failure", exc_info=True)
        return False


def cert_hash(signed_cert: dict) -> str:
    """SHA-256 hex of the full signed cert (including pubkey_hex + signature_hex).

    Used as prev_cert_hash in the next cert to form a chain.  Deterministic
    and stable: sorted keys, no whitespace, allow_nan=False (a properly
    issued cert can never contain NaN/Infinity — see sign_cert — so this
    raising ValueError means the input was never validly signed).
    """
    canonical = json.dumps(
        signed_cert, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------


def build_and_sign_cert(
    *,
    config: dict[str, str],
    screen_results: dict[str, Any],
    verdict: str,
    issued_at: str,
    key: SigningKey,
    debate_result: Any = None,
    prev_cert_hash: str | None = None,
) -> dict:
    """Assemble the full cert schema, then sign it.

    Parameters
    ----------
    config:
        {"model": str, "quant": str}
    screen_results:
        {
          "refusal_stability": {"score": float, "band": "LOW|MODERATE|HIGH"},
          "judge_agreement":   {"kappa": float, "band": "RELIABLE|MIXED|UNRELIABLE"}
        }
    verdict:
        "PASS" | "REVIEW" | "ROUTE"  (LOW->PASS, MODERATE->REVIEW, HIGH->ROUTE)
    issued_at:
        ISO-8601 UTC string — caller supplies; never call time() inside here.
    key:
        SigningKey instance.
    debate_result:
        Reserved for Stage 3; pass None (default).
    prev_cert_hash:
        sha256 hex of the prior signed cert (cert_hash(prev)); None = genesis.

    Returns
    -------
    Fully signed cert dict conforming to the certificate JSON schema.

    Raises
    ------
    ValueError
        If any score/kappa (or any other float) is NaN or ±Infinity — see
        sign_cert.  Issuance is the right place to fail loudly.
    """
    cert: dict[str, Any] = {
        "cert_id": uuid.uuid4().hex,
        "version": "1",
        "issued_at": issued_at,
        "config": config,
        "screen_results": screen_results,
        "debate_result": debate_result,
        "verdict": verdict,
        "prev_cert_hash": prev_cert_hash,
    }
    return sign_cert(cert, key)


__all__ = [
    "SigningKey",
    "sign_cert",
    "verify_cert",
    "cert_hash",
    "build_and_sign_cert",
]
