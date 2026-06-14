#!/usr/bin/env python3
"""Verify a QuantSafe certificate signature and optional local evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import attestation  # noqa: E402
import cert_signer  # noqa: E402


DEFAULT_ISSUER = "9a074a15598fef26f5fbd33e8d604cb6c2372989f164331c11018a83fcd98519"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a QuantSafe Ed25519 certificate."
    )
    parser.add_argument("certificate", type=Path, help="Path to certificate JSON")
    parser.add_argument(
        "--issuer",
        default=DEFAULT_ISSUER,
        help="Expected Ed25519 public key hex (defaults to the published issuer)",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        help="Also verify signed evidence-file hashes under this repository root",
    )
    args = parser.parse_args()

    try:
        certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"INVALID: cannot read certificate: {exc}", file=sys.stderr)
        return 1

    if not cert_signer.verify_cert(
        certificate, expected_pubkey_hex=args.issuer.lower()
    ):
        print("INVALID: signature or issuer check failed", file=sys.stderr)
        return 1

    artifact = certificate.get("artifact") or {}
    print(f"VALID signature: issuer={args.issuer.lower()}")
    print(f"artifact scope: {artifact.get('scope', 'unspecified')}")
    if artifact.get("repo_id") and artifact.get("revision"):
        print(f"artifact: {artifact['repo_id']}@{artifact['revision']}")

    if args.evidence_root is not None:
        mismatches = attestation.verify_evidence_files(
            certificate.get("evidence") or {},
            args.evidence_root.resolve(),
        )
        if mismatches:
            for mismatch in mismatches:
                print(f"INVALID: {mismatch}", file=sys.stderr)
            return 1
        print("VALID evidence: all signed local file hashes match")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
