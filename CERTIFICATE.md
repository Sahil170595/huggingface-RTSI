# QuantSafe Signed Screening Record v2

QuantSafe records are Ed25519-signed release-gate attestations.

For the 11 published AWQ/GPTQ checkpoints in the measured matrix, the signed
payload includes a publisher-linked Hugging Face repository and immutable
40-character revision. The historical study did not retain weight digests, so
this identifies the publisher's release target; it does not prove that those
exact weights generated the historical measurement. For older GGUF cells, the
record says `legacy-config-only`.

Every record also signs SHA-256 hashes for:

- `substrate/rtsi_table.csv`
- `substrate/judge_results.json`
- `substrate/validation_report.json`
- `rtsi_core.py`
- `attestation.py`
- `cert_signer.py`

A valid signature proves who issued the record and that its payload was not
changed. The evidence manifest is content-addressed, and the verifier also
enforces the v2 schema, publisher-linked artifact mapping, finite score range,
and consistency between the refusal band and release-gate action.

## Offline verification

Save the displayed record JSON, then run:

```bash
python scripts/verify_certificate.py record.json
```

To verify the signed evidence against a checkout of this repository:

```bash
python scripts/verify_certificate.py record.json --evidence-root .
```

The verifier pins the published issuer key by default:

```text
9a074a15598fef26f5fbd33e8d604cb6c2372989f164331c11018a83fcd98519
```

The Space's Foreign re-sign test demonstrates why issuer pinning is necessary:
a modified record can be signed with a different key and remain internally
self-consistent, but it still fails verification against the published issuer.
