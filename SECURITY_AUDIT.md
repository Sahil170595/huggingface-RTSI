# QuantSafe Adversarial Audit

Audit date: June 14, 2026.

## Scope

The pass covered the Gradio application, certificate and attestation logic,
Modal API, model-revision controls, public claims and links, deployment
packaging, dependency advisories, secret exposure, desktop/mobile rendering,
and the live Hugging Face Space.

## Fixed

- Upgraded to Gradio 6.18.0, Transformers 5.12.0, Hugging Face Hub 1.19.0,
  Starlette 1.3.1, and Pillow 12.2.0 to remove reported dependency advisories.
- Kept ZeroGPU and Modal model downloads on immutable Hugging Face revisions.
- Restricted public live inference to pinned models, known backends, and the
  fixed de-identified debate scenario.
- Hid both expensive GPU listeners and the page-load helper from Gradio's
  public API schema.
- Added HTML escaping and non-reflective validation on forged model, quant,
  certificate, and backend values.
- Added authenticated Modal input-size and token-budget bounds before GPU work.
- Pinned every package in the Modal image.
- Corrected stale wording that could be read as waiving direct safety testing.
- Added full-repository linting and a public GitHub Actions verification gate.

## Verification

- The full suite (**477 tests**) passes under Gradio 6.18.0 and Transformers 5.12.0. A smoke-runtime CI job installs the full pinned requirements.txt (CPU torch) and imports the entire runtime stack, so a transformers/torch API break at the pinned versions fails CI.
- `ruff check .` and `git diff --check` pass.
- Bandit reports no medium- or high-severity findings.
- Every public model revision and documentation link resolves.
- The six tabs switch and all production workflows complete on desktop.
- The mobile overflow menu reaches every tab without horizontal page overflow.
- Certificate issue, pinned verification, tamper detection, and foreign-key
  rejection pass in a real browser.
- The public Space is running and the signing key remains pinned to the
  README-published issuer.

## Residual Risk

`pip-audit` reports `CVE-2025-3000` against the current PyTorch 2.11.0 release.
The advisory concerns a crash in malformed `torch.jit` scripting. QuantSafe
does not call `torch.jit`, accept executable code, or expose a code-execution
surface. There is no newer patched PyTorch release available in the selected
ZeroGPU stack, so CI records this explicit, scoped exception.

Operational secrets remain outside the repository and Space upload:
`GRADIO_CERT_SIGNING_KEY_HEX`, `MODAL_TOKEN`, `MODAL_ENDPOINT`,
`OPENBMB_API_KEY`, and `HF_TOKEN`.

The OpenBMB sponsor endpoint is published as HTTP-only and uses a shared
hackathon token rather than a personal account credential. The transport
limitation is documented in the benchmark artifact; no claim of TLS
confidentiality is made.
