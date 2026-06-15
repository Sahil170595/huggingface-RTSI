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

## External-screen endpoint (`/screen_external_manifest`)

The "Test your own quant" feature adds one public, named endpoint that accepts a
user-supplied manifest of **aggregate** refusal features. The attack surface was
constrained by construction:

- **No untrusted execution paths reachable.** The handler does pure arithmetic
  over validated numbers plus the frozen 45-row substrate. It never loads or
  downloads a model, never fetches a URL, never accepts a raw prompt or
  completion, and never logs supplied content. Tests poison `socket.connect`,
  `urllib.request.urlopen`, and `AutoModelForCausalLM.from_pretrained` and assert
  a clean run, proving no egress or model load on the scoring path.
- **Strict input validation before any work.** Input is capped at 32 KB
  (measured on the wire bytes), parsed with `parse_constant` so `NaN`/`Infinity`
  JSON literals are rejected, then schema-checked: exact `schema_version`,
  64-hex probe digest, 40-hex revisions, unit-interval shares/rates/entropy,
  `n_refusals` integer in `[0, count]`, `mean_tokens_refusal >= 0`, and a strict
  no-unknown-fields / no-missing-fields policy. Every rejection returns a
  structured error with **no scoring** and never raises to the client.
- **No reflection of attacker-controlled metadata.** `repo_id` / `revision` /
  `quantization` are validated but are not echoed into the response, and the
  Gradio output is `gr.JSON` (structured data, not HTML), so a `<script>` repo
  id cannot reflect as markup. A regression test asserts an injected
  `<script>` string never appears in the serialized response.
- **Provisional and unsigned by contract.** The response is fixed to
  `scope: user-supplied-aggregate-evidence` with `signed: false`; the
  certificate-signing path is never invoked. The feature cannot mint a signed
  record from unverified user evidence.
- **No change to existing trust surfaces.** The frozen substrate, score and
  certificate semantics, provider integrations, the six tabs, and the
  heavy-worker concurrency limits are untouched; the two GPU listeners and the
  page-load helper remain private (`api_visibility: private`).

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
