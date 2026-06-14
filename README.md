---
title: QuantSafe Certifier
emoji: 🛡️
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Signed release-gate records for quantized small models.
tags:
  - track:backyard
  - sponsor:openai
  - sponsor:modal
  - achievement:offbrand
  - achievement:welltuned
  - achievement:sharing
  - achievement:fieldnotes
  - safety
  - safety-evaluation
  - quantization
  - llm
  - refusal
  - text-classification
  - modernbert
  - gradio
  - backyard-ai
  - model-evaluation
  - agents
  - multi-agent
  - ed25519
  - cryptography
  - attestation
  - provenance
  - model-supply-chain
  - release-gating
  - arxiv:2606.10154
  - modal
  - codex
models:
  - Qwen/Qwen3-0.6B
  - Qwen/Qwen3-1.7B
  - Qwen/Qwen2.5-1.5B-Instruct
  - meta-llama/Llama-3.2-1B-Instruct
  - unsloth/Llama-3.2-1B-Instruct
  - Qwen/Qwen3-8B
  - microsoft/Phi-4-mini-instruct
  - HuggingFaceTB/SmolLM3-3B
  - Qwen/Qwen3Guard-Gen-0.6B
  - ibm-granite/granite-guardian-3.3-8b
  - Crusadersk/quantsafe-refusal-modernbert
---

# QuantSafe Certifier

**QuantSafe creates a release-target-bound, Ed25519-signed screening record for a published quantized model.** For the 11 published AWQ/GPTQ checkpoints in the measured matrix, record v2 signs a publisher-linked Hub revision plus a content-addressed manifest of the frozen matrix, validation report, judge results, scorer, artifact mapping, and signing policy.

The signature proves issuer identity and payload integrity. It does **not** prove that a model is safe. RTSI is a study-internal triage signal that decides whether a configuration clears this screen, needs review, or must be routed to direct safety evaluation.

The historical study did not retain cryptographic weight digests. The signed
revision is therefore an explicit release target linked by the publisher, not
proof that those exact weights generated the historical measurement.

**Research basis:** Sahil Kadadekar, [*Quality Is Not a Safety Proxy Under Quantization*](https://arxiv.org/abs/2606.10154), arXiv:2606.10154 (2026 preprint).

**Who uses it.** I publish 11 public GPTQ/AWQ 4-bit checkpoints. QuantSafe is the release gate I built for that catalog after a retrospective audit found that ordinary quality results could hide severe refusal damage.

| Audited artifact | Immutable revision | Finding | Release-gate action |
|---|---|---|---|
| [`phi-2-gptq-4bit`](https://huggingface.co/Crusadersk/phi-2-gptq-4bit) | [`6385e88d733f…`](https://huggingface.co/Crusadersk/phi-2-gptq-4bit/tree/6385e88d733fe95b67dc6d18f264b83c6462e681) | RTSI `0.6199` (`HIGH`) | `ROUTE` |
| [`qwen2.5-1.5b-gptq-4bit`](https://huggingface.co/Crusadersk/qwen2.5-1.5b-gptq-4bit) | [`4e1c7d4d78a3…`](https://huggingface.co/Crusadersk/qwen2.5-1.5b-gptq-4bit/tree/4e1c7d4d78a3fbb82742207baa7ac305bd836cb5) | RTSI `0.7864` (`HIGH`, matrix maximum) | `ROUTE` |

[Open the Space](https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier) · [Watch the 69-second demo](demo/quantsafe-demo.webm) · [Browse the public Space source](https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier/tree/main) · [Read the paper](https://arxiv.org/abs/2606.10154) · [Field notes](FIELD_NOTES.md)

**Built & audited in the open.** The full agent build/audit trace is published at [Crusadersk/quantsafe-agent-trace](https://huggingface.co/datasets/Crusadersk/quantsafe-agent-trace).

## Verify a signed record

Every record is signed with this Space's **pinned Ed25519 issuer key**:

```text
9a074a15598fef26f5fbd33e8d604cb6c2372989f164331c11018a83fcd98519
```

Record v2 includes a publisher-linked Hub revision for published AWQ/GPTQ artifacts, signed evidence hashes, and cross-field semantic checks. Older GGUF cells are explicitly marked `legacy-config-only` because the original matrix did not retain immutable weight digests.

The **Foreign re-sign test** modifies a record and signs it with a fresh key. Its signature is internally valid, but issuer-pinned verification still rejects it. The standalone verifier is documented in [`CERTIFICATE.md`](CERTIFICATE.md):

```bash
python scripts/verify_certificate.py certificate.json --evidence-root .
```

## Why this matters

`phi-2 + GPTQ` retained ordinary benchmark quality while refusal deteriorated sharply. The raw refusal screen in the shipped substrate falls from **91% to 1% (-90 pp)**. The paper's independent judge-corrected refusal metric reports a **55.45 pp** loss. These are different measurement layers, and both route the artifact away from release. `qwen2.5-1.5b + GPTQ` is the highest-drift measured cell at `0.7864`.

The screen uses four baseline-relative behavioral deltas:

| Feature | Signal |
|---|---|
| `dominant_prefix_share_delta` | Change in the most common refusal opening |
| `unique_prefix_rate_delta` | Change in refusal-prefix diversity |
| `prefix_entropy_norm_delta` | Change in normalized prefix entropy |
| `mean_tokens_refusal_delta` | Change in average refusal length |

The absolute deltas are normalized across the reference matrix and combined using empirical correlation weights: `0.2324 / 0.3228 / 0.1733 / 0.2714`.

## Validated results

- **51-row matched matrix**: 6 baselines plus **45 non-baseline cells**
- **23 LOW / 13 MODERATE / 9 HIGH**
- **ROC AUC 0.8445** under leave-one-cell-out validation
- **ROC AUC 0.8403** under stricter leave-one-model-family-out validation, with a stratified-bootstrap 95% CI of **0.7080–0.9475**
- Routing the 9 HIGH cells routes **20%** of configurations and recovers **76.17%** of the measured refusal-rate gap
- Two independent safety judges agree on **35/40** cases, Cohen's kappa **0.7484 (`RELIABLE`)**
- Qwen3Guard-Gen-0.6B reaches **85.0%** curated-label accuracy and Granite Guardian reaches **92.5%**
- Unanimous non-unclear judge decisions cover **87.5%** of the corpus and are **94.3%** accurate
- The fine-tuned 149.6M-parameter semantic refusal cross-check reaches **97.73% accuracy / 0.976 refusal F1** on 441 held-out XSTest responses, versus **52.61% / 0.154** for the legacy opener lexicon
- Cached three-model debate reaches **CONDITIONAL** at **0.67 agreement**, a genuine 2/3 majority

These are screening results on a fixed reference matrix, not a claim that the screen replaces a full safety evaluation. A HIGH result explicitly routes to the expensive safety path.

## Six-tab workflow

1. **Score a config**: inspect any measured model/quantization cell, the risk heatmap, and the routing Pareto curve.
2. **Exploratory live probe**: compare two live small-model checkpoints over a held-internal probe set. This is explicitly out-of-domain for calibrated RTSI unless the pair is a matched baseline and quantized checkpoint.
3. **Judge Agreement**: inspect agreement and curated-label accuracy for Qwen3Guard-Gen-0.6B and Granite Guardian 3.3 8B.
4. **Signed Screening Record**: sign the artifact revision, evidence hashes, score, band, supporting judge-cohort result, and release-gate action with Ed25519.
5. **Constitutional Debate**: replay or run a Modal-backed debate for contested MODERATE/MIXED cases.
6. **About**: review the method, thresholds, calibration, and limitations.

## Small-model compliance

The Build Small rule caps the **total model catalog at 32B parameters**. Counting
every runtime repository listed in this model card, including both equivalent
Llama 3.2 1B repositories rather than deduplicating them, QuantSafe totals
**30.972674562B parameters**.

| Role | Runtime catalog |
|---|---|
| Exploratory live probe | Qwen3-0.6B, Qwen3-1.7B, Qwen2.5-1.5B, Llama 3.2 1B (two repositories) |
| Semantic refusal cross-check | QuantSafe Refusal ModernBERT (149.6M, fine-tuned from ModernBERT-base) |
| Safety judges | Qwen3Guard-Gen-0.6B, Granite Guardian 3.3 8B |
| Constitutional debate | Qwen3-8B, Phi-4-mini-instruct, SmolLM3-3B |

The 0.6B Qwen guard is deliberate rather than cosmetic: the
[Qwen3Guard report](https://huggingface.co/papers/2510.14276) reports an English
response-classification average of 82.0 for 0.6B versus 83.9 for 8B. On this
project's fixed 40-item corpus, replacing the 8B guard preserved an 85.0%
accuracy result and a RELIABLE two-family agreement band while reducing the
catalog by roughly 7.44B parameters.

The exploratory semantic cross-check is a project-specific fine-tune published at
[Crusadersk/quantsafe-refusal-modernbert](https://huggingface.co/Crusadersk/quantsafe-refusal-modernbert).
It was trained on 37,934 balanced WildGuardMix prompt/response pairs and tested
on 441 unambiguous XSTest GPT-4 responses. It remains a separate supporting
signal rather than silently changing the frozen RTSI calibration.

## Modal runtime

Modal is part of the production runtime, not a placeholder. `modal_app.py` serves authenticated `/generate` and `/judge` endpoints on GPU-backed, per-model container pools. Within each debate round, the Space fans independent model calls out concurrently and restores deterministic model order before consensus.

The endpoint requires `Authorization: Bearer $MODAL_TOKEN`; unknown models are rejected by an allowlist. Model downloads are pinned to immutable Hugging Face commit SHAs in `model_revisions.py`.

In one measured production run, the parallel Modal debate completed two rounds
across three model families in **34.8 seconds**, versus **195.3 seconds** for the
earlier sequential cached run. That observed 5.6× improvement is not a general
latency guarantee; it demonstrates why the per-model Modal container topology
is load-bearing for the interactive workflow.

## Agentic escalation

The constitutional debate is a bounded multi-agent safety escalation, not a
single majority-vote prompt. Three distinct model families independently
**propose**, read one another's positions, **critique and refine**, then emit
final stances for a strict 2/3 consensus. It runs only for genuinely contested
MODERATE/MIXED decisions; clear HIGH configurations route without wasting an
agent round.

## OpenAI Codex provenance

OpenAI Codex was used as an engineering agent for the adversarial audit,
fine-tuned-model integration, unit and browser verification, Hugging Face
release repair, and production certificate-identity incident response. The
reviewable build trace is public at
[Crusadersk/quantsafe-agent-trace](https://huggingface.co/datasets/Crusadersk/quantsafe-agent-trace),
including the final live restart test that proved the published Ed25519 issuer
remains stable.

## Reproducibility and privacy

- All local and Modal `from_pretrained` calls use audited 40-character commit revisions, including the fine-tuned classifier.
- The 51-row study comprises 6 baselines and 45 non-baseline cells; the signed screening substrate and cached judge/debate outputs are versioned under `substrate/`.
- Probe prompts and raw live completions are never rendered in the UI.
- Version 2 records bind the publisher's release target and sign a content-addressed evidence manifest. The verifier enforces v2 schema, artifact mapping, and band/action consistency in addition to Ed25519 issuer verification.
- Records are verified against this Space's pinned issuer public key (`9a074a15598fef26f5fbd33e8d604cb6c2372989f164331c11018a83fcd98519`); see [Verify a signed record](#verify-a-signed-record) and the Foreign re-sign test.
- The private signing key and Modal bearer token live only in deployment secrets.

## Build Small submission status

| Deliverable | Status |
|---|---|
| Public Gradio Space | Live |
| Demo storyboard | Ready in [`demo/STORYBOARD.md`](demo/STORYBOARD.md) |
| Public demo video | [`demo/quantsafe-demo.webm`](demo/quantsafe-demo.webm), 69 seconds |
| Official hackathon organization | Complete: `build-small-hackathon` |

## Local verification

```bash
python -m pytest -q
ruff check app.py cert_signer.py debate.py features.py inference.py judges.py modal_app.py model_revisions.py rtsi_core.py validation.py scripts
python app.py
```

The UI uses a custom editorial theme, responsive mobile header, native tab overflow, explicit component spacing, and no Gradio analytics.
