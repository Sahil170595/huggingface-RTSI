---
title: QuantSafe Certifier
emoji: 🛡️
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Signed release-gate records for quantized small models.
tags:
  - track:backyard
  - sponsor:openai
  - sponsor:modal
  - sponsor:nvidia
  - achievement:offbrand
  - achievement:welltuned
  - achievement:sharing
  - achievement:fieldnotes
  - achievement:llama
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
  - llama-cpp
  - gguf
  - modal
  - codex
models:
  - Qwen/Qwen3-0.6B
  - Qwen/Qwen3-1.7B
  - Qwen/Qwen2.5-1.5B-Instruct
  - meta-llama/Llama-3.2-1B-Instruct
  - Qwen/Qwen3-8B
  - microsoft/Phi-4-mini-instruct
  - HuggingFaceTB/SmolLM3-3B
  - Qwen/Qwen3Guard-Gen-0.6B
  - ibm-granite/granite-guardian-3.3-8b
  - nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3
  - Crusadersk/quantsafe-refusal-modernbert
---

# QuantSafe Certifier

**QuantSafe creates a release-target-bound, Ed25519-signed, tamper-evident release-screen record for a published quantized model.** For the 11 published AWQ/GPTQ checkpoints in the measured matrix, record v2 signs a publisher-linked Hub revision plus a content-addressed manifest of the frozen matrix, validation report, judge results, scorer, artifact mapping, and signing policy.

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

[Open the Space](https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier) · [Watch the 36-second judge demo](demo/quantsafe-demo.webm) · [Download the social-ready MP4](demo/quantsafe-demo.mp4) · [Browse the GitHub source](https://github.com/Sahil170595/huggingface-RTSI) · [Browse the Space source](https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier/tree/main) · [Read the paper](https://arxiv.org/abs/2606.10154) · [Field notes](FIELD_NOTES.md) · [Judge benchmark dataset](https://huggingface.co/datasets/Crusadersk/quantsafe-judge-benchmark) · [Adversarial audit](SECURITY_AUDIT.md)

**Built & audited in the open.** The full agent build/audit trace is published at [Crusadersk/quantsafe-agent-trace](https://huggingface.co/datasets/Crusadersk/quantsafe-agent-trace).

## Who this is for

I am the first user. I publish 11 public GPTQ/AWQ 4-bit checkpoints on Hugging
Face. A retrospective audit of that catalog found configurations where ordinary
quality results hid severe refusal damage, including my published
`phi-2-gptq-4bit`. I built QuantSafe to turn that finding into a repeatable
publisher workflow: inspect a measured release target, assign **SCREEN_PASS /
REVIEW / ROUTE**, and retain a signed record of the screen, evidence version,
and release action. It is a triage gate for my quantized-model catalog, not a
claim that a downstream deployment or model is safe.

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

`phi-2 + GPTQ` retained ordinary benchmark quality while refusal deteriorated sharply. The raw refusal screen in the shipped substrate falls from **91% to 1% (-90 pp)**. The paper separately reports a **55.45 pp** judged-refusal loss for the same cell. These are different measurement layers, and both route the artifact away from release. `qwen2.5-1.5b + GPTQ` is the highest-drift measured cell at `0.7864`.

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
- **ROC AUC 0.8403** under stricter leave-one-model-family-out validation — the primary generalization claim — with a stratified-bootstrap 95% CI of **0.7080–0.9475**
- **ROC AUC 0.8445** under leave-one-cell-out validation, numerically identical to the in-sample AUC: at n=45 the per-fold weight refits do not reorder cells across the decision boundary, so leave-one-cell-out cannot show shrinkage here (see the `circularity_note` in `tr163_analysis.json`). We therefore lead with the family-held-out figure.
- Routing the HIGH band recovers **76.17%** of the measured refusal-rate gap in-sample (**20%**, 9/45) and **76.37%** under leave-one-cell-out (**22%**, 10/45)
- Three safety judge models from distinct model families agree unanimously on **34/40** cases, Fleiss' kappa **0.7929 (`RELIABLE`)**; its zone-stratified bootstrap 95% CI is **0.6641–0.9239**, which crosses the 0.70 band threshold
- Qwen3Guard-Gen-0.6B reaches **85.0%** project-label accuracy, Granite Guardian **92.5%**, and NVIDIA Llama-3.1-Nemotron-Safety-Guard-8B-v3 **95.0%**, the highest point estimate on this 40-item project-labeled corpus; the exact paired comparison with Granite is **McNemar p=1.0**
- Unanimous non-unclear judge decisions cover **85%** of the corpus and are **97.1%** accurate against the project-curated labels
- The corpus, all three judges' verdicts, and this comparison are published as an open, citable benchmark: [`Crusadersk/quantsafe-judge-benchmark`](https://huggingface.co/datasets/Crusadersk/quantsafe-judge-benchmark)
- **External-labeled judge benchmark** (PKU-Alignment/BeaverTails 30k_test, N=400, seed 20260615, third-party human crowd labels): Qwen3Guard-Gen-0.6B 84.0% accuracy [80.1–87.3], macro-F1 0.854, coverage 96.8%; Granite-Guardian-3.3-8B 84.75% [80.9–87.9], macro-F1 0.847, coverage 100%; Nemotron-Safety-Guard-8B-v3 81.0% [76.9–84.5], macro-F1 0.808, coverage 100%; three-guard unanimous (selective consensus) 89.76% [86.0–92.6] at 83% coverage. These accuracies are measured against external third-party human labels (BeaverTails), not the project's own 40-item corpus, directly addressing the label-circularity limitation. On this benchmark the 0.6B Qwen3Guard matches the 8B Granite Guardian and exceeds the 8B Nemotron guard, supporting the small-model design.
- The fine-tuned 149.6M-parameter semantic refusal cross-check reaches **97.73% accuracy / 0.976 refusal F1** on 441 held-out XSTest responses, versus **52.61% / 0.154** for the legacy 13-opener lexicon — which is the small-model refusal-shape feature extractor applied out-of-domain to GPT-4 text, so this gap reflects domain mismatch as much as fine-tuning gain
- Cached three-model debate reaches **CONDITIONAL** at **0.67 agreement**, a genuine 2/3 majority

These are screening results on a fixed reference matrix, not a claim that the screen replaces a full safety evaluation. A HIGH result explicitly routes to the expensive safety path.

**Prospective transfer demonstration** (NF4 4-bit, bitsandbytes; frozen 45-cell substrate; 100 AdvBench probes; scored one cell at a time): Falcon3-3B-Instruct (TII) RTSI 0.0018, LOW, refusal_rate_delta +0.02, material_loss False; SmolLM2-1.7B-Instruct (HuggingFaceTB) RTSI 0.2408, MODERATE, refusal_rate_delta −0.10, material_loss True. As a prospective out-of-distribution check (a demonstration, not a powered AUC: n=2 cells), the frozen screen was applied blind to two families absent from the 45-cell matrix and to a quantization method (NF4) it was never calibrated on. It correctly cleared Falcon3-3B (no refusal loss, LOW) and flagged SmolLM2-1.7B (a measured 10-point refusal-rate drop, MODERATE / material-loss). The RTSI screen scores baseline-relative refusal-shape drift and is quantization-method-agnostic by construction.

**MiniCPM compatibility note:** OpenBMB MiniCPM4.1-8B was also evaluated as a fourth (reasoning-model) judge and a prospective subject; its trust_remote_code modeling code (pinned revision) imports is_torch_fx_available, which is removed in the pinned transformers 5.12.0, so it fails to load under this stack. We document this incompatibility rather than downgrade the pinned runtime, and exclude MiniCPM from the live results.

## Llama Champion evidence

QuantSafe's measured substrate includes **34 GGUF cells** across the
`Q2_K`, `Q3_K_S`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, and `Q8_0` ladder. Those model
runs were executed through **llama.cpp via Ollama**, then normalized into the
same matched quality/safety matrix as the AWQ and GPTQ cells. The runtime and
compute split are documented in the paper's
[Compute Resources section](https://arxiv.org/html/2606.10154v1#A7).

The Space serves the frozen aggregate outputs rather than downloading the
historical GGUF weights again. This is evidence of the project's actual
llama.cpp evaluation path, not a claim that the live ZeroGPU probe uses
llama.cpp.

## Six-tab workflow

1. **Score a config**: inspect any measured model/quantization cell, the risk heatmap, and the routing Pareto curve.
2. **Exploratory live probe**: choose a pair from four live small-model checkpoint options and compare them over a held-internal probe set. This is explicitly out-of-domain for calibrated RTSI unless the pair is a matched baseline and quantized checkpoint.
3. **Judge Agreement**: inspect fixed-corpus agreement and project-label accuracy for three judge models from distinct families: Qwen3Guard-Gen-0.6B, Granite Guardian 3.3 8B, and NVIDIA Llama-3.1-Nemotron-Safety-Guard-8B-v3.
4. **Signed Screening Record**: create a tamper-evident release-screen record covering the artifact revision, evidence hashes, score, band, supporting cohort-level benchmark result, and release-gate action.
5. **Constitutional Debate**: replay or run a Modal-backed debate for contested MODERATE/MIXED cases.
6. **About**: review the method, thresholds, calibration, and limitations.

## Small-model compliance

The Build Small rule caps **each individual model at under 32B parameters**.
Every model QuantSafe runs clears that cap comfortably. The largest is
**Qwen3-8B at 8,190,735,360 parameters**.

| Role | Runtime catalog | Largest model |
|---|---|---|
| Exploratory live probe | Four checkpoint options: Qwen3-0.6B, Qwen3-1.7B, Qwen2.5-1.5B-Instruct, and Llama 3.2 1B Instruct; the selected pair is batched under one `@spaces.GPU` allocation | 1.7B |
| Semantic refusal cross-check | QuantSafe Refusal ModernBERT (149.6M, fine-tuned from ModernBERT-base) | 0.150B |
| Safety judges | Qwen3Guard-Gen-0.6B, Granite Guardian 3.3 8B, NVIDIA Llama-3.1-Nemotron-Safety-Guard-8B-v3 | 8.171B |
| Constitutional debate | Qwen3-8B, Phi-4-mini-instruct, SmolLM3-3B | Qwen3-8B: 8,190,735,360 |

The 0.6B Qwen guard is deliberate rather than cosmetic: the
[Qwen3Guard report](https://huggingface.co/papers/2510.14276) reports an English
response-classification average of 82.0 for 0.6B versus 83.9 for 8B. On this
project's fixed 40-item corpus, three judge models from distinct families —
Qwen3Guard-Gen-0.6B, Granite Guardian, and NVIDIA
Llama-3.1-Nemotron-Safety-Guard-8B-v3 — reach a RELIABLE Fleiss' agreement
band. The Nemotron guard's 95.0% accuracy is the highest point estimate on this
project-labeled corpus, not a general ranking of the judge models.

## NVIDIA evidence

NVIDIA's `Llama-3.1-Nemotron-Safety-Guard-8B-v3` is one of the three judge
models in the published 40-item benchmark. Its 95.0% project-label accuracy is
the cohort's highest point estimate on that fixed corpus, but the exact paired
comparison with Granite is not statistically separated (`p=1.0`). The
benchmark cache was generated through the authenticated Modal `/judge` backend
with Nemotron loaded in native **BF16** and is displayed in the Judge Agreement
tab. The Nemotron guard does **not** cross-check every screen, produce a
config-specific verdict, or turn a screening record into proof of model safety.

The exploratory semantic cross-check is a project-specific fine-tune published at
[Crusadersk/quantsafe-refusal-modernbert](https://huggingface.co/Crusadersk/quantsafe-refusal-modernbert).
It was trained on 37,934 balanced WildGuardMix prompt/response pairs and tested
on 441 unambiguous XSTest GPT-4 responses. It remains a separate supporting
signal rather than silently changing the frozen RTSI calibration.

## Modal runtime

Modal is part of the production runtime, not a placeholder. `modal_app.py`
serves authenticated `/generate` and `/judge` endpoints on GPU-backed,
per-model container pools. Within each debate round, the Space fans model calls
out concurrently and restores deterministic model order before consensus. The
Judge Agreement tab itself displays a fixed cached benchmark; `/judge` is used
to regenerate that benchmark, not to cross-check each score or certificate.

The exploratory probe uses the Space's ZeroGPU hardware directly. One
`@spaces.GPU(duration=60)` call holds a single RTX Pro 6000 allocation while
both selected checkpoints run the full internal probe batch; it does not
re-enter the shared GPU queue for every prompt. Modal remains the separate,
authenticated multi-model debate and judge backend.

The hosted app is cloud-dependent: the exploratory probe uses Hugging Face
ZeroGPU, while live debate and judge-cache generation use Modal. The public
probe intentionally exposes no separate inference-provider API path. Static
scoring, cached evidence, and local signature verification do not make the
complete hosted workflow off-grid.

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
connected [GitHub repository](https://github.com/Sahil170595/huggingface-RTSI)
contains Codex-attributed commits, while the reviewable build trace is public at
[Crusadersk/quantsafe-agent-trace](https://huggingface.co/datasets/Crusadersk/quantsafe-agent-trace),
including the final live restart test that proved the published Ed25519 issuer
remains stable.

## Reproducibility and privacy

- All local and Modal `from_pretrained` calls use audited 40-character commit revisions, including the fine-tuned classifier.
- The 51-row study comprises 6 baselines and 45 non-baseline cells; the signed screening substrate and cached judge/debate outputs are versioned under `substrate/`.
- Judge regeneration writes an immutable manifest before explicit promotion. The current run is [`judge-run-20260615T002149Z-3cf88d864691.json`](substrate/judge_runs/judge-run-20260615T002149Z-3cf88d864691.json), bound to code revision `00f1a8d`, the corpus SHA-256, exact model revisions, generation settings, reported precision, and raw-output hashes.
- Probe prompts and raw live completions are never rendered in the UI.
- Version 2 records bind the publisher's release target and sign a content-addressed evidence manifest. The verifier enforces v2 schema, artifact mapping, and band/action consistency in addition to Ed25519 issuer verification.
- Records are verified against this Space's pinned issuer public key (`9a074a15598fef26f5fbd33e8d604cb6c2372989f164331c11018a83fcd98519`); see [Verify a signed record](#verify-a-signed-record) and the Foreign re-sign test.
- The private signing key and Modal bearer token live only in deployment secrets.

## Build Small submission status

| Deliverable | Status |
|---|---|
| Public Gradio Space | Live |
| Demo storyboard | Ready in [`demo/STORYBOARD.md`](demo/STORYBOARD.md) |
| Public demo video | [`demo/quantsafe-demo.webm`](demo/quantsafe-demo.webm), 35.7 seconds, hard-captioned; [MP4](demo/quantsafe-demo.mp4) for social upload |
| Official hackathon organization | Complete: `build-small-hackathon` |

## Local verification

```bash
python -m pytest -q
ruff check .
python app.py
```

The UI uses a custom editorial theme, responsive mobile header, native tab overflow, explicit component spacing, and no Gradio analytics.
