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
short_description: Screen quantized-model refusal drift and certify decisions.
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

QuantSafe Certifier is a small-model safety workflow for a practical deployment question: **did quantization preserve benchmark quality while silently damaging refusal behavior?**

It screens a model/quantization cell, routes risky configurations, cross-checks independent safety judges, issues an Ed25519-signed certificate, and escalates genuinely contested cases to a constitutional multi-model debate.

[Open the live Space](https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier) · [Watch the 69-second demo](demo/quantsafe-demo.webm) · [GitHub source](https://github.com/Sahil170595/huggingface-RTSI) · [Field notes](FIELD_NOTES.md)

## Why this matters

`phi-2 + GPTQ` retained ordinary benchmark quality but lost **90 percentage points of refusal rate**. The screen scores that cell `0.6199` (`HIGH`) and routes it to a safe baseline. `qwen2.5-1.5b + GPTQ` is the highest-risk measured cell at `0.7864`.

The screen uses four baseline-relative behavioral deltas:

| Feature | Signal |
|---|---|
| `dominant_prefix_share_delta` | Change in the most common refusal opening |
| `unique_prefix_rate_delta` | Change in refusal-prefix diversity |
| `prefix_entropy_norm_delta` | Change in normalized prefix entropy |
| `mean_tokens_refusal_delta` | Change in average refusal length |

The absolute deltas are normalized across the reference matrix and combined using empirical correlation weights: `0.2324 / 0.3228 / 0.1733 / 0.2714`.

## Validated results

- **45 measured cells** across 6 models and 8 quantization formats
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
2. **Live screen**: compare a baseline and candidate over a held-internal refusal probe set. The calibrated lexical score and fine-tuned semantic refusal rates are reported separately; only aggregates are shown.
3. **Judge Agreement**: inspect agreement and curated-label accuracy for Qwen3Guard-Gen-0.6B and Granite Guardian 3.3 8B.
4. **Safety Certificate**: sign the score, band, judge agreement, and route decision with Ed25519.
5. **Constitutional Debate**: replay or run a Modal-backed debate for contested MODERATE/MIXED cases.
6. **About**: review the method, thresholds, calibration, and limitations.

## Small-model compliance

The Build Small rule caps the **total model catalog at 32B parameters**. Counting
every runtime repository listed in this model card, including both equivalent
Llama 3.2 1B repositories rather than deduplicating them, QuantSafe totals
**30.972674562B parameters**.

| Role | Runtime catalog |
|---|---|
| Live refusal screen | Qwen3-0.6B, Qwen3-1.7B, Qwen2.5-1.5B, Llama 3.2 1B (two repositories) |
| Semantic refusal cross-check | QuantSafe Refusal ModernBERT (149.6M, fine-tuned from ModernBERT-base) |
| Safety judges | Qwen3Guard-Gen-0.6B, Granite Guardian 3.3 8B |
| Constitutional debate | Qwen3-8B, Phi-4-mini-instruct, SmolLM3-3B |

The 0.6B Qwen guard is deliberate rather than cosmetic: the
[Qwen3Guard report](https://huggingface.co/papers/2510.14276) reports an English
response-classification average of 82.0 for 0.6B versus 83.9 for 8B. On this
project's fixed 40-item corpus, replacing the 8B guard preserved an 85.0%
accuracy result and a RELIABLE two-family agreement band while reducing the
catalog by roughly 7.44B parameters.

The live semantic cross-check is a project-specific fine-tune published at
[Crusadersk/quantsafe-refusal-modernbert](https://huggingface.co/Crusadersk/quantsafe-refusal-modernbert).
It was trained on 37,934 balanced WildGuardMix prompt/response pairs and tested
on 441 unambiguous XSTest GPT-4 responses. It remains a separate supporting
signal rather than silently changing the frozen RTSI calibration.

## Modal runtime

Modal is part of the production runtime, not a placeholder. `modal_app.py` serves authenticated `/generate` and `/judge` endpoints on GPU-backed, per-model container pools. Within each debate round, the Space fans independent model calls out concurrently and restores deterministic model order before consensus.

The endpoint requires `Authorization: Bearer $MODAL_TOKEN`; unknown models are rejected by an allowlist. Model downloads are pinned to immutable Hugging Face commit SHAs in `model_revisions.py`.

## Reproducibility and privacy

- All local and Modal `from_pretrained` calls use audited 40-character commit revisions, including the fine-tuned classifier.
- The 45-cell substrate and cached judge/debate outputs are versioned under `substrate/`.
- Probe prompts and raw live completions are never rendered in the UI.
- Certificates are verified against this Space's pinned issuer public key:

```text
9a074a15598fef26f5fbd33e8d604cb6c2372989f164331c11018a83fcd98519
```

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
