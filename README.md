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
  - achievement:sharing
  - achievement:fieldnotes
  - safety
  - quantization
  - llm
  - refusal
  - gradio
models:
  - Qwen/Qwen3-0.6B
  - Qwen/Qwen3-1.7B
  - Qwen/Qwen3-8B
  - microsoft/Phi-4-mini-instruct
  - HuggingFaceTB/SmolLM3-3B
  - Qwen/Qwen3Guard-Gen-8B
  - ibm-granite/granite-guardian-3.3-8b
---

# QuantSafe Certifier

QuantSafe Certifier is a small-model safety workflow for a practical deployment question: **did quantization preserve benchmark quality while silently damaging refusal behavior?**

It screens a model/quantization cell, routes risky configurations, cross-checks independent safety judges, issues an Ed25519-signed certificate, and escalates genuinely contested cases to a constitutional multi-model debate.

[Open the live Space](https://huggingface.co/spaces/Crusadersk/quantsafe-certifier) · [GitHub source](https://github.com/Sahil170595/huggingface-RTSI) · [Field notes](FIELD_NOTES.md) · [Codex build trace](AGENT_TRACE.md)

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
- **ROC AUC 0.8445**, including leave-one-cell-out validation
- Routing the 9 HIGH cells routes **20%** of configurations and recovers **76.17%** of the measured refusal-rate gap
- Two independent safety judges agree on **35/40** cases, Cohen's kappa **0.7531 (`RELIABLE`)**
- Cached three-model debate reaches **CONDITIONAL** at **0.67 agreement**, a genuine 2/3 majority

These are screening results on a fixed reference matrix, not a claim that the screen replaces a full safety evaluation. A HIGH result explicitly routes to the expensive safety path.

## Six-tab workflow

1. **Score a config**: inspect any measured model/quantization cell, the risk heatmap, and the routing Pareto curve.
2. **Live screen**: compare a baseline and candidate over a held-internal refusal probe set. Only aggregate features are shown.
3. **Judge Agreement**: inspect agreement between Qwen3Guard-Gen-8B and Granite Guardian 3.3 8B.
4. **Safety Certificate**: sign the score, band, judge agreement, and route decision with Ed25519.
5. **Constitutional Debate**: replay or run a Modal-backed debate for contested MODERATE/MIXED cases.
6. **About**: review the method, thresholds, calibration, and limitations.

## Small-model compliance

The Build Small limit applies to each model individually. Every model used here is at most approximately 8.2B parameters, well below the **32B per-model cap**.

| Role | Largest model |
|---|---|
| Live refusal screen | Qwen3-1.7B |
| Safety judges | Qwen3Guard-Gen-8B / Granite Guardian 3.3 8B |
| Constitutional debate | Qwen3-8B |
| Reference matrix | Mistral-7B / Qwen2.5-7B |

## Modal runtime

Modal is part of the production runtime, not a placeholder. `modal_app.py` serves authenticated `/generate` and `/judge` endpoints on GPU-backed, per-model container pools. Within each debate round, the Space fans independent model calls out concurrently and restores deterministic model order before consensus.

The endpoint requires `Authorization: Bearer $MODAL_TOKEN`; unknown models are rejected by an allowlist. Model downloads are pinned to immutable Hugging Face commit SHAs in `model_revisions.py`.

## Reproducibility and privacy

- All local and Modal `from_pretrained` calls use audited 40-character commit revisions.
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
| Public demo video URL | Pending recording/upload |
| Social copy | Ready in [`social/POST.md`](social/POST.md) |
| Public social post URL | Pending publication |
| Official hackathon organization | Blocked: membership is visible, but the current role/token cannot create org Spaces |

The required demo and social links will be added here after publication. Final eligibility also requires an organization owner to grant create access or move this Space into `build-small-hackathon`.

## Local verification

```bash
python -m pytest -q
ruff check app.py cert_signer.py debate.py features.py inference.py judges.py modal_app.py model_revisions.py rtsi_core.py scripts
python app.py
```

The UI uses a custom editorial theme, responsive mobile header, native tab overflow, explicit component spacing, and no Gradio analytics.
