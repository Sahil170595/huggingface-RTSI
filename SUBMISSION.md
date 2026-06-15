# QuantSafe Certifier — Submission Checklist

**Official deadline:** June 15, 2026 at 23:59 UTC
(June 15, 2026 at 7:59 PM EDT).

## 1. Required Submission Gates

- [x] **Final public Space URL** — `https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier`
- [x] **Demo video** — `demo/quantsafe-demo.webm` (35.7 s, 1280x720, hard-captioned), with `demo/quantsafe-demo.mp4` for social upload
- [x] **Official org** — `build-small-hackathon`
- [ ] **Public social post** — publish it, then link its URL from `README.md`
- [ ] **Field Guide submission** — run the official preflight and submit the final Space

---

## 2. Six-Tab Tour (one line each)

| Tab | What it shows | Headline number |
|---|---|---|
| **Score a config** | Static refusal-drift lookup across 45 measured (model, quant) cells — 23 LOW / 13 MODERATE / 9 HIGH | AUC 0.8445 |
| **Exploratory live probe** | Selects a pair from four live checkpoint options and reports aggregate drift; it is explicitly outside the matched baseline/quant calibration | 97.73% external XSTest classifier accuracy |
| **Judge Agreement** | Three specialist guards plus a separate MiniCPM reasoning cross-check are measured against external labels | Fleiss' kappa = 0.7929 on the 40-item project corpus; BeaverTails N=400: Qwen3Guard 84.0%, Granite Guardian 84.75%, Nemotron 81.0%, MiniCPM 74.5%; specialist-guard unanimous 89.76% at 83% coverage |
| **Signed Screening Record** | Tamper-evident Ed25519 release-screen record over a publisher-linked release revision, content-addressed evidence, screen result, cohort-level benchmark result, and action (`SCREEN_PASS` / `REVIEW` / `ROUTE`) | release-target-bound; not proof of model safety or a config-specific judge evaluation |
| **Constitutional Debate** | Three small models argue "deploy or route" across Modal and OpenBMB | cached example: Qwen3-8B + MiniCPM4.1-8B + SmolLM3-3B -> ROUTE at 0.67 agreement |
| **About** | Defines the study-internal scope, validation, paper relationship, and limitations | arXiv:2606.10154 |

---

## 3. Hard-Constraint Checks

### Merit badges

- `achievement:offbrand`: custom editorial Gradio UI.
- `achievement:welltuned`: published QuantSafe Refusal ModernBERT fine-tune.
- `achievement:llama`: 34 GGUF cells evaluated through llama.cpp via Ollama.
- `achievement:sharing`: public agent trace in the GitHub repo, Space, and Hub dataset.
- `achievement:fieldnotes`: published engineering report.

The app does not claim `achievement:offgrid`; ZeroGPU, Modal, and OpenBMB are explicit
cloud dependencies. Static score lookup and
cached evidence can render without live inference, but the complete hosted
workflow is not local-only.

### Every runtime model individually under 32B

| Role | Models | Largest model |
|---|---|---|
| Refusal substrate (Score a config) | qwen2.5-1.5b, phi-2, llama3.2-1b, llama3.2-3b, qwen2.5-7b, mistral-7b | 7B |
| Exploratory live probe | Four checkpoint options: Qwen3-0.6B, Qwen3-1.7B, Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct | 1.7B |
| Semantic refusal cross-check | Crusadersk/quantsafe-refusal-modernbert | 0.150B |
| Safety judges (Judge Agreement) | Qwen3Guard-Gen-0.6B, Granite-Guardian-3.3-8b, Llama-3.1-Nemotron-Safety-Guard-8B-v3 | 8.171B |
| Debate models (Constitutional Debate) | Qwen3-8B, MiniCPM4.1-8B, SmolLM3-3B | Qwen3-8B: 8,190,735,360 |

The Build Small cap applies per individual model, not to the summed catalog;
every runtime repository above clears it comfortably. The largest is
**Qwen3-8B at 8,190,735,360 parameters**. The fixed reference matrix is stored
measurement data and does not load its source checkpoints at runtime.

### NVIDIA evidence

- `nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3` is one of three judge models
  from distinct families in the fixed 40-item benchmark.
- Its 95.0% project-label accuracy is the highest point estimate on this
  project-labeled corpus, not a general model ranking; the paired comparison
  with Granite is McNemar `p=1.0`.
- The cached benchmark was generated through the authenticated Modal `/judge`
  backend with Nemotron in native BF16. The Judge Agreement tab does not call
  the Nemotron guard for every score or certificate, and the cohort result is
  not config-specific.

### OpenBMB evidence

- `sponsor:openbmb` is claimed because MiniCPM4.1-8B is a real runtime model,
  not a metadata-only mention.
- On the deterministic BeaverTails N=400 sample, MiniCPM reaches 74.5% accuracy
  and 0.742 macro-F1 as a general-reasoning moderation cross-check.
- In the cached hybrid debate, MiniCPM changes DEPLOY -> ROUTE after critique
  and joins the final 2/3 ROUTE majority.
- The Hub reference is pinned; the hosted provider revision is explicitly
  recorded as unreported.
- The sponsor-published endpoint is HTTP-only and uses the shared hackathon
  token; that transport limitation is explicit in the artifact.

### Test-your-own-quant API

- Inside **Score a config**, a collapsed *"Test your own quant · API-ready"*
  panel and the public, named endpoint `/screen_external_manifest` screen a
  user-supplied **aggregate-feature** manifest (no raw prompts or completions).
- The endpoint never loads a model, fetches a URL, or signs the result: the
  report is provisional, `signed: false`, scope
  `user-supplied-aggregate-evidence`, and is a **screening recommendation, not a
  safety certification**.
- Input is capped at 32 KB and strictly validated (NaN/inf, malformed SHAs, and
  out-of-range metrics rejected with no scoring); the request contract is frozen
  in `schemas/external_screen_v1.schema.json`.
- Reuses the frozen 45-row substrate scoring path; per-feature contributions sum
  to the RTSI score. Refusal collapse forces `HIGH`/`ROUTE`; both-sides-zero is
  `UNKNOWN`/`INSUFFICIENT_SIGNAL`. No existing score, certificate, provider, tab,
  or concurrency behavior changes.

### Gradio app

- `app.py` uses `import gradio as gr` and launches via `demo.launch()`.
- Space `README.md` YAML front matter has `sdk: gradio`.

### HF Space

- Final Space: `huggingface.co/spaces/build-small-hackathon/quantsafe-certifier`.
- `requirements.txt` lists `gradio`, `numpy`, and all runtime deps.
- Hardware/runtime split: ZeroGPU powers the batched two-checkpoint exploratory
  probe; authenticated Modal GPU endpoints and the OpenBMB MiniCPM API power
  live debate; Modal regenerates the judge cache; the Judge Agreement tab
  displays cached results.

---

## 4. Pre-Submission Exposure Grep

Run from the repo root. Must return zero matches before submitting:

```bash
grep -rniE "neurips|iclr|icml|openreview|submission #|under review|blind review|banterhearts|tr134" . \
  --exclude=rtsi_core.py \
  --exclude=SUBMISSION.md \
  --exclude-dir=.git \
  --exclude-dir=__pycache__
# Then run a second pass for the blind method-name acronyms, kept in an
# internal-only list (deliberately NOT enumerated in this public file).
```

Expected output: _(empty)_ — zero matches. `SUBMISSION.md` is excluded because this section's own command text would otherwise match itself; `.git` is excluded because packed history objects retain old text and are never served by the Space.

Note: the grep now also covers `substrate/*.json`. A path leak was found and scrubbed from the substrate JSON artifacts; re-run the exposure grep including those files to confirm zero matches.

Note: `rtsi_core.py` is the vendored internal scorer — excluded as a known internal residual; its symbol names are not user-facing and do not appear in any UI tab.

---

## 5. Move the Final Space into the Official Organization

The organization-owned Space is public. Recheck before submitting:

1. Confirm `build-small-hackathon/quantsafe-certifier` reaches `RUNNING`.
2. Confirm every tab loads and the live debate button is enabled.
3. Confirm README and demo overlays use the organization URL.

---

## 6. Provider Deployment Runbook

The live backend is currently deployed and wired. Use this runbook after backend changes:

1. Deploy `modal_app.py` to Modal:
   ```bash
   modal deploy modal_app.py
   ```
2. Copy the HTTPS endpoint URL printed by Modal after deploy.
3. In the HF Space secrets panel, set:
   ```
   MODAL_ENDPOINT=<the endpoint URL from step 2>
   MODAL_TOKEN=<the Modal bearer token>
   OPENBMB_API_KEY=<the Build Small OpenBMB key>
   ```
4. Restart the Space (Settings -> Factory reboot).
5. Confirm the "Run live debate" button is active and run an authenticated smoke request.

Note: the cached example (Qwen3-8B + MiniCPM4.1-8B + SmolLM3-3B, MODERATE config, ROUTE at 0.67 agreement) plays back without live provider calls.

---

## 7. Warm the Space Before Recording

HF Spaces sleep after inactivity. Before recording the demo video:

1. Open `https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier` in a browser.
2. Wait for the status indicator to go green.
3. On the Exploratory live probe tab, run the default ZeroGPU pair once; do not present this cross-model result as a calibrated release decision.
4. Then start recording — the first recorded run reuses the cached weights.

The exploratory tab decodes all ten probes as one tensor batch per checkpoint
inside a single 60-second ZeroGPU allocation. A measured warm production run
completed in about 30 seconds; warm the models before recording and cut any
cold-download wait from the final video.

---

## 8. Verified Headline Numbers (do not alter)

| Claim | Value | Source |
|---|---|---|
| Measured (model, quant) cells | 45 | tr163_analysis.json |
| Risk split | 23 LOW / 13 MODERATE / 9 HIGH | tr163_analysis.json |
| ROC AUC (leave-one-cell-out) | 0.8445 | tr163_analysis.json |
| ROC AUC (leave-one-model-family-out) | 0.8403 (95% bootstrap CI 0.7080–0.9475) | validation_report.json |
| Fraction of configs routed (HIGH band) | 22% (10/45) leave-one-cell-out (in-sample 20%, 9/45) | tr163_analysis.json -> out_of_sample_loocv.high_band |
| Refusal-rate gap recovered (HIGH band) | 76.37% leave-one-cell-out (in-sample 76.17%) | tr163_analysis.json -> out_of_sample_loocv.high_band |
| total_gap | 0.113778 | tr163_analysis.json |
| phi-2 + GPTQ refusal_rate_delta | -0.90 (loses 90 percentage points) | rtsi_table.csv |
| phi-2 + GPTQ score | 0.6199, HIGH | rtsi_table.csv |
| qwen2.5-1.5b + GPTQ score (highest-risk cell) | 0.7864, HIGH | rtsi_table.csv |
| Inter-judge Fleiss' kappa | 0.7929; zone-stratified bootstrap 95% CI 0.6641–0.9239 | judge_results.json (Qwen3Guard-Gen-0.6B + Granite-Guardian-3.3-8b + Llama-3.1-Nemotron-Safety-Guard-8B-v3) |
| Judges agree / split | 34/40 unanimous, 6 split (all borderline) | judge_results.json |
| Judge project-label accuracy | Qwen3Guard 85.0%; Granite 92.5%; Nemotron guard 95.0% (highest point estimate; paired McNemar p=1.0 vs Granite) | judge_results.json |
| Unanimous-panel selective accuracy | 97.1% at 85% coverage | judge_results.json |
| Fine-tuned semantic refusal classifier | 97.73% accuracy; 0.976 refusal F1 on 441 XSTest responses | Crusadersk/quantsafe-refusal-modernbert/metrics.json |
| Legacy opener lexicon on same XSTest split | 52.61% accuracy; 0.154 refusal F1 | Crusadersk/quantsafe-refusal-modernbert/metrics.json |
| Debate example consensus | ROUTE at 0.67 agreement (2 ROUTE, 1 DEPLOY), 49.3 s | debate_examples.json (Qwen3-8B + MiniCPM4.1-8B + SmolLM3-3B; Modal + OpenBMB) |
| External-labeled judge benchmark | BeaverTails 30k_test, N=400, seed 20260615, third-party human crowd labels; Qwen3Guard 84.0% [80.1–87.3] F1 0.854 cov 96.8%; Granite Guardian 84.75% [80.9–87.9] F1 0.847 cov 100%; Nemotron 81.0% [76.9–84.5] F1 0.808 cov 100%; MiniCPM 74.5% [70.0–78.5] F1 0.742 cov 100%; three-specialist-guard unanimous 89.76% [86.0–92.6] at 83% coverage | substrate/external_judge_eval.json |
| Prospective NF4 transfer (demonstration, n=2) | Falcon3-3B-Instruct: RTSI 0.0018, LOW, refusal_rate_delta +0.02, material_loss False; SmolLM2-1.7B-Instruct: RTSI 0.2408, MODERATE, refusal_rate_delta −0.10, material_loss True | substrate/prospective_validation.json |
| MiniCPM4.1-8B hosted evidence | OpenBMB API; 74.5% BeaverTails accuracy; live debate participant; Hub reference SHA pinned, provider revision unreported | substrate/external_judge_eval.json; substrate/debate_examples.json |
