# QuantSafe Certifier — Submission Checklist

## 1. Three Required Deliverables

- [x] **Final public Space URL** — `https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier`
- [x] **Demo video** — `demo/quantsafe-demo.webm` (69 s, 1280x720)
- [x] **Official org** — `build-small-hackathon`

---

## 2. Six-Tab Tour (one line each)

| Tab | What it shows | Headline number |
|---|---|---|
| **Score a config** | Static refusal-drift lookup across 45 measured (model, quant) cells — 23 LOW / 13 MODERATE / 9 HIGH | AUC 0.8445 |
| **Exploratory live probe** | Compares two live small-model checkpoints and reports aggregate drift; it is explicitly outside the matched baseline/quant calibration | 97.73% external XSTest classifier accuracy |
| **Judge Agreement** | Two independent safety classifiers label a 40-prompt corpus; agreement and curated-label accuracy are reported separately | kappa = 0.7484 (RELIABLE); 35/40 agree; unanimous decisions are 94.3% accurate |
| **Signed Screening Record** | Ed25519-signed record over a publisher-linked release revision, content-addressed evidence, screen result, cohort-level kappa, and action (`SCREEN_PASS` / `REVIEW` / `ROUTE`), verified against the pinned issuer key | release-target-bound and tamper-evident |
| **Constitutional Debate** | Small models argue "deploy or route" on MODERATE / MIXED configs under a constitution and reach consensus | cached example: 3 models -> CONDITIONAL at 0.67 agreement (genuine 2/3 majority) |
| **About** | Defines the study-internal scope, validation, paper relationship, and limitations | arXiv:2606.10154 |

---

## 3. Hard-Constraint Checks

### Total runtime model catalog <=32B

| Role | Models | Size |
|---|---|---|
| Refusal substrate (Score a config) | qwen2.5-1.5b, phi-2, llama3.2-1b, llama3.2-3b, qwen2.5-7b, mistral-7b | <=7B |
| Exploratory live probe | Qwen3-0.6B, Qwen3-1.7B, Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct (+ unsloth mirror) | <=2B |
| Semantic refusal cross-check | Crusadersk/quantsafe-refusal-modernbert | 0.150B |
| Safety judges (Judge Agreement) | Qwen3Guard-Gen-0.6B, Granite-Guardian-3.3-8b | 0.752B + 8.171B |
| Debate models (Constitutional Debate) | Qwen3-8B, Phi-4-mini-instruct, SmolLM3-3B | <=8.2B |

Counting every runtime repository listed in the Space model card, including
both equivalent Llama 3.2 1B repositories rather than deduplicating them, the
catalog totals **30.972674562B parameters**. The fixed reference matrix is stored
measurement data and does not load its source checkpoints at runtime.

### Gradio app

- `app.py` uses `import gradio as gr` and launches via `demo.launch()`.
- Space `README.md` YAML front matter has `sdk: gradio`.

### HF Space

- Final Space: `huggingface.co/spaces/build-small-hackathon/quantsafe-certifier`.
- `requirements.txt` lists `gradio`, `numpy`, and all runtime deps.
- Hardware tier: ZeroGPU hosts the Space; authenticated Modal GPU endpoints power remote debate/judge inference.

---

## 4. Pre-Submission Exposure Grep

Run from the repo root. Must return zero matches before submitting:

```bash
grep -rniE "neurips|iclr|icml|openreview|submission #|under review|blind review" . \
  --exclude=rtsi_core.py \
  --exclude=SUBMISSION.md \
  --exclude-dir=.git \
  --exclude-dir=__pycache__
# Then run a second pass for the blind method-name acronyms, kept in an
# internal-only list (deliberately NOT enumerated in this public file).
```

Expected output: _(empty)_ — zero matches. `SUBMISSION.md` is excluded because this section's own command text would otherwise match itself; `.git` is excluded because packed history objects retain old text and are never served by the Space.

Note: `rtsi_core.py` is the vendored internal scorer — excluded as a known internal residual; its symbol names are not user-facing and do not appear in any UI tab.

---

## 5. Move the Final Space into the Official Organization

The organization-owned Space is public. Recheck before submitting:

1. Confirm `build-small-hackathon/quantsafe-certifier` reaches `RUNNING`.
2. Confirm every tab loads and the live debate button is enabled.
3. Confirm README and demo overlays use the organization URL.

---

## 6. Modal Deployment Runbook

The live backend is currently deployed and wired. Use this runbook after backend changes:

1. Deploy `modal_app.py` to Modal:
   ```bash
   modal deploy modal_app.py
   ```
2. Copy the HTTPS endpoint URL printed by Modal after deploy.
3. In the HF Space secrets panel, set:
   ```
   MODAL_ENDPOINT=<the endpoint URL from step 2>
   ```
4. Restart the Space (Settings -> Factory reboot).
5. Confirm the "Run live debate" button is active and run an authenticated smoke request.

Note: the cached example (Qwen3-8B + Phi-4-mini-instruct + SmolLM3-3B, MODERATE/MIXED config, CONDITIONAL at 0.67 agreement) plays back correctly without Modal.

---

## 7. Warm the Space Before Recording

HF Spaces sleep after inactivity. Before recording the demo video:

1. Open `https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier` in a browser.
2. Wait for the status indicator to go green.
3. On the Exploratory live probe tab, use the remote backend for a short smoke run; do not present this cross-model result as a calibrated release decision.
4. Then start recording — the first real run in the video reuses the cached weights.

The exploratory tab runs each probe sequentially and shows per-probe progress. The first cold run is the slow part, so warm the selected backend before recording and do not include the cold-start in the final cut.

---

## 8. Verified Headline Numbers (do not alter)

| Claim | Value | Source |
|---|---|---|
| Measured (model, quant) cells | 45 | tr163_analysis.json |
| Risk split | 23 LOW / 13 MODERATE / 9 HIGH | tr163_analysis.json |
| ROC AUC (leave-one-cell-out) | 0.8445 | tr163_analysis.json |
| ROC AUC (leave-one-model-family-out) | 0.8403 (95% bootstrap CI 0.7080–0.9475) | validation_report.json |
| Fraction of configs routed (HIGH band) | 20% (9/45) | tr163_analysis.json -> in_sample.high_band |
| Refusal-rate gap recovered (HIGH band) | 76.17% | tr163_analysis.json -> in_sample.high_band |
| total_gap | 0.113778 | tr163_analysis.json |
| phi-2 + GPTQ refusal_rate_delta | -0.90 (loses 90 percentage points) | rtsi_table.csv |
| phi-2 + GPTQ score | 0.6199, HIGH | rtsi_table.csv |
| qwen2.5-1.5b + GPTQ score (highest-risk cell) | 0.7864, HIGH | rtsi_table.csv |
| Inter-judge Cohen's kappa | 0.7484 (RELIABLE) | judge_results.json (Qwen3Guard-Gen-0.6B + Granite-Guardian-3.3-8b) |
| Judges agree / split | 35/40 agree, 5 split | judge_results.json |
| Judge curated-label accuracy | Qwen3Guard 85.0%; Granite 92.5% | judge_results.json |
| Unanimous-panel selective accuracy | 94.3% at 87.5% coverage | judge_results.json |
| Fine-tuned semantic refusal classifier | 97.73% accuracy; 0.976 refusal F1 on 441 XSTest responses | Crusadersk/quantsafe-refusal-modernbert/metrics.json |
| Legacy opener lexicon on same XSTest split | 52.61% accuracy; 0.154 refusal F1 | Crusadersk/quantsafe-refusal-modernbert/metrics.json |
| Debate example consensus | CONDITIONAL at 0.67 agreement (2 CONDITIONAL, 1 ROUTE) | debate_examples.json (Qwen3-8B + Phi-4-mini-instruct + SmolLM3-3B) |
