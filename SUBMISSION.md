# QuantSafe Certifier — Submission Checklist

## 1. Three Required Deliverables

- [x] **Final public Space URL** — `https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier`
- [x] **Demo video** — `demo/quantsafe-demo.webm` (69 s, 1280x720)
- [ ] **Social post** — draft in `social/POST.md`; post to X and LinkedIn before submitting the form
- [x] **Official org** — `build-small-hackathon`

---

## 2. Five-Screen Tour (one line each)

| Tab | What it shows | Headline number |
|---|---|---|
| **Score a config** | Static refusal-drift lookup across 45 measured (model, quant) cells — 23 LOW / 13 MODERATE / 9 HIGH | AUC 0.8445 |
| **Live screen** | Runs a small model live (transformers) and computes the same refusal-drift score in real time | 9 HIGH cells = 20% of configs, recovers 76.17% of the refusal-rate gap |
| **Judge Agreement** | Two independent safety classifiers label a 40-prompt corpus; Cohen's kappa measures whether the judge cohort can be trusted | kappa = 0.75 (RELIABLE); 35/40 agree, 5 split |
| **Safety Certificate** | Ed25519-signed certificate over the screen results — verdict (PASS / REVIEW / ROUTE) + kappa, verified against this Space's pinned issuer key; tamper test flips a field and the signature catches it | tamper-evident |
| **Constitutional Debate** | Small models argue "deploy or route" on MODERATE / MIXED configs under a constitution and reach consensus | cached example: 3 models -> CONDITIONAL at 0.67 agreement (genuine 2/3 majority) |

---

## 3. Hard-Constraint Checks

### Model size <=32B (every model <=9B)

| Role | Models | Size |
|---|---|---|
| Refusal substrate (Score a config) | qwen2.5-1.5b, phi-2, llama3.2-1b, llama3.2-3b, qwen2.5-7b, mistral-7b | <=7B |
| Live screen | Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct | <=1.5B |
| Safety judges (Judge Agreement) | Qwen3Guard-Gen-8B, Granite-Guardian-3.3-8b | each <=8.2B |
| Debate models (Constitutional Debate) | Qwen3-8B, Phi-4-mini-instruct, SmolLM3-3B | <=8.2B |

All models pass the rule because each individual model is below 32B. The largest model in the workflow is approximately 8.2B.

### Gradio app

- `app.py` uses `import gradio as gr` and launches via `demo.launch()`.
- Space `README.md` YAML front matter has `sdk: gradio`.

### HF Space

- Final Space: `huggingface.co/spaces/build-small-hackathon/quantsafe-certifier`.
- `requirements.txt` lists `gradio`, `numpy`, and all runtime deps.
- Hardware tier: CPU Basic covers substrate lookup and the live CPU tab; authenticated Modal GPU endpoints power remote debate/judge inference.

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
3. Confirm README, social copy, and demo overlays use the organization URL.
4. Do not submit until the public social-post URL is in README.

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
3. On the Live screen tab: trigger one dummy run with the smallest model (Qwen3-0.6B) to load weights into memory and warm the cache.
4. Then start recording — the first real run in the video reuses the cached weights.

On CPU Basic the live screen runs each probe sequentially and shows per-probe progress; the first cold run (weight download + load) is the slow part, so warm it before recording and keep the default small model. Do not include the cold-start in the final cut.

---

## 8. Verified Headline Numbers (do not alter)

| Claim | Value | Source |
|---|---|---|
| Measured (model, quant) cells | 45 | tr163_analysis.json |
| Risk split | 23 LOW / 13 MODERATE / 9 HIGH | tr163_analysis.json |
| ROC AUC (leave-one-cell-out) | 0.8445 | tr163_analysis.json |
| Fraction of configs routed (HIGH band) | 20% (9/45) | tr163_analysis.json -> in_sample.high_band |
| Refusal-rate gap recovered (HIGH band) | 76.17% | tr163_analysis.json -> in_sample.high_band |
| total_gap | 0.113778 | tr163_analysis.json |
| phi-2 + GPTQ refusal_rate_delta | -0.90 (loses 90 percentage points) | rtsi_table.csv |
| phi-2 + GPTQ score | 0.6199, HIGH | rtsi_table.csv |
| qwen2.5-1.5b + GPTQ score (highest-risk cell) | 0.7864, HIGH | rtsi_table.csv |
| Inter-judge Cohen's kappa | 0.7531 (RELIABLE) | judge_results.json (Qwen3Guard-Gen-8B + Granite-Guardian-3.3-8b) |
| Judges agree / split | 35/40 agree, 5 split | judge_results.json |
| Debate example consensus | CONDITIONAL at 0.67 agreement (2 CONDITIONAL, 1 ROUTE) | debate_examples.json (Qwen3-8B + Phi-4-mini-instruct + SmolLM3-3B) |
