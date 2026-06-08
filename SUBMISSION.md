# QuantSafe Router — Submission Checklist

## Three Required Deliverables

- [ ] **Space URL** — `https://huggingface.co/spaces/[your-username]/quant-safe-router`
- [ ] **Demo video** — 60–90 s, script in `demo/STORYBOARD.md`
- [ ] **Social post** — drafts in `social/POST.md`; post to X and LinkedIn before submitting the form

---

## Pre-Submission Exposure Check

Run this from the repo root. Must return zero matches:

```bash
grep -rniE "neurips|iclr|icml|openreview|submission #|under review|blind review|\bRTSI\b|\bJTP\b|\bTAIS\b|\bCRI\b" . --exclude=SUBMISSION.md --exclude-dir=__pycache__ --exclude=rtsi_core.py
```

Expected output: _(empty)_

_(Note: `rtsi_core.py` is the vendored internal scorer — excluded as a known internal residual; its symbol names are not user-facing.)_

---

## Hard Constraint Checks

**Model size ≤ 32B**
- Substrate models: qwen2.5-1.5b, phi-2, llama3.2-1b, llama3.2-3b, qwen2.5-7b, mistral-7b — all ≤ 7B. Pass.
- Live tab models: Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct — all ≤ 7B. Pass.

**Gradio app**
- `app.py` must use `import gradio as gr` and launch via `demo.launch()`.
- Space must have `sdk: gradio` in `README.md` YAML front matter.

**HF Space**
- Repo must be under `huggingface.co/spaces/`.
- `requirements.txt` must list `gradio`, `numpy`, and any other runtime deps.
- Hardware tier: CPU Basic is sufficient for the substrate and live-CPU tabs. Upgrade to GPU Small if using the `hf` or `modal` backend live.

---

## Safety / Exposure Rules (final gate)

- No raw probe prompts or raw completions displayed in any UI tab.
- No links to `github.com/Sahil170595/RTSI` (carries a venue citation).
- Cite validation only as: "validated by leave-one-cell-out, AUC 0.8445".

---

## Warm the Space Before Recording

HF Spaces go to sleep after inactivity. Before recording the demo video:

1. Open the Space URL in a browser.
2. Wait for it to wake (status indicator green).
3. For the Live tab: trigger one dummy run to load the model weights into memory.
4. Then start recording. The first real run in the video will be fast.

Cold-start model load (transformers CPU) can take 30–60 s for 1B models — do not include that in the final cut.

---

## Verified Headline Numbers (do not alter)

| Claim | Value | Source |
|---|---|---|
| Cells in substrate | 45 | tr163_analysis.json |
| Risk split | 23 LOW / 13 MODERATE / 9 HIGH | tr163_analysis.json |
| ROC AUC (in-sample + LOOCV) | 0.8445 | tr163_analysis.json |
| Fraction routed (HIGH band) | 20% (9/45) | tr163_analysis.json → in_sample.high_band |
| Gap recovered (HIGH band) | 76.17% | tr163_analysis.json → in_sample.high_band |
| total_gap | 0.113778 | tr163_analysis.json |
| phi-2 + GPTQ refusal_rate_delta | −0.90 | rtsi_table.csv |
| phi-2 + GPTQ rtsi_score | 0.6199, HIGH | rtsi_table.csv |
| qwen2.5-1.5b + GPTQ rtsi_score | 0.7864, HIGH | rtsi_table.csv |
