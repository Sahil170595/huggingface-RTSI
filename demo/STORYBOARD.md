# QuantSafe Router — Demo Storyboard (60–90 s)

## Arc Summary
Open on a silent safety failure → score it live → show the router fix it → close on the headline number.

---

## Shot List

### Shot 1 — Hook (0–10 s)
**Screen**: Title card, plain background.
**Voiceover**: "phi-2 quantized to GPTQ lost 90 points of refusal rate — from 91% down to 1%. The model still answered questions fine. Standard benchmarks wouldn't catch it."
**Text overlay**: `phi-2 + GPTQ: refusal_rate_delta = −0.90`

---

### Shot 2 — Open the Substrate tab (10–25 s)
**Screen**: Browser on the QuantSafe Router Space, "Substrate Explorer" tab active.
**Action**: Select `phi-2` from the model dropdown, `GPTQ` from the quant dropdown. Click Score.
**Screen shows**:
```
Refusal-drift score:  0.6199
Risk band:            HIGH
Decision:             Route to safe baseline
```
**Voiceover**: "The Refusal Stability Screen scores it 0.62 — HIGH risk. The router says: don't deploy this config, use the safe baseline instead."

---

### Shot 3 — Show the worst cell (25–38 s)
**Screen**: Still on Substrate Explorer. Select `qwen2.5-1.5b` + `GPTQ`.
**Screen shows**:
```
Refusal-drift score:  0.7864    ← highest-risk cell in the study
Risk band:            HIGH
Decision:             Route to safe baseline
```
**Voiceover**: "qwen2.5-1.5b GPTQ scores 0.79 — the single highest refusal-drift score across all 45 tested configurations."
**Camera lingers** on the four feature deltas panel (dominant_prefix_share_delta, unique_prefix_rate_delta, prefix_entropy_norm_delta, mean_tokens_refusal_delta) to show the signal.

---

### Shot 4 — The headline (38–50 s)
**Screen**: Cut to the "Routing Analysis" panel or a simple static chart of the Pareto curve.
**Voiceover**: "Route just 20% of configs — the 9 HIGH-risk cells — and you recover 76% of the quality-safety gap. AUC 0.84, validated leave-one-cell-out across 45 cells."
**Text overlay**:
```
20% routed → 76.17% of gap recovered
AUC = 0.8445  (LOOCV)
45 cells · 6 models · 8 quant levels
```

---

### Shot 5 — Live tab (50–75 s)
**Screen**: Switch to "Live Score" tab.
**Action**: Select `Qwen/Qwen2.5-1.5B-Instruct` from the model picker, pick a quant level (e.g. Q4_K_M), click "Run & Score".
**Screen shows**: progress bar while probe set runs, then:
```
Aggregate features computed
Refusal-drift score:  0.03
Risk band:            LOW
Decision:             Safe to deploy
```
**Voiceover**: "The Live tab runs a small probe set against the selected model right here in the Space — no raw prompts or completions displayed — and scores the config in real time."
**Note for recording**: warm the Space first (see SUBMISSION.md). First-run model load can take 30–60 s; cut before that or speed-ramp it.

---

### Shot 6 — Close (75–90 s)
**Screen**: Return to title card or Space landing.
**Voiceover**: "QuantSafe Router — four behavioral features, one score, one routing decision. No ground-truth labels at scoring time."
**Text overlay**:
```
QuantSafe Router
huggingface.co/spaces/[your-space-url]
```

---

## Numbers used in this storyboard (all verified from substrate)
| Claim | Source |
|---|---|
| phi-2 + GPTQ refusal_rate_delta = −0.90 | rtsi_table.csv row 4 |
| phi-2 + GPTQ refusal-drift score = 0.6199, HIGH | rtsi_table.csv row 4 |
| qwen2.5-1.5b + GPTQ refusal-drift score = 0.7864, HIGH | rtsi_table.csv row 2 |
| 20% routed, 76.17% gap recovered | tr163_analysis.json → in_sample.high_band |
| AUC = 0.8445, LOOCV | tr163_analysis.json → out_of_sample_loocv.roc_auc |
| 45 cells, 23 LOW / 13 MODERATE / 9 HIGH | tr163_analysis.json → risk_distribution |
