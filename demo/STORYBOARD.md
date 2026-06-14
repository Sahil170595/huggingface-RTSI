# QuantSafe Certifier - Demo Storyboard

Current public cut: [`quantsafe-demo.webm`](quantsafe-demo.webm), a
49.4-second 1280x720 walkthrough built from verified captures of the
organization-owned production Space. It uses hard captions so every claim
remains readable without audio. The social-ready H.264 copy is
[`quantsafe-demo.mp4`](quantsafe-demo.mp4).

## Shot List

### 1. The hook (0-4 s)

Open on the concrete publisher failure rather than a product logo reel.

Caption:

> One of my quantized releases kept its benchmarks and lost its refusals.

### 2. Detect and route (4-13 s)

Show the measured phi-2 GPTQ cell, its 91% to 1% refusal collapse, the
`0.6199 HIGH` score, and the Pareto routing decision.

Caption:

> QuantSafe calls HIGH and blocks the release. Route the riskiest 20% and
> recover 76% of the measured refusal-rate gap.

### 3. Real ZeroGPU probe (13-18 s)

Show the completed Qwen3-0.6B versus Qwen3-1.7B exploratory run with the
`zerogpu` backend selected.

Caption:

> A real RTX Pro 6000 probe, not a mock. Two Qwen checkpoints, ten private
> probes, aggregate drift only. Completed in 27 seconds.

This cross-model comparison is explicitly exploratory. It is not a calibrated
matched baseline/quantized verdict and cannot be used to issue a record.

### 4. Bind the decision (18-23 s)

Issue a v2 record for a published GPTQ artifact. Keep the immutable Hub
revision, `ROUTE` action, evidence binding, and public issuer key visible.

Caption:

> Turn the decision into a portable signed record bound to a published Hub
> revision, evidence hashes, and issuer identity.

### 5. Verify and attack (23-31 s)

Show the green `VALID` result against the README-published issuer key, then the
red `INVALID` result after one signed field is changed.

Caption:

> Verification is pinned to the published production key, not the key inside
> the record. Flip one signed field and the signature fails.

### 6. Constitutional debate (31-40 s)

Show the three independent model families and the final consensus card from
the cached production debate.

Caption:

> Borderline calls escalate. Qwen3-8B, Phi-4-mini, and SmolLM3 reach a genuine
> two-thirds CONDITIONAL verdict while exposing the dissenting ROUTE vote.

### 7. Evidence and close (40-49.4 s)

Show the About tab, then close on the measured evidence, small-model stack,
paper identifier, and production URL.

Caption:

> 45 measured cells. 34 GGUF cells through llama.cpp. Family-transfer
> validation, a fine-tuned cross-check, and arXiv:2606.10154.

## Verified Numbers

| Claim | Value |
|---|---:|
| Measured non-baseline cells | 45 |
| Risk split | 23 LOW / 13 MODERATE / 9 HIGH |
| phi-2 + GPTQ refusal change | 91% to 1% (-90 pp raw screen) |
| Highest RTSI cell | qwen2.5-1.5b + GPTQ, 0.7864 HIGH |
| Leave-one-cell-out ROC AUC | 0.8445 |
| Leave-one-family-out ROC AUC | 0.8403 |
| Judge agreement | kappa 0.7484, RELIABLE |
| Unanimous-panel accuracy | 94.3% at 87.5% coverage |
| Fine-tuned refusal classifier | 97.73% accuracy / 0.976 F1 |
| Debate consensus | CONDITIONAL, 2/3 |
| GGUF llama.cpp cells | 34 |
| Runtime model catalog | 30.972674562B / 32B |

The source Space is
<https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier>.
