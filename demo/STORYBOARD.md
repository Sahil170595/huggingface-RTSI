# QuantSafe Certifier - Demo Storyboard

Current public cut: [`quantsafe-demo.webm`](quantsafe-demo.webm), a
48-second 1280x720 walkthrough captured from the organization-owned production
Space. It uses hard captions so every claim remains readable without audio.

## Shot List

### 1. The failure and release gate (0-8 s)

Show the measured configuration lookup, Pareto routing curve, and highest-risk
cell.

Caption:

> The failure: quantization kept benchmarks stable while refusals collapsed
> from 91% to 1%. QuantSafe routes the highest-drift cells.

### 2. Real ZeroGPU probe (8-16 s)

Show the completed Qwen3-0.6B versus Qwen3-1.7B exploratory run with the
`zerogpu` backend selected.

Caption:

> Real ZeroGPU: two Qwen checkpoints, 20 generations, one 60-second
> allocation. Aggregate results only; probes stay private.

This cross-model comparison is explicitly exploratory. It is not a calibrated
matched baseline/quantized verdict and cannot be used to issue a record.

### 3. Independent judge agreement (16-24 s)

Show the two-family judge table, kappa badge, and verdict chart.

Caption:

> Independent judges: kappa 0.75, RELIABLE. Unanimous decisions reach 94.3%
> accuracy at 87.5% coverage.

### 4. Pinned signed record (24-32 s)

Issue and verify a v2 record for a published GPTQ artifact. Keep the immutable
revision, `ROUTE` action, public key, and green `VALID` result visible.

Caption:

> Signed record v2: immutable Hub revision, evidence hashes, and action.
> Ed25519 verification is pinned to the published issuer key.

### 5. Constitutional debate (32-40 s)

Show the final round and consensus card from the cached three-family debate.

Caption:

> Contested cases escalate: three small-model families debate under a
> constitution, then reach a genuine two-thirds CONDITIONAL verdict.

### 6. Evidence and close (40-48 s)

Show the About tab with the research scope, calibration, and limitations.

Caption:

> 34 GGUF cells ran through llama.cpp via Ollama. Five merit badges.
> 30.973B total runtime catalog. arXiv:2606.10154.

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
