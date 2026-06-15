# QuantSafe Certifier - Demo Storyboard

Target public cut: [`quantsafe-demo.webm`](quantsafe-demo.webm), a roughly
36-second 1280x720 walkthrough built from verified production captures. It uses
hard captions so every claim remains readable without audio. The social-ready
H.264 copy is [`quantsafe-demo.mp4`](quantsafe-demo.mp4).

## Shot List

### 1. Hook (0-4 s)

> One of my quantized releases kept its benchmarks and lost its refusals.

Open on the concrete publisher failure: **91% to 1%**.

### 2. Failure and route (4-12 s)

Show the measured `phi-2 + GPTQ` cell and Pareto route decision.

> Benchmarks stayed flat. Refusals collapsed.

> QuantSafe calls HIGH and returns ROUTE. Route the riskiest 20% and recover
> 76% of the measured refusal-rate gap.

### 3. Nemotron cross-check (12-16 s)

Show the three-family Judge Agreement tab.

> Three guard-model families expose where the evidence splits. The Nemotron
> guard has the highest point estimate: 95% on this 40-item project-labeled
> corpus.

This is fixed-corpus cohort evidence, not a config-specific safety judgment.

### 4. Bind, verify, and attack (16-27 s)

Issue a record for the published GPTQ artifact, verify it against the
README-published issuer key, then alter one signed field.

> Turn the decision into a portable signed record.

> The production issuer key verifies.

> Flip one signed field: INVALID.

The record is tamper-evident evidence of the screen, release target, and action.
It is not proof that the model is safe.

### 5. Publisher action (27-31 s)

Show the public model card warning on the screened release.

> The gate changed a real public release. The model card now carries the ROUTE
> decision and requires direct safety evaluation before deployment.

### 6. Close (31-36 s)

Close on the measured evidence, sponsor/runtime stack, paper identifier, and
production URL.

## Verified Numbers

| Claim | Value |
|---|---:|
| Measured non-baseline cells | 45 |
| Risk split | 23 LOW / 13 MODERATE / 9 HIGH |
| phi-2 + GPTQ refusal change | 91% to 1% (-90 pp raw screen) |
| Highest RTSI cell | qwen2.5-1.5b + GPTQ, 0.7864 HIGH |
| Leave-one-cell-out ROC AUC | 0.8445 |
| Leave-one-family-out ROC AUC | 0.8403 |
| Judge agreement | Fleiss' kappa 0.7929; 95% CI 0.6641–0.9239 |
| Unanimous-panel accuracy | 97.1% at 85% coverage |
| NVIDIA judge evidence | Nemotron guard 95.0% point estimate; p=1.0 vs Granite |
| Fine-tuned refusal classifier | 97.73% accuracy / 0.976 F1 |
| GGUF llama.cpp cells | 34 |
| Largest runtime model | Qwen3-8B, 8,190,735,360 parameters |

The source Space is
<https://huggingface.co/spaces/build-small-hackathon/quantsafe-certifier>.
