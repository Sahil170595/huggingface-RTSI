# QuantSafe Certifier: Field Notes

## The failure mode

Quantization evaluation usually asks whether task quality survived. That misses a deployment-relevant failure: a model can preserve ordinary benchmark behavior while its refusal behavior changes sharply.

The reference matrix contains a concrete example. `phi-2 + GPTQ` loses 0.90 refusal-rate points while retaining acceptable task quality. That motivated a lightweight behavioral screen that can decide where a full safety battery is worth paying for.

## Design

The Refusal Stability Screen compares a candidate with a baseline using four refusal-shape features: dominant prefix share, unique prefix rate, normalized prefix entropy, and mean refusal length. It deliberately does not use ground-truth safety labels at scoring time.

The workflow then adds three checks around that score:

1. Independent small safety judges measure whether the judge cohort itself agrees.
2. An Ed25519 certificate binds the score, judge agreement, and route decision.
3. A constitutional debate handles only genuinely contested cases rather than applying majority vote to foregone decisions.

## What worked

- A four-feature screen reached ROC AUC 0.8445 on the 45-cell matrix.
- Routing the HIGH band covers 20% of cells and recovers 76.17% of the measured refusal-rate gap.
- The judge cohort reached kappa 0.7531 and exposed five split cases instead of hiding them.
- A three-model debate produced a strict 2/3 CONDITIONAL majority for the cached contested example.
- Per-model Modal containers made remote debate turns naturally parallelizable.

## Engineering lessons

The first Modal implementation described parallel containers but called them sequentially from the debate engine. The audit corrected that mismatch by fanning out remote model calls within each round while retaining deterministic response order for consensus and cached output.

An end-to-end production run through the public Space completed two rounds across three models in **34.8 seconds**. The earlier cached sequential run recorded **195.3 seconds**. This is one observed warm-runtime comparison, not a general latency guarantee, but it confirms that the Space now uses the Modal container topology it documents.

Reproducibility also required more than pinning Python packages. Every model loader now pins an immutable Hugging Face repository commit, preventing an upstream `main` branch change from silently altering live behavior.

For the UI, most visible spacing came from Gradio HTML's implicit padding and a large mobile header. Explicit padding choices, responsive typography, and moving Google Fonts from a rejected CSS `@import` into the document head removed the console warning and tightened the first screen.

## Limits

- The 45-cell matrix is small and uses 2024-generation checkpoints.
- A refusal-shape shift is a triage signal, not proof of harmful capability.
- Probe-set sensitivity and model-family transfer need broader external validation.
- The cached judge and debate artifacts are reproducible records, but live stochastic generation can differ.
- Human review remains necessary for contested or high-impact deployments.

## Next experiment

The highest-value follow-up is a larger blinded matrix with more model families, multiple probe sets, and prospective evaluation on newly quantized checkpoints. That would test whether the current thresholds transfer or need family-specific calibration.
