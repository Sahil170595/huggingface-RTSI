# QuantSafe Certifier: Field Notes

## The failure mode

Quantization evaluation usually asks whether task quality survived. That misses a deployment-relevant failure: a model can preserve ordinary benchmark behavior while its refusal behavior changes sharply.

The reference matrix contains a concrete example. `phi-2 + GPTQ` loses 0.90 refusal-rate points while retaining acceptable task quality. That motivated a lightweight behavioral screen that can decide where a full safety battery is worth paying for.

## Design

The Refusal Stability Screen compares a candidate with a baseline using four refusal-shape features: dominant prefix share, unique prefix rate, normalized prefix entropy, and mean refusal length. It deliberately does not use ground-truth safety labels at scoring time.

The workflow then adds four checks around that score:

1. A fine-tuned 149.6M-parameter ModernBERT classifier independently checks semantic refusal rates.
2. Independent small safety judges measure whether the judge cohort itself agrees.
3. An Ed25519 record binds the published artifact revision, frozen evidence
   hashes, score, judge-cohort result, and release-gate action.
4. A constitutional debate handles only genuinely contested cases rather than applying majority vote to foregone decisions.

## What worked

- A four-feature screen reached ROC AUC 0.8445 on the 45-cell matrix.
- Routing the HIGH band covers 20% of cells and recovers 76.17% of the measured refusal-rate gap.
- The smaller Qwen3Guard-Gen-0.6B plus Granite Guardian cohort reached kappa
  0.7484 and exposed five split cases instead of hiding them.
- Each judge is also checked against curated labels: Qwen3Guard reaches 85.0%
  accuracy, Granite reaches 92.5%, and unanimous non-unclear decisions are
  94.3% accurate over 87.5% of the corpus.
- Leave-one-model-family-out validation reaches AUC 0.8403 (95% stratified
  bootstrap CI 0.7080–0.9475), close to the row-level 0.8445 result.
- A project-specific refusal classifier trained on 37,934 balanced
  WildGuardMix pairs reaches 97.73% accuracy and 0.976 refusal F1 on 441
  external XSTest responses. The legacy opener lexicon reaches 52.61% and
  0.154 on the same responses.
- A three-model debate produced a strict 2/3 CONDITIONAL majority for the cached contested example.
- Per-model Modal containers made remote debate turns naturally parallelizable.

## Engineering lessons

The first Modal implementation described parallel containers but called them sequentially from the debate engine. The audit corrected that mismatch by fanning out remote model calls within each round while retaining deterministic response order for consensus and cached output.

An end-to-end production run through the public Space completed two rounds across three models in **34.8 seconds**. The earlier cached sequential run recorded **195.3 seconds**. This is one observed warm-runtime comparison, not a general latency guarantee, but it confirms that the Space now uses the Modal container topology it documents.

Reproducibility also required more than pinning Python packages. Every model loader now pins an immutable Hugging Face repository commit, preventing an upstream `main` branch change from silently altering live behavior.

The signed record follows the same rule. For the 11 published AWQ/GPTQ
checkpoints, it binds the exact Hub revision plus SHA-256 hashes of the matrix,
validation report, judge results, and scorer. Historical GGUF rows are labeled
`legacy-config-only` because the original study did not retain immutable weight
digests. A valid signature proves issuer identity and payload integrity; it
does not prove broad model safety.

The official challenge page states that total parameters must stay at or below
32B. Running the tiny Qwen3Guard-Gen-0.6B guard is a deliberate small-model bet:
paired with Granite Guardian it still reaches kappa 0.7484 (RELIABLE) and
surfaces five split cases instead of hiding them. Counting every runtime
repository, including the duplicate Llama 3.2 1B mirror and the fine-tuned
semantic classifier, the complete catalog totals 30.972674562B.

The semantic model is intentionally a cross-check rather than a replacement
for the lexical feature extractor. Replacing the feature definition after
calibration would make the 45-cell RTSI validation claims incomparable. The UI
therefore reports both signals and labels their roles explicitly.

For the UI, most visible spacing came from Gradio HTML's implicit padding and a large mobile header. Explicit padding choices, responsive typography, and moving Google Fonts from a rejected CSS `@import` into the document head removed the console warning and tightened the first screen.

## Limits

- The 45-cell matrix is small and uses 2024-generation checkpoints; the wide
  family-held-out AUC interval makes that uncertainty explicit.
- A refusal-shape shift is a triage signal, not proof of harmful capability.
- The thresholds are study-internal. Cross-stack and cross-model comparisons
  need recalibration; the live two-checkpoint tab is therefore exploratory only.
- The judge kappa is a cohort-level support metric, not a config-specific
  judgment.
- Probe-set sensitivity and model-family transfer need broader external validation.
- Curated judge labels are not a substitute for an independently collected,
  blinded human benchmark.
- XSTest measures refusal classification, not broad harmfulness detection or
  quantization robustness.
- The cached judge and debate artifacts are reproducible records, but live stochastic generation can differ.
- Human review remains necessary for contested or high-impact deployments.

## Next experiment

The highest-value follow-up is a larger blinded matrix with more model families, multiple probe sets, and prospective evaluation on newly quantized checkpoints. That would test whether the current thresholds transfer or need family-specific calibration.
