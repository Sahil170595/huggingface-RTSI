# QuantSafe Certifier: Field Notes

## The failure mode

Quantization evaluation usually asks whether task quality survived. That misses a deployment-relevant failure: a model can preserve ordinary benchmark behavior while its refusal behavior changes sharply.

The reference matrix contains a concrete example. `phi-2 + GPTQ` loses 0.90 refusal-rate points while retaining acceptable task quality. That motivated a lightweight behavioral screen that can decide where a full safety battery is worth paying for.

I publish 11 public GPTQ/AWQ 4-bit checkpoints on Hugging Face. QuantSafe is
the release-screen workflow I built after this retrospective audit of my own
catalog: inspect a measured release target, assign SCREEN_PASS / REVIEW / ROUTE,
and retain a signed record of the screen and evidence version.

## Design

The Refusal Stability Screen compares a candidate with a baseline using four refusal-shape features: dominant prefix share, unique prefix rate, normalized prefix entropy, and mean refusal length. It deliberately does not use ground-truth safety labels at scoring time.

The workflow then adds four checks around that score:

1. A fine-tuned 149.6M-parameter ModernBERT classifier independently checks semantic refusal rates.
2. Three small safety judge models from distinct families measure fixed-corpus
   cohort agreement and project-label accuracy.
3. An Ed25519 tamper-evident release-screen record binds the published artifact
   revision, frozen evidence hashes, score, cohort-level benchmark result, and
   release-gate action.
4. A constitutional debate handles only genuinely contested cases rather than applying majority vote to foregone decisions.

## What worked

- A four-feature screen reached ROC AUC 0.8445 on the 45-cell matrix.
- Routing the HIGH band recovers 76.37% of the measured refusal-rate gap under leave-one-cell-out evaluation (22%, 10/45); the in-sample figure (76.17%, 20%, 9/45) is a mechanism demo per the tr163_analysis.json circularity note.
- Three judge models from distinct families — Qwen3Guard-Gen-0.6B, Granite
  Guardian, and NVIDIA Llama-3.1-Nemotron-Safety-Guard-8B-v3 — reached Fleiss'
  kappa 0.7929 and exposed six split cases (all borderline) instead of hiding
  them. A zone-stratified bootstrap gives a 95% interval of 0.6641–0.9239, so
  the point estimate meets the preset RELIABLE band while the interval crosses
  its 0.70 threshold.
- Each judge is also checked against project labels: Qwen3Guard reaches 85.0%
  accuracy, Granite reaches 92.5%, and the Nemotron guard reaches 95.0%, the highest
  point estimate on this 40-item project-labeled corpus. The one-item lead over
  Granite is not statistically separated (exact paired McNemar p=1.0).
  Unanimous non-unclear decisions are 97.1% accurate over 85% of the corpus.
- Leave-one-model-family-out validation reaches AUC 0.8403 (95% stratified
  bootstrap CI 0.7080–0.9475), close to the row-level 0.8445 result.
- A project-specific refusal classifier trained on 37,934 balanced
  WildGuardMix pairs reaches 97.73% accuracy and 0.976 refusal F1 on 441
  external XSTest responses. The legacy opener lexicon reaches 52.61% and
  0.154 on the same responses.
- A hybrid three-model debate produced a strict 2/3 ROUTE majority. MiniCPM
  changed from DEPLOY to ROUTE after reading the other models' arguments.
- Per-model Modal containers made remote debate turns naturally parallelizable.
- A single ZeroGPU allocation now batches both live checkpoints across the full
  exploratory probe set instead of queueing once per prompt.
- The 34-cell GGUF slice was run through llama.cpp via Ollama, covering the
  Q2_K through Q8_0 ladder before normalization into the matched matrix.
- **External-labeled judge benchmark** (PKU-Alignment/BeaverTails `30k_test`, N=400, seed 20260615): Qwen3Guard 84.0%, Granite Guardian 84.75%, Nemotron 81.0%, and MiniCPM4.1-8B 74.5%. The 89.76%-accuracy selective consensus at 83% coverage uses only the three specialist guards. MiniCPM is a separate general-reasoning cross-check.
- **Prospective NF4 transfer** (demonstration, n=2 cells, not a powered AUC): the frozen screen assigned Falcon3-3B-Instruct RTSI 0.0018 LOW with no measured refusal loss and SmolLM2-1.7B-Instruct RTSI 0.2408 MODERATE with a measured 10-point refusal-rate drop. This is directionally consistent evidence, not proof that the thresholds transfer to NF4.

## Engineering lessons

The first Modal implementation described parallel containers but called them sequentially from the debate engine. The audit corrected that mismatch by fanning out remote model calls within each round while retaining deterministic response order for consensus and cached output.

The published hybrid run completed two rounds across Modal and OpenBMB in
**49.3 seconds**. An earlier all-Modal parallel run completed in 34.8 seconds;
the original sequential cache recorded 195.3 seconds. These are individual
warm-runtime observations, not general latency guarantees.

The runtime split is deliberately explicit. Hugging Face ZeroGPU runs the
batched exploratory probe. Authenticated Modal per-model GPU containers and
the OpenBMB MiniCPM API run live debate; Modal regenerates the fixed judge
benchmark. The Judge Agreement tab displays that cache. The complete hosted
workflow is cloud-dependent, not off-grid.

Reproducibility also required more than pinning Python packages. Every model loader now pins an immutable Hugging Face repository commit, preventing an upstream `main` branch change from silently altering live behavior.

The external BeaverTails sample is also bound to dataset revision
`8401fe609d288129cc684a9b3be6a93e41cfe678` and an ordered 400-row corpus
SHA-256, so the MiniCPM and specialist-guard comparisons are not seed-only.

Judge regeneration now writes an immutable run artifact before any cache
promotion. The current artifact binds code revision `00f1a8d`, the corpus hash,
all three model revisions, generation settings, backend-reported precision
(including Nemotron BF16), elapsed time, verdict digest, and a SHA-256 digest
for every raw completion.

For the 11 published AWQ/GPTQ checkpoints, the signed record binds the
publisher's release-target revision plus a content-addressed evidence manifest.
The historical study did not retain weight digests, so this does not prove that
the linked revision generated the measurement. Historical GGUF rows are labeled
`legacy-config-only`. A valid record proves issuer identity, payload integrity,
and v2 policy consistency for the release-screen record; it does not prove that
the model was broadly safety-evaluated or is safe.

The official challenge rule caps each individual model at under 32B parameters;
every model QuantSafe runs clears that cap with room to spare. The largest is
Qwen3-8B at **8,190,735,360 parameters**. Running the tiny
Qwen3Guard-Gen-0.6B guard is still a deliberate small-model bet: together with
Granite Guardian and NVIDIA Llama-3.1-Nemotron-Safety-Guard-8B-v3, the
three-family judge cohort reaches Fleiss' kappa 0.7929 (RELIABLE) and surfaces
six split cases instead of hiding them. The Nemotron guard's 95.0% accuracy is the
highest point estimate on this fixed project-labeled corpus, not a general
ranking.

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
- The 40-item judge corpus uses single-author project labels (all 16 borderline items labeled "safe"); the external BeaverTails N=400 third-party-labeled benchmark is the corrective, where the three guards drop to 81-85% and Nemotron falls from the project-corpus top (95%) to last (81%) on independent labels.
- XSTest measures refusal classification, not broad harmfulness detection or
  quantization robustness.
- The cached judge and debate artifacts are reproducible records, but live stochastic generation can differ.
- Human review remains necessary for contested or high-impact deployments.
- OpenBMB MiniCPM4.1-8B is served by the sponsor's hosted API. Its Hub
  reference is pinned, but the provider does not report the exact served
  revision; the artifact records that limitation explicitly. The sponsor
  published an HTTP-only endpoint and shared challenge token, so transport
  confidentiality is not claimed.

## Next experiment

The highest-value follow-up is a larger blinded matrix with more model families, multiple probe sets, and prospective evaluation on newly quantized checkpoints. That would test whether the current thresholds transfer or need family-specific calibration.

The NF4 prospective demonstration (n=2 cells) produced directionally correct results but is not a powered generalization test. Scaling to a proper prospective cohort — more families, multiple NF4 seeds, matched baselines — would determine whether the thresholds hold or need NF4-specific recalibration.
