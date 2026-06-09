# QuantSafe Certifier — Launch Posts

---

## X (Twitter)

**Hook tweet (<=280 chars)**

> phi-2 + GPTQ: refusal rate collapsed 90 percentage points. Standard quality benchmarks saw nothing.
>
> QuantSafe Certifier screens, certifies, and debates quantized model safety — entirely on small models (<=9B).
>
> @huggingface Space: huggingface.co/spaces/Crusadersk/quantsafe-certifier
> Built with @Gradio | GPU via @modal_labs

---

**Thread (4 tweets)**

**Tweet 1 / 4 — The problem**

> A quant setting can silently delete your model's refusals.
>
> phi-2 + GPTQ: 90 percentage-point refusal collapse. qwen2.5-1.5b + GPTQ: highest refusal-drift score in the dataset (0.79).
>
> Task quality benchmarks flagged neither. Behavioral probing did.
>
> That's what QuantSafe Certifier screens for. 1/4

**Tweet 2 / 4 — The screens**

> Four tabs. One pipeline. All <=9B models.
>
> Refusal Stability: 45 measured (model, quant) cells, ROC AUC 0.84. Route just 20% of configs, recover 76% of the refusal-rate gap.
>
> Live Screen: score YOUR config in real time — no uploads, runs transformers directly in the Space.
>
> Judge Agreement: two independent classifiers (Llama-Guard-3-8B + ShieldGemma-9b), Cohen's kappa = 0.74. Flags the 4/40 contested cases for human review. 2/4

**Tweet 3 / 4 — The cryptographic angle**

> Screen results aren't just a number — they're SIGNED.
>
> Safety Certificate tab: Ed25519 signature over the verdict (PASS / REVIEW / ROUTE) + the judge-agreement kappa. Anyone can verify with the public key.
>
> Tamper test built in: flip one field, signature fails. Cryptographically tamper-evident safety attestation. Novel for this workflow. 3/4

**Tweet 4 / 4 — The debate**

> Contested configs (MODERATE refusal + mixed judge agreement) go to a constitutional debate.
>
> Three small models — Qwen2.5-1.5B, Qwen2.5-0.5B, SmolLM2-1.7B — argue "deploy or route" over rounds and reach consensus.
>
> Cached real result: consensus ROUTE at 0.67 agreement. Live-run button activates when the @modal_labs GPU backend is wired.
>
> The entire pipeline: screening + two judges + a 3-model debate = zero models above 9B. 4/4

---

## LinkedIn

**Draft**

A quantized model can look fine on every benchmark while silently losing its ability to refuse harmful prompts. No quality metric surfaces this. Behavioral probing does.

phi-2 + GPTQ dropped 90 percentage points of refusal rate. qwen2.5-1.5b + GPTQ scored the highest refusal-drift risk in a 45-cell dataset covering 6 models and 8 quantization levels. Standard evaluations flagged neither.

**QuantSafe Certifier** is a Gradio Space that runs a complete safety-certification workflow for any (model, quantization) config — four tabs, all on models no larger than 9B.

**Refusal Stability Screen**
Scores how much a quantization destabilizes a model's refusal behavior. 45 measured cells, ROC AUC 0.84 (leave-one-cell-out). Routing the 9 HIGH-risk cells — just 20% of configs — recovers 76% of the refusal-rate gap. A Live Screen tab scores your own config in real time using transformers directly in the Space; nothing is uploaded.

**Judge Agreement**
Two independent safety classifiers — Llama-Guard-3-8B and ShieldGemma-9b — label a 40-prompt internal corpus. Inter-judge Cohen's kappa = 0.74, which is reliable agreement. They agree on 36/40 and split on 4. The point is not that either judge is definitive; it is that cross-checking two independent classifiers MEASURES whether the judge cohort can be trusted for this config, and honestly surfaces the contested cases that need a human.

**Safety Certificate**
Ed25519-signed certificate over the screen verdict (PASS / REVIEW / ROUTE) and the kappa. Verifiable with the included public key. A built-in tamper test flips a field and shows the signature failing — cryptographically tamper-evident safety attestation. This is the part of the pipeline I haven't seen elsewhere: not just a score, but a signed, portable proof that a specific config was evaluated and by what criteria.

**Constitutional Debate**
For configs that land in the genuinely contested middle — MODERATE refusal drift AND mixed judge agreement — three small models (Qwen2.5-1.5B, Qwen2.5-0.5B, SmolLM2-1.7B) debate "deploy or route to a safe baseline" under a constitution and reach consensus. Cached real result: consensus ROUTE at 0.67 agreement after multi-round debate. The live-run button activates when the Modal GPU backend is wired.

The entire pipeline — screening, two judges, and a three-model constitutional debate — runs on models no larger than 9B. That is the thesis: real multi-model safety orchestration that fits inside a small-model budget, solving a real problem for anyone deploying quantized local models.

Built with Gradio, hosted on Hugging Face Spaces. GPU acceleration via Modal.

Try it: huggingface.co/spaces/Crusadersk/quantsafe-certifier

#MachineLearning #LLM #ModelSafety #Quantization #HuggingFace #Gradio

---

## Notes
- Replace the Space URL placeholder with the final public link before posting.
- The hook tweet is 278 chars — fits within the 280-char limit.
- LinkedIn draft intentionally omits the Demo video line; add `Demo video: [URL]` before the hashtags if a recording is available.
- Do NOT include any venue, review, or submission language in any post.
- Do NOT use the internal screen acronyms or protocol names in public-facing copy.
