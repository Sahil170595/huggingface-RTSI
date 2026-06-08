# QuantSafe Router — Launch Posts

---

## X (Twitter)

**Draft A — lead with the failure**

> Some quant settings silently delete your model's refusals.
>
> phi-2 + GPTQ: refusal rate went from 91% → 1%. Standard quality benchmarks didn't flag it.
>
> QuantSafe Router scores a (model, quant) config for refusal-template stability and tells you whether to deploy or fall back to a safe baseline.
>
> Route just 20% of configs → recover 76% of the quality-safety gap. AUC 0.84, validated leave-one-cell-out.
>
> Built on @Gradio, hosted on @huggingface 🤗
> Accelerated by @modal_labs
>
> Try it: [Space URL]

---

**Draft B — lead with the number**

> 20% of quantized configs routed. 76% of the safety gap recovered.
>
> That's what RTSI — Refusal Template Stability Index — buys you: a four-feature behavioral screen that flags quantization cells where retained quality masks safety degradation.
>
> No ground-truth labels at scoring time. Just prefix entropy, token counts, and refusal-opening distributions.
>
> Live demo on @huggingface, built with @Gradio, accelerated by @modal_labs
>
> [Space URL]

---

## LinkedIn

**Draft**

> Quantization can silently break your model's safety behavior — and standard benchmarks won't tell you.
>
> We found that phi-2 + GPTQ dropped from 91% refusal rate to 1%. The model's task performance was fine. Only behavioral probing caught it.
>
> QuantSafe Router is a Gradio app that scores any (model, quantization) config on the Refusal Template Stability Index (RTSI). Four features derived entirely from refusal-output behavior — no ground-truth labels needed at scoring time:
>
> - Dominant prefix share shift
> - Unique prefix rate shift
> - Normalized prefix entropy shift
> - Mean refusal token count shift
>
> Key results across 45 tested cells (6 models ≤7B, 8 quant levels):
> - ROC AUC = 0.84, validated by leave-one-cell-out cross-validation
> - Route the 9 HIGH-risk cells (20% of configs) → recover 76% of the quality-safety gap
> - The highest-risk cell: qwen2.5-1.5b + GPTQ, RTSI score 0.79
>
> The Live tab runs probe sets directly in the Space so you can score your own model config without uploading completions anywhere.
>
> Built with Gradio and hosted on Hugging Face Spaces. GPU acceleration via Modal ($270 in credits from the HF Build Small hackathon — thank you @Modal).
>
> Try it: [Space URL]
> Demo video: [Video URL]
>
> #MachineLearning #LLM #ModelSafety #Quantization #HuggingFace #Gradio

---

## Notes
- Replace `[Space URL]` and `[Video URL]` before posting.
- Tag @huggingface and @Gradio on X; mention Modal.
- Do NOT include any venue, review, or submission language in any post.
