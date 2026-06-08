"""modal_app.py — Modal GPU endpoint for the QuantSafe Router live tab.

This is the $270-Modal-credits sponsor-lane accelerator. It is OFF by default;
the Space runs fine on the "cpu" backend without it.

Deploy:
    modal deploy modal_app.py

Then set the env var in your HF Space secrets (or .env):
    MODAL_ENDPOINT=<URL printed by `modal deploy`>

Run the Space live tab with backend="modal" to use the GPU path.

API contract (POST /infer):
    Request  JSON: {"prompts": [...], "max_new_tokens": 64}
    Response JSON: {"completions": [...], "token_counts": [...]}
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Container image — torch + transformers, enough to run <=7B instruct models
# ---------------------------------------------------------------------------

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "sentencepiece>=0.1.99",
    )
)

app = modal.App("quantsafe-router", image=_image)

# Default model — small, instruct-tuned, fits on a T4 easily.
_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# ---------------------------------------------------------------------------
# GPU function — model loaded once per container lifetime
# ---------------------------------------------------------------------------

@app.cls(gpu="t4", timeout=300)
class RTSIInferenceServer:
    """Loads the model at container boot; serves batched inference requests."""

    model_id: str = modal.parameter(default=_DEFAULT_MODEL)

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.mdl.eval()

    @modal.method()
    def generate(self, prompts: list[str], max_new_tokens: int = 64) -> dict:
        import torch
        completions: list[str] = []
        token_counts: list[int] = []
        for prompt in prompts:
            if getattr(self.tok, "chat_template", None):
                messages = [{"role": "user", "content": prompt}]
                enc_text = self.tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                enc_text = prompt
            input_ids = self.tok(enc_text, return_tensors="pt").input_ids.cuda()
            prompt_len = input_ids.shape[-1]
            with torch.no_grad():
                out_ids = self.mdl.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tok.eos_token_id,
                )
            gen_ids = out_ids[0, prompt_len:]
            text = self.tok.decode(gen_ids, skip_special_tokens=True)
            completions.append(text)
            token_counts.append(int(gen_ids.shape[-1]))
        return {"completions": completions, "token_counts": token_counts}


# ---------------------------------------------------------------------------
# Web endpoint — accepts HTTP POST, delegates to the GPU class
# ---------------------------------------------------------------------------

@app.function()
@modal.web_endpoint(method="POST", label="infer")
def infer_endpoint(body: dict) -> dict:
    """HTTP POST /infer — proxies to RTSIInferenceServer.generate().

    Request JSON:
        {"prompts": ["..."], "max_new_tokens": 64}

    Response JSON:
        {"completions": ["..."], "token_counts": [12]}
    """
    prompts: list[str] = body.get("prompts", [])
    max_new_tokens: int = int(body.get("max_new_tokens", 64))
    if not prompts:
        return {"completions": [], "token_counts": []}
    server = RTSIInferenceServer()
    return server.generate.remote(prompts, max_new_tokens)
