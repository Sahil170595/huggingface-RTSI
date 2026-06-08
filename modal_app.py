"""modal_app.py — Modal GPU backend for the Constitutional Debate tab.

Serves debate.py's backend="modal" path.  The endpoint accepts a single-prompt
generation request and returns a single text completion — exactly what
debate.py's _generate_modal() POSTs and reads.

API contract (POST /generate):
    Request  JSON: {"model": "<hf_model_id>", "prompt": "<text>", "max_new_tokens": 220}
    Response JSON: {"text": "<completion>"}

Allowed model IDs (hardcoded allowlist — reject unknown model strings):
    "Qwen/Qwen2.5-7B-Instruct"          (default; production debate quality)
    "Qwen/Qwen2.5-1.5B-Instruct"        (small; fast round-trips during dev)
    "Qwen/Qwen2.5-0.5B-Instruct"        (tiny; CI / smoke tests)
    "mistralai/Mistral-7B-Instruct-v0.3" (second debate voice alongside Qwen-7B)
    "HuggingFaceTB/SmolLM2-1.7B-Instruct" (third ungated small model)

Multi-model debate support:
    Each Modal container loads ONE model (the `model_id` parameter).  The debate
    engine calls /generate once per model per round.  Modal cold-starts one
    container per distinct model_id and keeps them warm in parallel — so a 3-model
    debate issues 3 concurrent calls to 3 containers, not 3 sequential calls to 1.

GPU:
    Default "a10g" (24 GB VRAM) — fits Qwen-7B + Mistral-7B comfortably in fp16.
    Swap to "t4" for the 0.5B / 1.5B models if you want to save credits.

=== DEPLOY RUNBOOK (run these steps when Modal credits are approved) ===

    1.  Install the Modal client (do this once):
            pip install modal

    2.  Authenticate (opens a browser, links to your Modal account):
            modal setup

    3.  Deploy this file:
            modal deploy modal_app.py

        Modal prints a URL like:
            https://<your-workspace>--debate-generate.modal.run

    4.  Copy that URL into the HF Space secret (or local .env):
            MODAL_ENDPOINT=https://<your-workspace>--debate-generate.modal.run

    5.  In the Debate tab (debate.py / app.py) set backend="modal".
        No code change needed — debate.py reads MODAL_ENDPOINT at call time.

    6.  Verify the endpoint is live:
            curl -s -X POST $MODAL_ENDPOINT \
              -H "Content-Type: application/json" \
              -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"Hello","max_new_tokens":20}' \
            | python -m json.tool
        Expect: {"text": "..."}

    7.  To change GPU tier (e.g. "t4" for smaller models):
            Edit the gpu= argument on DebateInferenceServer and redeploy.
            No Space-side changes needed.

    8.  To add a new allowed model:
            Add its HF model ID to ALLOWED_MODELS below and redeploy.

=== END RUNBOOK ===
"""

from __future__ import annotations

import os
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Allowlist — reject any model string not in this set to prevent abuse.
# Multi-model debates reference these by exact HF repo name.
# ---------------------------------------------------------------------------

ALLOWED_MODELS: set[str] = {
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ---------------------------------------------------------------------------
# Container image — torch + transformers in fp16, bitsandbytes for 4-bit on A10g
# ---------------------------------------------------------------------------

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",   # 4-bit quantisation on A10g for 7B models
        "sentencepiece>=0.1.99",
        "protobuf>=4.25.0",       # required by sentencepiece wheels
    )
)

app = modal.App("debate-backend", image=_image)

# ---------------------------------------------------------------------------
# GPU inference class — one container per model_id, loaded once at cold-start.
#
# Modal spawns a separate container for each distinct model_id parameter value.
# That means a 3-model debate gets 3 containers running in parallel — generation
# latency per round is bounded by the slowest single model, not sum(models).
# ---------------------------------------------------------------------------

@app.cls(
    gpu="a10g",         # 24 GB VRAM; fits Qwen-7B + Mistral-7B in fp16 easily
    timeout=300,        # seconds; a single 220-token generation << 60 s on A10g
    container_idle_timeout=120,  # keep warm between debate rounds (2-min window)
)
class DebateInferenceServer:
    """Loads one instruct model at container boot; serves single-prompt generation.

    The model_id parameter is baked into the container at deploy time.  Modal
    routes each unique model_id to its own container pool, so concurrent
    multi-model debates don't queue behind each other.
    """

    model_id: str = modal.parameter(default=_DEFAULT_MODEL)

    @modal.enter()
    def load(self) -> None:
        """Cold-start: download + load model into GPU memory once."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if self.model_id not in ALLOWED_MODELS:
            raise ValueError(
                f"model_id {self.model_id!r} is not in the allowed list. "
                f"Allowed: {sorted(ALLOWED_MODELS)}"
            )

        # Use 4-bit NF4 quantisation for 7B models to keep VRAM under 10 GB.
        # 0.5B / 1.5B models skip quantisation (they're already tiny).
        use_4bit = "7B" in self.model_id or "7b" in self.model_id
        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            if use_4bit
            else None
        )

        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.mdl.eval()

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 220) -> str:
        """Generate a single completion for one debate model turn.

        Args:
            prompt:         The full prompt string (system + user + prior turns).
            max_new_tokens: Token budget for this generation step.

        Returns:
            The generated completion text (decoded, no prompt echo).
        """
        import torch

        # Apply chat template when the tokeniser ships one (all instruct models do).
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
                do_sample=True,
                temperature=0.7,        # slight diversity between debating models
                top_p=0.9,
                repetition_penalty=1.1, # prevent looping on short contexts
                pad_token_id=self.tok.eos_token_id,
            )

        gen_ids = out_ids[0, prompt_len:]
        return self.tok.decode(gen_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Web endpoint — HTTP POST /generate, matching debate.py's modal backend path.
#
# debate.py sends:  POST MODAL_ENDPOINT  {"model": "...", "prompt": "...", "max_new_tokens": 220}
# This returns:                          {"text": "..."}
#
# The @modal.web_endpoint label becomes the URL path suffix printed by `modal deploy`.
# ---------------------------------------------------------------------------

@app.function()
@modal.web_endpoint(method="POST", label="generate")
def generate_endpoint(body: dict[str, Any]) -> dict[str, str]:
    """HTTP POST handler.  Validates the request and delegates to the GPU class.

    Request JSON:
        {
            "model":          "<hf_model_id>",     # must be in ALLOWED_MODELS
            "prompt":         "<text>",
            "max_new_tokens": 220                  # optional, default 220
        }

    Response JSON:
        {"text": "<completion>"}

    Error responses (HTTP 400):
        {"error": "<message>"}
    """
    model_id: str = body.get("model", _DEFAULT_MODEL)
    prompt: str = body.get("prompt", "")
    max_new_tokens: int = int(body.get("max_new_tokens", 220))

    if model_id not in ALLOWED_MODELS:
        # Return a structured error rather than a 500 so debate.py can surface it.
        return {"error": f"model {model_id!r} not allowed. Allowed: {sorted(ALLOWED_MODELS)}"}

    if not prompt.strip():
        return {"error": "prompt must be a non-empty string"}

    if not (1 <= max_new_tokens <= 1024):
        max_new_tokens = max(1, min(max_new_tokens, 1024))

    server = DebateInferenceServer(model_id=model_id)
    text = server.generate.remote(prompt, max_new_tokens)
    return {"text": text}
