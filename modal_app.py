"""modal_app.py — Modal GPU backend for the Constitutional Debate tab.

Serves debate.py's backend="modal" path.  The endpoint accepts a single-prompt
generation request and returns a single text completion — exactly what
debate.py's _generate_modal() POSTs and reads.

API contract (POST /generate):
    Request  header: Authorization: Bearer <QUANTSAFE_MODAL_TOKEN>
                     (token lives in the modal.Secret "quantsafe-auth")
    Request  JSON: {"model": "<hf_model_id>", "prompt": "<text>", "max_new_tokens": 220}
    Response JSON: {"text": "<completion>", "quantization": "<actual precision>"}
    Errors:        HTTP 401 (secret unset, or bearer token missing/mismatched)
                   HTTP 400 (unknown model, empty prompt, bad max_new_tokens)
                   — FastAPI HTTPException, body {"detail": "<message>"}

Endpoints:
    POST /generate — debate-turn generation (models in DEBATE_MODELS)
    POST /judge    — safety-judge classification (models in JUDGE_MODELS);
                     request {"model", "prompt", "response", "max_new_tokens"},
                     response {"text": "<raw judge completion>", "quantization"};
                     the judge's own moderation chat template is applied
                     server-side (Granite Guardian needs guardian_config).

Allowed model IDs are the hardcoded DEBATE_MODELS / JUDGE_MODELS allowlists
below — unknown model strings are rejected with HTTP 400.

Multi-model debate support:
    Each Modal container loads ONE model (the `model_id` parameter).  The debate
    engine calls /generate once per model per round.  Modal cold-starts one
    container per distinct model_id and keeps them warm in parallel — so a 3-model
    debate issues 3 concurrent calls to 3 containers, not 3 sequential calls to 1.

GPU:
    Default "a10g" (24 GB VRAM) — fits Qwen-7B + Mistral-7B comfortably in fp16.
    Swap to "t4" for the 0.5B / 1.5B models if you want to save credits.

=== DEPLOY RUNBOOK (run after backend or model changes) ===

    1.  Install the Modal client + fastapi (do this once; fastapi is imported
        at module level for the endpoint's auth header, so the deploy machine
        needs it too):
            pip install modal fastapi

    2.  Authenticate (opens a browser, links to your Modal account):
            modal setup

    3.  Create the shared auth secret (once; deploy fails without it).
        Pick a long random token, e.g.:
            python -c "import secrets; print(secrets.token_urlsafe(32))"
            modal secret create quantsafe-auth QUANTSAFE_MODAL_TOKEN=<that-token>
        The SAME value must be set as the MODAL_TOKEN secret on the HF Space
        (clients send it as "Authorization: Bearer <MODAL_TOKEN>").

    4.  Deploy this file:
            modal deploy modal_app.py

        Modal prints a URL like:
            https://<your-workspace>--debate-generate.modal.run

    5.  Copy that URL into the HF Space secret (or local .env):
            MODAL_ENDPOINT=https://<your-workspace>--debate-generate.modal.run

    6.  In the Debate tab (debate.py / app.py) set backend="modal".
        No code change needed — debate.py reads MODAL_ENDPOINT at call time.

    7.  Verify the endpoint is live:
            curl -s -X POST $MODAL_ENDPOINT \
              -H "Content-Type: application/json" \
              -H "Authorization: Bearer $MODAL_TOKEN" \
              -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","prompt":"Hello","max_new_tokens":20}' \
            | python -m json.tool
        Expect: {"text": "...", "quantization": "fp16"}
        Without the Authorization header, expect HTTP 401 {"detail": "..."}.

    8.  To change GPU tier (e.g. "t4" for smaller models):
            Edit the gpu= argument on DebateInferenceServer and redeploy.
            No Space-side changes needed.

    9.  To add a new allowed model:
            Add its HF model ID to ALLOWED_MODELS below and redeploy.

=== END RUNBOOK ===
"""

import os
from typing import Any

# fastapi is needed at IMPORT time (Header() lives in the endpoint signature),
# both in the container (via fastapi[standard] in the image) and on the deploy
# machine: `pip install fastapi`. It is NOT a dependency of the modal client.
import fastapi
import modal

from model_revisions import model_revision

# NOTE: do NOT add `from __future__ import annotations` here — it stringizes the
# class annotations and breaks modal.parameter() type validation (model_id: str
# would arrive as the string 'str'). Modal needs the eager type object.

# ---------------------------------------------------------------------------
# Allowlist — reject any model string not in this set to prevent abuse.
# Multi-model debates reference these by exact HF repo name.
# ---------------------------------------------------------------------------

DEBATE_MODELS: set[str] = {
    # 2024-generation cohort (kept for cached-replay compatibility)
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    # Current Modal side of the hybrid debate cohort.
    "Qwen/Qwen3-8B",
    "HuggingFaceTB/SmolLM3-3B",
}

# Safety-judge models are served only by the /judge endpoint, which applies
# each judge's own classification chat template server-side (Granite Guardian
# additionally needs a guardian_config the generic /generate path cannot express).
JUDGE_MODELS: set[str] = {
    "Qwen/Qwen3Guard-Gen-0.6B",
    "ibm-granite/granite-guardian-3.3-8b",
    "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3",
}

ALLOWED_MODELS: set[str] = DEBATE_MODELS | JUDGE_MODELS

_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_INPUT_CHARS = 32_768
MAX_NEW_TOKENS = 1_024


MODEL_LOAD_POLICIES: dict[str, dict[str, object]] = {
    # Legacy 7B debate models retain their deployed NF4 memory policy.
    "Qwen/Qwen2.5-7B-Instruct": {
        "precision": "nf4-4bit",
        "torch_dtype": "float16",
        "load_in_4bit": True,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "precision": "nf4-4bit",
        "torch_dtype": "float16",
        "load_in_4bit": True,
    },
    # Remaining debate models retain their existing unquantized fp16 policy.
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "precision": "fp16",
        "torch_dtype": "float16",
        "load_in_4bit": False,
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "precision": "fp16",
        "torch_dtype": "float16",
        "load_in_4bit": False,
    },
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": {
        "precision": "fp16",
        "torch_dtype": "float16",
        "load_in_4bit": False,
    },
    "Qwen/Qwen3-8B": {
        "precision": "fp16",
        "torch_dtype": "float16",
        "load_in_4bit": False,
    },
    "HuggingFaceTB/SmolLM3-3B": {
        "precision": "fp16",
        "torch_dtype": "float16",
        "load_in_4bit": False,
    },
    # Judge policies are explicit because similarly sized models can require
    # different native dtypes.
    "Qwen/Qwen3Guard-Gen-0.6B": {
        "precision": "fp16",
        "torch_dtype": "float16",
        "load_in_4bit": False,
    },
    "ibm-granite/granite-guardian-3.3-8b": {
        "precision": "fp16",
        "torch_dtype": "float16",
        "load_in_4bit": False,
    },
    "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3": {
        "precision": "bf16",
        "torch_dtype": "bfloat16",
        "load_in_4bit": False,
    },
}


def _load_policy_for(model_id: str) -> dict[str, object]:
    """Return the explicit load policy for a served model."""
    try:
        return MODEL_LOAD_POLICIES[model_id]
    except KeyError as exc:
        raise ValueError(f"No Modal load policy configured for model {model_id!r}") from exc


def _dtype_precision(dtype: object) -> str:
    """Normalize a loaded torch dtype to the public precision label."""
    labels = {
        "torch.float16": "fp16",
        "torch.bfloat16": "bf16",
    }
    try:
        return labels[str(dtype)]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported loaded model dtype: {dtype}") from exc

# ---------------------------------------------------------------------------
# Container image — torch + transformers, bitsandbytes for NF4 on A10g
# ---------------------------------------------------------------------------

_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        # Current v5 ships Qwen3 (enable_thinking), SmolLM3,
        # Qwen3Guard-Gen and Granite-Guardian-3.3 chat templates.
        "transformers==5.12.0",
        "accelerate==1.14.0",
        "bitsandbytes==0.49.2",   # 4-bit quantisation on A10g for the legacy 7B cohort
        "sentencepiece==0.2.1",
        "protobuf==7.35.1",       # required by sentencepiece wheels
        "fastapi[standard]==0.137.0",  # Modal 1.x web endpoints are FastAPI-backed
    )
    # judges.py is the single source of truth for the NemoGuard classification
    # prompt (build_nemotron_guard_prompt). Its module-level imports are all
    # stdlib (numpy is lazy-imported inside the kappa helpers), so it is safe to
    # ship into the container image without pulling a heavy dependency at import.
    .add_local_python_source("model_revisions", "judges")
)

app = modal.App("debate-backend", image=_image)

# Persist the HF model cache across cold starts so the 7B weights download ONCE.
# A fresh container otherwise re-downloads ~28 GB (2x 7B) on every cold start
# (~3 min cold debate); with the volume, repeat cold-starts are load-only (~20-40 s).
_hf_cache = modal.Volume.from_name("debate-hf-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# GPU inference class — one container per model_id, loaded once at cold-start.
#
# Modal spawns a separate container for each distinct model_id parameter value.
# That means a 3-model debate gets 3 containers running in parallel — generation
# latency per round is bounded by the slowest single model, not sum(models).
# ---------------------------------------------------------------------------

@app.cls(
    gpu="A10G",         # 24 GB VRAM; fits Qwen-7B + Mistral-7B in fp16 easily
    timeout=300,        # seconds; a single 220-token generation << 60 s on A10g
    scaledown_window=300,  # keep warm ~5 min between debates during a judging session
    volumes={"/root/.cache/huggingface": _hf_cache},  # persist model downloads across cold starts
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

        policy = _load_policy_for(self.model_id)
        load_dtype = getattr(torch, str(policy["torch_dtype"]))
        use_4bit = bool(policy["load_in_4bit"])
        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=load_dtype,
            )
            if use_4bit
            else None
        )

        revision = model_revision(self.model_id)
        self.tok = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=revision,
        )
        self.mdl = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=revision,
            quantization_config=bnb_config,
            dtype=load_dtype,
            device_map="auto",
        )
        self.mdl.eval()
        if use_4bit:
            if not getattr(self.mdl, "is_loaded_in_4bit", False):
                raise RuntimeError(
                    f"{self.model_id} was configured for NF4 but did not load in 4-bit"
                )
            actual_precision = "nf4-4bit"
        else:
            actual_precision = _dtype_precision(self.mdl.dtype)

        expected_precision = str(policy["precision"])
        if actual_precision != expected_precision:
            raise RuntimeError(
                f"{self.model_id} loaded as {actual_precision}, expected "
                f"{expected_precision}"
            )
        self.precision = actual_precision

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 220) -> dict[str, str]:
        """Generate a single completion for one debate model turn.

        Args:
            prompt:         The full prompt string (system + user + prior turns).
            max_new_tokens: Token budget for this generation step.

        Returns:
            The decoded completion and the worker-verified load precision.
        """
        import torch

        # Apply chat template when the tokeniser ships one (all instruct models do).
        if getattr(self.tok, "chat_template", None):
            mid = self.model_id.lower()
            messages = [{"role": "user", "content": prompt}]
            template_kwargs: dict = {}
            # Reasoning-mode suppression: a 220-token debate turn cannot afford
            # a <think> preamble. Qwen3 exposes enable_thinking in its template;
            # SmolLM3 reads a /no_think system flag.
            if "qwen3" in mid and "guard" not in mid:
                template_kwargs["enable_thinking"] = False
            if "smollm3" in mid:
                messages = [{"role": "system", "content": "/no_think"}] + messages
            enc_text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
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
        text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        return {"text": text, "quantization": self.precision}

    @modal.method()
    def judge(
        self, prompt: str, response: str, max_new_tokens: int = 48
    ) -> dict[str, str]:
        """Classify one (prompt, response) pair with this container's judge model.

        Applies the judge's OWN moderation chat template (the whole reason the
        /judge endpoint exists — Granite Guardian needs guardian_config,
        Qwen3Guard moderates the conversation turns directly, and NemoGuard
        takes a single pre-rendered classification user message). Decoding is
        greedy: judge verdicts must be deterministic.

        Returns the raw completion and worker-verified load precision; the
        caller parses the verdict (judges.py parse_qwen3guard /
        parse_granite_guardian / parse_nemotron_guard).
        """
        import torch

        mid = self.model_id.lower()
        if "nemotron-safety-guard" in mid:
            # NemoGuard expects ONE user message whose content is the fully
            # rendered classification prompt (taxonomy + conversation + output
            # instruction). build_nemotron_guard_prompt is the single source of
            # truth for that string (judges.py), kept byte-exact with the
            # model's own inference_script. We then apply the tokenizer's
            # (Llama-3.1) chat template with add_generation_prompt=True.
            from judges import build_nemotron_guard_prompt

            rendered = build_nemotron_guard_prompt(prompt, response)
            enc_text = self.tok.apply_chat_template(
                [{"role": "user", "content": rendered}],
                tokenize=False,
                add_generation_prompt=True,
            )
        elif "granite-guardian" in mid:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            # Granite Guardian templates take the risk definition via
            # guardian_config; "harm" is the umbrella social-harm risk.
            enc_text = self.tok.apply_chat_template(
                messages,
                guardian_config={"risk_name": "harm"},
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            # Qwen3Guard-Gen: template formats the moderation request over the
            # conversation turns as-is.
            enc_text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

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
        text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        return {"text": text, "quantization": self.precision}


# ---------------------------------------------------------------------------
# Web endpoint — HTTP POST /generate, matching the shared client contract
# (debate.py's modal backend and inference.py's _infer_modal).
#
# Clients send:  POST MODAL_ENDPOINT  {"model": "...", "prompt": "...", "max_new_tokens": 220}
#                with header           Authorization: Bearer <QUANTSAFE_MODAL_TOKEN>
# This returns:                        {"text": "...", "quantization": "<actual precision>"}
# Errors:        fastapi.HTTPException -> {"detail": "..."} with 401 (auth) / 400 (input).
#
# The @modal.fastapi_endpoint label becomes the URL path suffix printed by `modal deploy`.
# ---------------------------------------------------------------------------

def _require_bearer_auth(authorization: str) -> None:
    """Shared bearer-token check for both web endpoints. Raises 401 on failure."""
    import hmac

    expected = os.environ.get("QUANTSAFE_MODAL_TOKEN", "")
    if not expected:
        raise fastapi.HTTPException(
            status_code=401,
            detail="endpoint auth is not configured: the quantsafe-auth secret "
                   "does not expose QUANTSAFE_MODAL_TOKEN",
        )
    if not hmac.compare_digest(authorization, f"Bearer {expected}"):
        raise fastapi.HTTPException(
            status_code=401,
            detail="missing or invalid Authorization header "
                   "(expected: 'Bearer <token>')",
        )


def _bounded_text(field: str, value: Any) -> str:
    """Validate one authenticated text input before scheduling GPU work."""
    if not isinstance(value, str) or not value.strip():
        raise fastapi.HTTPException(
            status_code=400, detail=f"{field} must be a non-empty string",
        )
    if len(value) > MAX_INPUT_CHARS:
        raise fastapi.HTTPException(
            status_code=400,
            detail=f"{field} exceeds the {MAX_INPUT_CHARS}-character limit",
        )
    return value


def _token_budget(body: dict[str, Any], default: int) -> int:
    """Parse a bounded generation budget; booleans are not integer budgets."""
    raw = body.get("max_new_tokens", default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise fastapi.HTTPException(
            status_code=400, detail="max_new_tokens must be an integer",
        )
    value = raw
    if not (1 <= value <= MAX_NEW_TOKENS):
        raise fastapi.HTTPException(
            status_code=400,
            detail=f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}",
        )
    return value


@app.function(secrets=[modal.Secret.from_name("quantsafe-auth")])
@modal.fastapi_endpoint(method="POST", label="generate")
def generate_endpoint(
    body: dict[str, Any],
    authorization: str = fastapi.Header(default=""),
) -> dict[str, str]:
    """HTTP POST handler.  Authenticates, validates, delegates to the GPU class.

    Auth:
        Requires "Authorization: Bearer <QUANTSAFE_MODAL_TOKEN>".  The expected
        token comes from the modal.Secret "quantsafe-auth".  If the secret is
        unset OR the header is missing/mismatched -> HTTP 401.

    Request JSON:
        {
            "model":          "<hf_model_id>",     # must be in ALLOWED_MODELS
            "prompt":         "<text>",
            "max_new_tokens": 220                  # optional, default 220
        }

    Response JSON (HTTP 200):
        {"text": "<completion>", "quantization": "<actual precision>"}

    Error responses (fastapi.HTTPException, body {"detail": "<message>"}):
        401  secret unset, or Authorization bearer token missing/mismatched
        400  unknown model, empty prompt, or non-integer max_new_tokens
    """
    _require_bearer_auth(authorization)

    model_id = body.get("model", _DEFAULT_MODEL)
    prompt = _bounded_text("prompt", body.get("prompt", ""))
    max_new_tokens = _token_budget(body, 220)

    if not isinstance(model_id, str) or model_id not in DEBATE_MODELS:
        raise fastapi.HTTPException(
            status_code=400,
            detail=f"model {model_id!r} not allowed. Allowed: {sorted(DEBATE_MODELS)}",
        )

    server = DebateInferenceServer(model_id=model_id)
    return server.generate.remote(prompt, max_new_tokens)


@app.function(secrets=[modal.Secret.from_name("quantsafe-auth")])
@modal.fastapi_endpoint(method="POST", label="judge")
def judge_endpoint(
    body: dict[str, Any],
    authorization: str = fastapi.Header(default=""),
) -> dict[str, str]:
    """HTTP POST handler for safety-judge classification.

    Request JSON:
        {
            "model":          "<hf_model_id>",   # must be in JUDGE_MODELS
            "prompt":         "<user prompt being judged>",
            "response":       "<assistant response being judged>",
            "max_new_tokens": 48                 # optional
        }

    Response JSON (HTTP 200):
        {"text": "<raw judge completion>", "quantization": "<actual precision>"}

    The raw completion is returned untouched; verdict parsing lives client-side
    in judges.py (parse_qwen3guard / parse_granite_guardian) so the parsing
    logic stays unit-testable without a GPU.

    Errors mirror /generate: 401 (auth), 400 (unknown judge model / bad input).
    """
    _require_bearer_auth(authorization)

    model_id = body.get("model", "")
    prompt = _bounded_text("prompt", body.get("prompt", ""))
    response = _bounded_text("response", body.get("response", ""))
    max_new_tokens = _token_budget(body, 48)

    if not isinstance(model_id, str) or model_id not in JUDGE_MODELS:
        raise fastapi.HTTPException(
            status_code=400,
            detail=f"judge model {model_id!r} not allowed. Allowed: {sorted(JUDGE_MODELS)}",
        )

    server = DebateInferenceServer(model_id=model_id)
    return server.judge.remote(prompt, response, max_new_tokens)
