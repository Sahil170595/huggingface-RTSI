"""inference.py — backend-swappable text generation for the live refusal-drift tab.

Three backends:
  "cpu"    transformers AutoModelForCausalLM on CPU (default, no ext deps at import time)
  "hf"     huggingface_hub InferenceClient (requires HF_TOKEN env var)
  "modal"  HTTP POST to a Modal GPU endpoint (requires MODAL_ENDPOINT env var)

Usage:
    from inference import infer
    completions, token_counts = infer(model_id, prompts, backend="cpu")
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# CPU backend — lazy-load cache
# ---------------------------------------------------------------------------

_cpu_cache: dict[str, tuple] = {}  # model_id -> (tokenizer, model)


def _load_cpu(model_id: str):
    if model_id in _cpu_cache:
        return _cpu_cache[model_id]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import
    import torch
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    mdl.eval()
    _cpu_cache[model_id] = (tok, mdl)
    return tok, mdl


def _infer_cpu(
    model_id: str,
    prompts: list[str],
    max_new_tokens: int,
) -> tuple[list[str], list[int]]:
    import torch
    tok, mdl = _load_cpu(model_id)
    completions: list[str] = []
    token_counts: list[int] = []
    for prompt in prompts:
        # Apply chat template when available so instruct models behave correctly.
        if getattr(tok, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            enc_text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            enc_text = prompt
        input_ids = tok(enc_text, return_tensors="pt").input_ids
        prompt_len = input_ids.shape[-1]
        with torch.no_grad():
            out_ids = mdl.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        # Strip the prompt tokens; decode only the generated portion.
        gen_ids = out_ids[0, prompt_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True)
        completions.append(text)
        token_counts.append(int(gen_ids.shape[-1]))
    return completions, token_counts


# ---------------------------------------------------------------------------
# HF Inference API backend
# ---------------------------------------------------------------------------

def _infer_hf(
    model_id: str,
    prompts: list[str],
    max_new_tokens: int,
) -> tuple[list[str], list[int]]:
    try:
        from huggingface_hub import InferenceClient  # lazy import
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for backend='hf'. "
            "Install it with: pip install huggingface_hub"
        ) from exc
    token = os.environ.get("HF_TOKEN")
    client = InferenceClient(model=model_id, token=token)
    completions: list[str] = []
    token_counts: list[int] = []
    for prompt in prompts:
        result = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
        )
        text = result if isinstance(result, str) else getattr(result, "generated_text", str(result))
        completions.append(text)
        # HF Inference API may not return token counts; approximate via whitespace split.
        token_counts.append(len(text.split()))
    return completions, token_counts


# ---------------------------------------------------------------------------
# Modal GPU endpoint backend
# ---------------------------------------------------------------------------

def _infer_modal(
    model_id: str,
    prompts: list[str],
    max_new_tokens: int,
) -> tuple[list[str], list[int]]:
    endpoint = os.environ.get("MODAL_ENDPOINT")
    if not endpoint:
        raise EnvironmentError(
            "MODAL_ENDPOINT env var is not set. "
            "Deploy modal_app.py first and set MODAL_ENDPOINT to the printed URL."
        )
    try:
        import requests  # lazy import
    except ImportError as exc:
        raise ImportError(
            "requests is required for backend='modal'. "
            "Install it with: pip install requests"
        ) from exc
    payload = {"prompts": prompts, "max_new_tokens": max_new_tokens}
    resp = requests.post(endpoint, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["completions"], data["token_counts"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer(
    model_id: str,
    prompts: list[str],
    backend: str = "cpu",
    max_new_tokens: int = 64,
) -> tuple[list[str], list[int]]:
    """Run inference for a batch of prompts.

    Args:
        model_id:       HF model identifier, e.g. "Qwen/Qwen2.5-1.5B-Instruct".
        prompts:        List of raw user-turn strings.
        backend:        "cpu" | "hf" | "modal".
        max_new_tokens: Generation budget per prompt.

    Returns:
        (completions, token_counts) — parallel lists, one entry per prompt.
    """
    backend = backend.lower().strip()
    if backend == "cpu":
        return _infer_cpu(model_id, prompts, max_new_tokens)
    if backend == "hf":
        return _infer_hf(model_id, prompts, max_new_tokens)
    if backend == "modal":
        return _infer_modal(model_id, prompts, max_new_tokens)
    raise ValueError(
        f"Unknown backend {backend!r}. Choose 'cpu', 'hf', or 'modal'."
    )
