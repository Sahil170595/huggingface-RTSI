"""inference.py — backend-swappable text generation for the live refusal-drift tab.

Four backends:
  "zerogpu" transformers AutoModelForCausalLM on a @spaces.GPU allocation
  "cpu"    transformers AutoModelForCausalLM on CPU (default, no ext deps at import time)
  "hf"     huggingface_hub InferenceClient.chat_completion (set HF_TOKEN for
           provider routing / rate limits)
  "modal"  HTTP POST to a Modal GPU endpoint (requires MODAL_ENDPOINT; sends
           "Authorization: Bearer <MODAL_TOKEN>" when MODAL_TOKEN is set)

Modal endpoint contract (mirrors modal_app.py):
  Request  JSON: {"model": "<hf_model_id>", "prompt": "<text>", "max_new_tokens": N}
  Response JSON: {"text": "<completion>", "quantization": "<precision>"}
  Errors:  non-2xx with a FastAPI {"detail": "<message>"} body -> RuntimeError here.

Usage:
    from inference import infer
    completions, token_counts = infer(model_id, prompts, backend="cpu")
"""

from __future__ import annotations

import os
import threading

from model_revisions import model_revision

# ---------------------------------------------------------------------------
# CPU backend — lazy-load LRU cache, bounded so fp32 weights can't OOM the
# 16 GB CPU Basic Space. At most the CURRENT run's (baseline, candidate) pair
# stays resident; older models are deleted and garbage-collected first.
# ---------------------------------------------------------------------------

#: Maximum number of CPU models held in memory at once. A live screen loads
#: exactly two (baseline + candidate); keep the pair, evict everything older.
MAX_CACHED_CPU_MODELS: int = 2

_cpu_cache: dict[str, tuple] = {}  # model_id -> (tokenizer, model); insertion order == LRU order
_cpu_cache_lock = threading.Lock()


def _encode_prompt(tokenizer, model_id: str, prompt: str):
    """Tokenize one instruct prompt without double-inserting special tokens."""
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        template_kwargs: dict = {}
        mid = model_id.lower()
        if "qwen3" in mid and "guard" not in mid:
            template_kwargs["enable_thinking"] = False
        if "smollm3" in mid:
            messages = [{"role": "system", "content": "/no_think"}] + messages
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **template_kwargs,
        )
    return tokenizer(prompt, return_tensors="pt")


def _load_cpu_model(model_id: str) -> tuple:
    """Actually download + instantiate (tokenizer, model) on CPU.

    Split out of :func:`_load_cpu` so tests can stub the heavyweight load
    while still exercising the real cache/eviction logic.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import
    import torch
    revision = model_revision(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        # Keep float32 on CPU for the 1-1.5B live models: it is the numerically
        # safe default and fits comfortably once the cache is bounded. Do NOT
        # switch dtype silently — drift numbers must stay comparable.
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    mdl.eval()
    return tok, mdl


def _load_cpu(model_id: str) -> tuple:
    """Return a cached (tokenizer, model) pair, loading + evicting as needed.

    Thread-safe: Gradio can fire concurrent live runs, and loading the same
    fp32 model twice in parallel would double peak memory.
    """
    import gc
    with _cpu_cache_lock:
        if model_id in _cpu_cache:
            # Refresh LRU position so the current run's pair survives eviction.
            _cpu_cache[model_id] = _cpu_cache.pop(model_id)
            return _cpu_cache[model_id]
        # Evict oldest entries BEFORE loading so peak residency never exceeds
        # MAX_CACHED_CPU_MODELS models.
        evicted = False
        while len(_cpu_cache) >= MAX_CACHED_CPU_MODELS:
            oldest_id = next(iter(_cpu_cache))
            del _cpu_cache[oldest_id]
            evicted = True
        if evicted:
            gc.collect()  # release evicted fp32 weights before the next download
        tok, mdl = _load_cpu_model(model_id)
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
        enc = _encode_prompt(tok, model_id, prompt)
        prompt_len = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out_ids = mdl.generate(
                **enc,  # input_ids + attention_mask
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
# ZeroGPU backend — called only from app.py's single @spaces.GPU allocation.
# ---------------------------------------------------------------------------

MAX_CACHED_GPU_MODELS: int = 2
_gpu_cache: dict[str, tuple] = {}
_gpu_cache_lock = threading.Lock()


def _load_gpu_model(model_id: str) -> tuple:
    """Load a pinned small model in fp16 on the active ZeroGPU CUDA device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = model_revision(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    return tokenizer, model


def _load_gpu(model_id: str) -> tuple:
    """Return a bounded GPU model cache inside ZeroGPU's CUDA emulation."""
    import gc
    import torch

    with _gpu_cache_lock:
        if model_id in _gpu_cache:
            _gpu_cache[model_id] = _gpu_cache.pop(model_id)
            return _gpu_cache[model_id]
        while len(_gpu_cache) >= MAX_CACHED_GPU_MODELS:
            oldest_id = next(iter(_gpu_cache))
            del _gpu_cache[oldest_id]
            gc.collect()
            torch.cuda.empty_cache()
        bundle = _load_gpu_model(model_id)
        _gpu_cache[model_id] = bundle
        return bundle


def _infer_zerogpu(
    model_id: str,
    prompts: list[str],
    max_new_tokens: int,
) -> tuple[list[str], list[int]]:
    """Generate one tensor batch while a real ZeroGPU allocation is held."""
    import torch

    tokenizer, model = _load_gpu(model_id)
    if not prompts:
        return [], []

    # Left padding keeps each completion aligned after the shared input width,
    # allowing all probes to decode in one GPU batch instead of one at a time.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = []
    for prompt in prompts:
        encoded = _encode_prompt(tokenizer, model_id, prompt)
        rows.append({key: value[0] for key, value in encoded.items()})
    encoded_batch = tokenizer.pad(rows, padding=True, return_tensors="pt")
    encoded_batch = {key: value.to("cuda") for key, value in encoded_batch.items()}
    prompt_width = encoded_batch["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded_batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[:, prompt_width:].detach().cpu()
    completions = tokenizer.batch_decode(generated, skip_special_tokens=True)
    token_counts = [
        int(row.ne(tokenizer.pad_token_id).sum().item())
        for row in generated
    ]
    return completions, token_counts


def infer_zerogpu_pair(
    baseline_model: str,
    candidate_model: str,
    prompts: list[str],
    max_new_tokens: int = 64,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Run both sides under the caller's single @spaces.GPU allocation."""
    base_completions, base_counts = _infer_zerogpu(
        baseline_model, prompts, max_new_tokens
    )
    candidate_completions, candidate_counts = _infer_zerogpu(
        candidate_model, prompts, max_new_tokens
    )
    return (
        base_completions,
        base_counts,
        candidate_completions,
        candidate_counts,
    )


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
        # chat_completion applies the model's chat template server-side and
        # reports real token usage — no hand-rolled prompts, no whitespace
        # "token" counting.
        try:
            result = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
            )
            text = result.choices[0].message.content or ""
            count = int(result.usage.completion_tokens)
        except Exception as exc:
            raise RuntimeError(
                f"hf backend: provider call failed for {model_id!r}: {exc}. "
                "Check that HF_TOKEN is set and the model is served by an "
                "inference provider."
            ) from exc
        completions.append(text)
        token_counts.append(count)
    return completions, token_counts


# ---------------------------------------------------------------------------
# Modal GPU endpoint backend
# ---------------------------------------------------------------------------

#: Per-request timeout. A Modal cold start (download + load a 7B model) can
#: exceed 120 s; 300 s matches the endpoint's own timeout budget.
_MODAL_TIMEOUT_S: int = 300

_count_tok_cache: dict[str, object] = {}  # model_id -> tokenizer (counting only)


def _load_count_tokenizer(model_id: str):
    """Load (and cache) the model's tokenizer for client-side token counting.

    Split out of :func:`_infer_modal` so tests can stub it without ever
    downloading a real tokenizer.
    """
    if model_id in _count_tok_cache:
        return _count_tok_cache[model_id]
    from transformers import AutoTokenizer  # lazy import
    tok = AutoTokenizer.from_pretrained(model_id, revision=model_revision(model_id))
    _count_tok_cache[model_id] = tok
    return tok


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
    headers: dict[str, str] = {}
    token = os.environ.get("MODAL_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    # The endpoint returns text only; count tokens client-side with the model's
    # own tokenizer (loaded once per call) so counts stay comparable with the
    # cpu backend's generated-token counts.
    tok = _load_count_tokenizer(model_id)
    completions: list[str] = []
    token_counts: list[int] = []
    for prompt in prompts:
        payload = {
            "model": model_id,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
        resp = requests.post(
            endpoint, json=payload, headers=headers, timeout=_MODAL_TIMEOUT_S,
        )
        if not 200 <= resp.status_code < 300:
            # FastAPI errors arrive as {"detail": "<message>"}; surface that
            # message verbatim so the UI shows a clean error.
            try:
                detail = resp.json().get("detail") or resp.text
            except Exception:
                detail = resp.text
            raise RuntimeError(
                f"modal backend error (HTTP {resp.status_code}): {detail}"
            )
        data = resp.json()
        text = str(data["text"])
        completions.append(text)
        token_counts.append(len(tok(text, add_special_tokens=False).input_ids))
    return completions, token_counts


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
