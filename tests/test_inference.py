"""inference.py tests — modal contract, hf chat_completion, cpu cache eviction.

Every test here is offline: requests.post is monkeypatched (no network),
huggingface_hub.InferenceClient is replaced with a fake (no provider calls),
and the cpu loader is stubbed (no model downloads). The token-counting
tokenizer for the modal backend is likewise a fake — no real tokenizer is
ever fetched.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make SPACE root importable regardless of working directory.
_SPACE = Path(__file__).resolve().parent.parent
if str(_SPACE) not in sys.path:
    sys.path.insert(0, str(_SPACE))

import inference
from inference import MAX_CACHED_CPU_MODELS, infer


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code: int, payload=None, text: str = "",
                 json_raises: bool = False):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self._json_raises = json_raises

    def json(self):
        if self._json_raises:
            raise ValueError("response body is not JSON")
        return self._payload


class _FakeCountTokenizer:
    """Counts 'tokens' by whitespace split; asserts completion-only counting."""

    def __call__(self, text: str, add_special_tokens: bool = True):
        # Client-side counts must exclude special tokens (no BOS) to stay
        # comparable with the cpu backend's generated-token counts.
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=text.split())


def _patch_modal_env(monkeypatch, token: str | None = "sekret-token"):
    monkeypatch.setenv("MODAL_ENDPOINT", "https://example--debate-generate.modal.run")
    if token is None:
        monkeypatch.delenv("MODAL_TOKEN", raising=False)
    else:
        monkeypatch.setenv("MODAL_TOKEN", token)
    monkeypatch.setattr(inference, "_load_count_tokenizer",
                        lambda mid: _FakeCountTokenizer())


# ---------------------------------------------------------------------------
# (a) modal backend — payload shape, auth header, error contract
# ---------------------------------------------------------------------------

class TestModalBackend:
    def test_payload_shape_and_auth_header(self, monkeypatch):
        _patch_modal_env(monkeypatch)
        calls: list[dict] = []

        def _fake_post(url, json=None, headers=None, timeout=None):
            calls.append({"url": url, "json": json, "headers": headers,
                          "timeout": timeout})
            return _FakeResponse(200, {"text": "a generated completion",
                                       "quantization": "nf4-4bit"})

        monkeypatch.setattr("requests.post", _fake_post)
        completions, counts = infer(
            "Qwen/Qwen2.5-7B-Instruct", ["p1", "p2"],
            backend="modal", max_new_tokens=99,
        )

        # One POST per prompt, exact contract payload, bearer auth, 300 s timeout.
        assert len(calls) == 2
        assert calls[0]["json"] == {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "prompt": "p1",
            "max_new_tokens": 99,
        }
        assert calls[1]["json"]["prompt"] == "p2"
        for c in calls:
            assert c["headers"]["Authorization"] == "Bearer sekret-token"
            assert c["timeout"] == 300  # cold start can exceed 120 s
        assert completions == ["a generated completion"] * 2

    def test_no_modal_token_sends_no_auth_header(self, monkeypatch):
        _patch_modal_env(monkeypatch, token=None)
        seen_headers: list[dict] = []

        def _fake_post(url, json=None, headers=None, timeout=None):
            seen_headers.append(headers)
            return _FakeResponse(200, {"text": "ok", "quantization": "fp16"})

        monkeypatch.setattr("requests.post", _fake_post)
        infer("Qwen/Qwen2.5-1.5B-Instruct", ["p"], backend="modal")
        assert "Authorization" not in seen_headers[0]

    def test_non_2xx_raises_runtime_error_with_detail(self, monkeypatch):
        _patch_modal_env(monkeypatch)
        detail = "missing or invalid Authorization header"

        def _fake_post(url, json=None, headers=None, timeout=None):
            return _FakeResponse(401, {"detail": detail})

        monkeypatch.setattr("requests.post", _fake_post)
        with pytest.raises(RuntimeError, match="missing or invalid Authorization"):
            infer("m", ["p"], backend="modal")

    def test_400_detail_surfaces_in_runtime_error(self, monkeypatch):
        _patch_modal_env(monkeypatch)

        def _fake_post(url, json=None, headers=None, timeout=None):
            return _FakeResponse(400, {"detail": "model 'bad' not allowed"})

        monkeypatch.setattr("requests.post", _fake_post)
        with pytest.raises(RuntimeError, match="not allowed"):
            infer("bad", ["p"], backend="modal")

    def test_non_json_error_body_falls_back_to_text(self, monkeypatch):
        _patch_modal_env(monkeypatch)

        def _fake_post(url, json=None, headers=None, timeout=None):
            return _FakeResponse(502, text="Bad Gateway", json_raises=True)

        monkeypatch.setattr("requests.post", _fake_post)
        with pytest.raises(RuntimeError, match="Bad Gateway"):
            infer("m", ["p"], backend="modal")

    def test_token_counts_are_client_side_and_parallel(self, monkeypatch):
        _patch_modal_env(monkeypatch)
        texts = iter(["one two three", "just one-token", ""])

        def _fake_post(url, json=None, headers=None, timeout=None):
            return _FakeResponse(200, {"text": next(texts), "quantization": "fp16"})

        monkeypatch.setattr("requests.post", _fake_post)
        completions, counts = infer("m", ["a", "b", "c"], backend="modal")
        assert len(counts) == len(completions) == 3
        # Whitespace fake tokenizer: counts mirror the completion text.
        assert counts == [3, 2, 0]

    def test_tokenizer_loaded_once_per_call(self, monkeypatch):
        _patch_modal_env(monkeypatch)
        loads: list[str] = []

        def _counting_loader(model_id):
            loads.append(model_id)
            return _FakeCountTokenizer()

        monkeypatch.setattr(inference, "_load_count_tokenizer", _counting_loader)
        monkeypatch.setattr(
            "requests.post",
            lambda url, json=None, headers=None, timeout=None:
                _FakeResponse(200, {"text": "ok", "quantization": "fp16"}),
        )
        infer("m", ["a", "b", "c"], backend="modal")
        assert loads == ["m"]  # once per infer() call, not per prompt

    def test_missing_endpoint_raises(self, monkeypatch):
        monkeypatch.delenv("MODAL_ENDPOINT", raising=False)
        with pytest.raises(EnvironmentError, match="MODAL_ENDPOINT"):
            infer("m", ["p"], backend="modal")


# ---------------------------------------------------------------------------
# (b) hf backend — chat_completion call shape + usage-based token counts
# ---------------------------------------------------------------------------

def _make_fake_hf_module(record: list[dict], content: str = "hf completion",
                         completion_tokens: int = 7, raise_exc: Exception | None = None):
    """Build a fake huggingface_hub module whose InferenceClient records calls."""

    class _FakeInferenceClient:
        def __init__(self, model=None, token=None):
            record.append({"init": {"model": model, "token": token}})

        def chat_completion(self, messages=None, max_tokens=None):
            record.append({"chat": {"messages": messages, "max_tokens": max_tokens}})
            if raise_exc is not None:
                raise raise_exc
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(completion_tokens=completion_tokens),
            )

    mod = types.ModuleType("huggingface_hub")
    mod.InferenceClient = _FakeInferenceClient
    return mod


class TestHfBackend:
    def test_chat_completion_call_shape(self, monkeypatch):
        record: list[dict] = []
        monkeypatch.setitem(sys.modules, "huggingface_hub",
                            _make_fake_hf_module(record))
        monkeypatch.setenv("HF_TOKEN", "hf_test")

        completions, counts = infer("m", ["hello"], backend="hf", max_new_tokens=33)

        chat = next(r["chat"] for r in record if "chat" in r)
        assert chat["messages"] == [{"role": "user", "content": "hello"}]
        assert chat["max_tokens"] == 33
        assert completions == ["hf completion"]
        # Token counts come from response.usage.completion_tokens, NOT whitespace.
        assert counts == [7]

    def test_counts_parallel_to_completions(self, monkeypatch):
        record: list[dict] = []
        monkeypatch.setitem(sys.modules, "huggingface_hub",
                            _make_fake_hf_module(record, completion_tokens=11))
        completions, counts = infer("m", ["a", "b", "c"], backend="hf")
        assert len(counts) == len(completions) == 3
        assert counts == [11, 11, 11]

    def test_provider_failure_raises_clean_runtime_error(self, monkeypatch):
        record: list[dict] = []
        monkeypatch.setitem(
            sys.modules, "huggingface_hub",
            _make_fake_hf_module(record, raise_exc=ValueError("provider exploded")),
        )
        with pytest.raises(RuntimeError, match="hf backend") as excinfo:
            infer("m", ["p"], backend="hf")
        # The original provider error must survive into the message.
        assert "provider exploded" in str(excinfo.value)


# ---------------------------------------------------------------------------
# (c) cpu backend — bounded LRU cache (mocked loader; no downloads)
# ---------------------------------------------------------------------------

class TestCpuCacheEviction:
    @pytest.fixture(autouse=True)
    def _fresh_cache(self, monkeypatch):
        monkeypatch.setattr(inference, "_cpu_cache", {})
        self.loads: list[str] = []
        monkeypatch.setattr(
            inference, "_load_cpu_model",
            lambda mid: (self.loads.append(mid) or (f"tok-{mid}", f"mdl-{mid}")),
        )

    def test_pair_constant(self):
        # The live screen loads exactly (baseline, candidate) — pin the bound.
        assert MAX_CACHED_CPU_MODELS == 2

    def test_cache_never_exceeds_pair(self):
        for mid in ["a", "b", "c", "d"]:
            inference._load_cpu(mid)
            assert len(inference._cpu_cache) <= MAX_CACHED_CPU_MODELS
        # Only the CURRENT pair survives.
        assert set(inference._cpu_cache) == {"c", "d"}

    def test_cache_hit_does_not_reload(self):
        inference._load_cpu("a")
        inference._load_cpu("b")
        tok, mdl = inference._load_cpu("a")
        assert self.loads == ["a", "b"]
        assert (tok, mdl) == ("tok-a", "mdl-a")

    def test_lru_refresh_protects_current_pair(self):
        inference._load_cpu("a")
        inference._load_cpu("b")
        inference._load_cpu("a")  # refresh: 'a' is now most-recent
        inference._load_cpu("c")  # must evict 'b', not 'a'
        assert set(inference._cpu_cache) == {"a", "c"}

    def test_returned_pair_matches_loader(self):
        tok, mdl = inference._load_cpu("x")
        assert tok == "tok-x"
        assert mdl == "mdl-x"


# ---------------------------------------------------------------------------
# (d) public API contract
# ---------------------------------------------------------------------------

class TestInferDispatch:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            infer("m", ["p"], backend="banana")

    def test_backend_is_normalised(self, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(
            inference, "_infer_modal",
            lambda *a: (seen.append("modal") or ([], [])),
        )
        infer("m", [], backend="  Modal ")
        assert seen == ["modal"]
