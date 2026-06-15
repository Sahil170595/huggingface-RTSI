"""Modal model-load policy tests with no Modal, GPU, or model dependencies."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parent.parent


class _Image:
    @classmethod
    def debian_slim(cls, **_kwargs):
        return cls()

    def pip_install(self, *_args, **_kwargs):
        return self

    def add_local_python_source(self, *_args, **_kwargs):
        return self


class _App:
    def __init__(self, *_args, **_kwargs):
        pass

    def cls(self, *_args, **_kwargs):
        return lambda obj: obj

    def function(self, *_args, **_kwargs):
        return lambda obj: obj


class _Volume:
    @classmethod
    def from_name(cls, *_args, **_kwargs):
        return cls()


class _Secret:
    @classmethod
    def from_name(cls, *_args, **_kwargs):
        return cls()


class _HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _decorator(*_args, **_kwargs):
    return lambda obj: obj


@pytest.fixture()
def modal_app_module(monkeypatch):
    fake_modal = types.ModuleType("modal")
    fake_modal.Image = _Image
    fake_modal.App = _App
    fake_modal.Volume = _Volume
    fake_modal.Secret = _Secret
    fake_modal.parameter = lambda *, default: default
    fake_modal.enter = _decorator
    fake_modal.method = _decorator
    fake_modal.fastapi_endpoint = _decorator

    fake_fastapi = types.ModuleType("fastapi")
    fake_fastapi.Header = lambda *, default="": default
    fake_fastapi.HTTPException = _HTTPException

    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)

    module_name = "_quantsafe_modal_policy_test"
    spec = importlib.util.spec_from_file_location(module_name, _ROOT / "modal_app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _DType:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return f"torch.{self.name}"


def _install_fake_load_stack(monkeypatch, *, force_dtype=None):
    calls: dict[str, object] = {}
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = _DType("float16")
    fake_torch.bfloat16 = _DType("bfloat16")

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["tokenizer"] = {"model_id": model_id, **kwargs}
            return object()

    class _Model:
        def __init__(self, dtype, quantization_config):
            self.dtype = force_dtype if force_dtype is not None else dtype
            self.is_loaded_in_4bit = quantization_config is not None
            self.eval_called = False

        def eval(self):
            self.eval_called = True

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model"] = {"model_id": model_id, **kwargs}
            model = _Model(kwargs["dtype"], kwargs["quantization_config"])
            calls["loaded_model"] = model
            return model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = AutoModelForCausalLM
    fake_transformers.AutoTokenizer = AutoTokenizer
    fake_transformers.BitsAndBytesConfig = BitsAndBytesConfig

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return fake_torch, calls


def test_every_allowed_model_has_an_explicit_policy(modal_app_module):
    assert set(modal_app_module.MODEL_LOAD_POLICIES) == modal_app_module.ALLOWED_MODELS


def test_existing_debate_model_policies_are_preserved(modal_app_module):
    policies = modal_app_module.MODEL_LOAD_POLICIES
    assert policies["Qwen/Qwen2.5-7B-Instruct"]["precision"] == "nf4-4bit"
    assert policies["mistralai/Mistral-7B-Instruct-v0.3"]["precision"] == "nf4-4bit"

    fp16_models = modal_app_module.DEBATE_MODELS - {
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    }
    assert all(policies[model_id]["precision"] == "fp16" for model_id in fp16_models)


def test_nemotron_loads_and_reports_bfloat16(
    modal_app_module, monkeypatch
):
    fake_torch, calls = _install_fake_load_stack(monkeypatch)
    server = modal_app_module.DebateInferenceServer()
    server.model_id = "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3"

    server.load()

    model_call = calls["model"]
    assert model_call["dtype"] is fake_torch.bfloat16
    assert model_call["quantization_config"] is None
    assert server.precision == "bf16"
    assert calls["loaded_model"].eval_called is True


def test_legacy_7b_policy_builds_nf4_config(modal_app_module, monkeypatch):
    fake_torch, calls = _install_fake_load_stack(monkeypatch)
    server = modal_app_module.DebateInferenceServer()
    server.model_id = "Qwen/Qwen2.5-7B-Instruct"

    server.load()

    model_call = calls["model"]
    quantization_config = model_call["quantization_config"]
    assert model_call["dtype"] is fake_torch.float16
    assert quantization_config.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": fake_torch.float16,
    }
    assert server.precision == "nf4-4bit"


def test_unexpected_loaded_dtype_fails_closed(modal_app_module, monkeypatch):
    fake_torch, _calls = _install_fake_load_stack(
        monkeypatch, force_dtype=_DType("float16")
    )
    server = modal_app_module.DebateInferenceServer()
    server.model_id = "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3"

    with pytest.raises(RuntimeError, match="loaded as fp16, expected bf16"):
        server.load()

    assert str(fake_torch.bfloat16) == "torch.bfloat16"


def test_judge_endpoint_forwards_worker_reported_precision(
    modal_app_module, monkeypatch
):
    payload = {"text": '{"Response Safety": "safe"}', "quantization": "bf16"}

    class _RemoteJudge:
        @staticmethod
        def remote(*_args):
            return payload

    class _Server:
        judge = _RemoteJudge()

        def __init__(self, *, model_id):
            assert model_id == "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3"

    monkeypatch.setattr(modal_app_module, "DebateInferenceServer", _Server)
    monkeypatch.setenv("QUANTSAFE_MODAL_TOKEN", "test-token")

    result = modal_app_module.judge_endpoint(
        {
            "model": "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3",
            "prompt": "prompt",
            "response": "response",
        },
        authorization="Bearer test-token",
    )

    assert result is payload
