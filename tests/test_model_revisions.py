"""Pinned-model revision coverage and format checks."""

from __future__ import annotations

import re

import pytest

from model_revisions import MODEL_REVISIONS, model_revision


EXPECTED_MODELS = {
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen3-8B",
    "microsoft/Phi-4-mini-instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "Qwen/Qwen3Guard-Gen-0.6B",
    "ibm-granite/granite-guardian-3.3-8b",
    "nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3",
    "Crusadersk/quantsafe-refusal-modernbert",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "meta-llama/Llama-3.2-1B-Instruct",
}


def test_all_runtime_models_have_pinned_revisions():
    assert set(MODEL_REVISIONS) == EXPECTED_MODELS


def test_revisions_are_full_commit_shas():
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in MODEL_REVISIONS.values())


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="No pinned revision"):
        model_revision("unknown/model")
