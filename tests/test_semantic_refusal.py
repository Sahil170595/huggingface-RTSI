"""Tests for the fine-tuned semantic refusal cross-check."""

from __future__ import annotations

import pytest

import semantic_refusal


def test_format_exchange_matches_training_template():
    assert semantic_refusal.format_exchange("hello", "world") == (
        "[USER]\nhello\n\n[ASSISTANT]\nworld"
    )


def test_classify_refusals_aggregates_probabilities(monkeypatch):
    captured = {}

    def fake_predict(texts):
        captured["texts"] = texts
        return [0.9, 0.2, 0.5]

    monkeypatch.setattr(
        semantic_refusal,
        "_predict_refusal_probabilities",
        fake_predict,
    )
    result = semantic_refusal.classify_refusals(
        ["p1", "p2", "p3"],
        ["r1", "r2", "r3"],
    )

    assert captured["texts"][0] == "[USER]\np1\n\n[ASSISTANT]\nr1"
    assert result["n_items"] == 3
    assert result["n_refusals"] == 2
    assert result["refusal_rate"] == pytest.approx(2 / 3)
    assert [item["is_refusal"] for item in result["items"]] == [True, False, True]


def test_empty_input_does_not_load_model(monkeypatch):
    def fail(_texts):
        raise AssertionError("model should not load")

    monkeypatch.setattr(semantic_refusal, "_predict_refusal_probabilities", fail)
    result = semantic_refusal.classify_refusals([], [])
    assert result["n_items"] == 0
    assert result["refusal_rate"] == 0.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        semantic_refusal.classify_refusals(["prompt"], [])


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        semantic_refusal.classify_refusals([], [], threshold=threshold)
