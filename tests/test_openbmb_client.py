from __future__ import annotations

import pytest

import openbmb_client


class _Response:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_missing_key_fails_before_network(monkeypatch):
    monkeypatch.delenv("OPENBMB_API_KEY", raising=False)
    monkeypatch.setattr(
        openbmb_client.requests,
        "post",
        lambda *_args, **_kwargs: pytest.fail("network should not be called"),
    )
    with pytest.raises(EnvironmentError, match="OPENBMB_API_KEY"):
        openbmb_client.chat([{"role": "user", "content": "test"}])


def test_chat_uses_secret_without_returning_it(monkeypatch):
    monkeypatch.setenv("OPENBMB_API_KEY", "private-test-key")
    calls = {}

    def fake_post(url, *, headers, json, timeout):
        calls.update(url=url, headers=headers, json=json, timeout=timeout)
        return _Response(
            200,
            {
                "model": "MiniCPM4.1-8B",
                "system_fingerprint": "test-fingerprint",
                "choices": [{"message": {"content": "STANCE: ROUTE"}}],
            },
        )

    monkeypatch.setattr(openbmb_client.requests, "post", fake_post)
    result = openbmb_client.chat([{"role": "user", "content": "test"}])

    assert result["text"] == "STANCE: ROUTE"
    assert calls["headers"]["Authorization"] == "Bearer private-test-key"
    assert calls["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "private-test-key" not in str(result)


def test_batch_chat_preserves_choice_order(monkeypatch):
    monkeypatch.setenv("OPENBMB_API_KEY", "private-test-key")

    def fake_post(_url, *, headers, json, timeout):
        assert headers["Authorization"].startswith("Bearer ")
        assert len(json["messages"]) == 2
        assert timeout == openbmb_client.DEFAULT_TIMEOUT_S
        return _Response(
            200,
            {
                "choices": [
                    {"index": 1, "message": {"content": "unsafe"}},
                    {"index": 0, "message": {"content": "safe"}},
                ]
            },
        )

    monkeypatch.setattr(openbmb_client.requests, "post", fake_post)
    out = openbmb_client.batch_chat(
        [
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ]
    )
    assert out == ["safe", "unsafe"]


def test_batch_chat_rejects_duplicate_indices(monkeypatch):
    monkeypatch.setenv("OPENBMB_API_KEY", "private-test-key")
    monkeypatch.setattr(
        openbmb_client.requests,
        "post",
        lambda *_args, **_kwargs: _Response(
            200,
            {
                "choices": [
                    {"index": 0, "message": {"content": "safe"}},
                    {"index": 0, "message": {"content": "unsafe"}},
                ]
            },
        ),
    )
    with pytest.raises(RuntimeError, match="duplicate or missing"):
        openbmb_client.batch_chat(
            [
                [{"role": "user", "content": "one"}],
                [{"role": "user", "content": "two"}],
            ]
        )


def test_non_2xx_error_does_not_expose_key(monkeypatch):
    monkeypatch.setenv("OPENBMB_API_KEY", "private-test-key")
    monkeypatch.setattr(
        openbmb_client.requests,
        "post",
        lambda *_args, **_kwargs: _Response(
            503,
            {"detail": "temporarily down: private-test-key"},
        ),
    )
    with pytest.raises(RuntimeError, match="temporarily down") as exc:
        openbmb_client.chat([{"role": "user", "content": "test"}])
    assert "private-test-key" not in str(exc.value)
    assert "[redacted]" in str(exc.value)
