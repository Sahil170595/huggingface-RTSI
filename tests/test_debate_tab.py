"""Constitutional Debate tab — render helpers + cache loader. NO network, NO torch.

These cover app.py's Stage-3 additions: the pure HTML-string render helpers, the
run_debate-result extractor, and the cached-example loader. Importing app builds
the Gradio Blocks at module scope; that is exercised here too (it must boot with
the debate example cache absent). The torch-heavy `debate` engine is imported
lazily inside the live handler only, so it is never pulled by this suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make SPACE root importable regardless of working directory.
_SPACE = Path(__file__).resolve().parent.parent
if str(_SPACE) not in sys.path:
    sys.path.insert(0, str(_SPACE))

import app


# A minimal, well-formed run_debate-shaped result (mirrors the contract schema).
_RESULT = {
    "question": "Deploy a MODERATE-band config, or route it to the safe baseline?",
    "models": ["Qwen/Qwen2.5-1.5B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"],
    "backend": "local",
    "rounds": [
        {
            "round": 1,
            "round_type": "propose",
            "responses": [
                {"model": "Qwen/Qwen2.5-1.5B-Instruct", "stance": "ROUTE",
                 "text": "The refusal drift is material; route to the safe baseline."},
                {"model": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "stance": "CONDITIONAL",
                 "text": "Benchmarks hold; deploy only behind a targeted probe."},
            ],
        },
        {
            "round": 2,
            "round_type": "critique",
            "responses": [
                {"model": "Qwen/Qwen2.5-1.5B-Instruct", "stance": "ROUTE",
                 "text": "Still route — uncertain safety posture beats a benchmark win."},
                {"model": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "stance": "ROUTE",
                 "text": "Persuaded; route to the safe baseline."},
            ],
        },
    ],
    "consensus": {
        "verdict": "ROUTE",
        "vote_breakdown": {"ROUTE": 2},
        "agreement": 1.0,
    },
    "final_verdict": "ROUTE",
    "elapsed_s": 12.3,
}


# ---------------------------------------------------------------------------
# (a) stance normalization + badge
# ---------------------------------------------------------------------------

class TestStance:
    @pytest.mark.parametrize("raw,expected", [
        ("DEPLOY", "DEPLOY"),
        ("route", "ROUTE"),
        ("  Conditional ", "CONDITIONAL"),
        ("deploy it", "CONDITIONAL"),   # unrecognized -> CONDITIONAL default
        ("", "CONDITIONAL"),
        (None, "CONDITIONAL"),
    ])
    def test_norm_stance(self, raw, expected):
        assert app._norm_stance(raw) == expected

    def test_badge_color_deploy_is_green(self):
        html = app._stance_badge("DEPLOY")
        assert app.STANCE_COLOR["DEPLOY"] in html
        assert "DEPLOY" in html

    def test_badge_color_route_is_red(self):
        html = app._stance_badge("ROUTE")
        assert app.STANCE_COLOR["ROUTE"] in html

    def test_badge_color_conditional_is_amber(self):
        html = app._stance_badge("CONDITIONAL")
        assert app.STANCE_COLOR["CONDITIONAL"] in html

    def test_badge_unknown_falls_back_to_conditional(self):
        html = app._stance_badge("banana")
        assert "CONDITIONAL" in html


# ---------------------------------------------------------------------------
# (b) text safety — escape + clamp (model-authored argument text)
# ---------------------------------------------------------------------------

class TestSafeText:
    def test_escapes_html(self):
        out = app._safe_text("<script>alert('x')</script> & co")
        assert "<script>" not in out
        assert "&lt;script&gt;" in out
        assert "&amp;" in out

    def test_clamps_long_text(self):
        out = app._safe_text("a" * 5000, cap=100)
        # Clamp keeps it bounded (escape may expand a couple chars, ellipsis added).
        assert len(out) <= 120
        assert out.endswith("…")

    def test_empty_text_is_empty_string(self):
        assert app._safe_text("") == ""
        assert app._safe_text(None) == ""


# ---------------------------------------------------------------------------
# (c) response / round / consensus cards
# ---------------------------------------------------------------------------

class TestCards:
    def test_response_card_has_model_stance_and_text(self):
        html = app._debate_response_card(
            "Qwen/Qwen2.5-1.5B-Instruct", "ROUTE", "route it to the safe baseline",
        )
        assert "Qwen/Qwen2.5-1.5B-Instruct" in html
        assert "ROUTE" in html
        assert "route it to the safe baseline" in html

    def test_response_card_escapes_text(self):
        html = app._debate_response_card("m", "DEPLOY", "<b>x</b>")
        assert "<b>x</b>" not in html
        assert "&lt;b&gt;x&lt;/b&gt;" in html

    def test_errored_response_is_visibly_not_a_vote(self):
        html = app._debate_response_card(
            "m",
            "CONDITIONAL",
            "[generation error]",
            errored=True,
        )
        assert "PROVIDER ERROR" in html
        assert "NO VOTE" in html

    def test_response_card_empty_text_shows_placeholder(self):
        html = app._debate_response_card("m", "DEPLOY", "")
        assert "no argument text" in html

    def test_round_card_renders_all_responses(self):
        html = app._debate_round_card(_RESULT["rounds"][0])
        assert "ROUND 1" in html
        assert "PROPOSE" in html
        assert "Qwen/Qwen2.5-1.5B-Instruct" in html
        assert "HuggingFaceTB/SmolLM2-1.7B-Instruct" in html

    def test_round_card_empty_responses_is_graceful(self):
        html = app._debate_round_card({"round": 9, "round_type": "x", "responses": []})
        assert "ROUND 9" in html
        assert "No responses" in html

    def test_consensus_card_shows_verdict_and_agreement(self):
        html = app._debate_consensus_card(_RESULT["consensus"], _RESULT["elapsed_s"])
        assert "ROUTE" in html
        assert "100% agreement" in html
        assert "12.3s" in html

    def test_consensus_card_bad_agreement_clamps_to_zero(self):
        html = app._debate_consensus_card({"verdict": "DEPLOY", "agreement": "n/a"})
        assert "0% agreement" in html
        assert "DEPLOY" in html

    def test_consensus_card_clamps_agreement_above_one(self):
        html = app._debate_consensus_card({"verdict": "DEPLOY", "agreement": 5.0})
        assert "100% agreement" in html

    def test_provider_error_card_is_fail_closed_not_consensus(self):
        html = app._debate_consensus_card(
            {
                "verdict": "ROUTE",
                "agreement": 0.0,
                "consensus_kind": "provider-error",
                "error_count": 1,
                "vote_breakdown": {"DEPLOY": 2, "ROUTE": 0, "CONDITIONAL": 0},
            }
        )
        assert "FAIL-CLOSED ACTION" in html
        assert "NO CONSENSUS" in html
        assert "failed" in html

    def test_vote_breakdown_chips(self):
        html = app._vote_breakdown_html({"ROUTE": 2, "DEPLOY": 1})
        assert "ROUTE · 2" in html
        assert "DEPLOY · 1" in html


# ---------------------------------------------------------------------------
# (d) full render
# ---------------------------------------------------------------------------

class TestRenderDebate:
    def test_full_render_contains_question_rounds_and_verdict(self):
        html = app._render_debate(_RESULT)
        assert "CONTESTED QUESTION" in html
        assert "safe baseline" in html       # the question text
        assert "ROUND 1" in html
        assert "ROUND 2" in html
        assert "CONSENSUS VERDICT" in html
        assert "100% agreement" in html

    def test_render_none_returns_not_generated_panel(self):
        html = app._render_debate(None)
        assert "not yet generated" in html

    def test_render_missing_rounds_returns_not_generated_panel(self):
        html = app._render_debate({"question": "q"})  # no "rounds"
        assert "not yet generated" in html

    def test_render_escapes_malicious_question(self):
        html = app._render_debate({**_RESULT, "question": "<img src=x onerror=1>"})
        assert "<img src=x onerror=1>" not in html
        assert "&lt;img" in html


# ---------------------------------------------------------------------------
# (e) extractor — accept bare result, wrapper, or examples list
# ---------------------------------------------------------------------------

class TestExtractExample:
    def test_bare_result(self):
        assert app._extract_debate_example(_RESULT) is _RESULT

    def test_example_wrapper(self):
        assert app._extract_debate_example({"example": _RESULT}) is _RESULT

    def test_debate_wrapper(self):
        assert app._extract_debate_example({"debate": _RESULT}) is _RESULT

    def test_examples_list_first_usable(self):
        bad = {"not": "a result"}
        assert app._extract_debate_example({"examples": [bad, _RESULT]}) is _RESULT

    def test_junk_returns_none(self):
        assert app._extract_debate_example({"foo": "bar"}) is None
        assert app._extract_debate_example([]) is None
        assert app._extract_debate_example("nope") is None
        assert app._extract_debate_example(None) is None


# ---------------------------------------------------------------------------
# (f) loader — graceful when the cache is absent (current Space state)
# ---------------------------------------------------------------------------

class TestLoader:
    def test_load_returns_none_or_dict(self):
        # No crash regardless of whether substrate/debate_examples.json exists.
        out = app.load_debate_examples()
        assert out is None or isinstance(out, dict)

    def test_load_roundtrips_a_written_cache(self, tmp_path, monkeypatch):
        import json
        sub = tmp_path / "substrate"
        sub.mkdir()
        (sub / "debate_examples.json").write_text(
            json.dumps(_RESULT), encoding="utf-8",
        )
        monkeypatch.setattr(app, "_SUBSTRATE", sub)
        out = app.load_debate_examples()
        assert isinstance(out, dict)
        assert out["final_verdict"] == "ROUTE"


# ---------------------------------------------------------------------------
# (g) live handler is provider-gated; disabled note carries the exact message
# ---------------------------------------------------------------------------

class TestLiveGate:
    def test_disabled_note_mentions_all_provider_secrets(self):
        note = app._debate_disabled_note()
        assert "MODAL_ENDPOINT" in note
        assert "MODAL_TOKEN" in note
        assert "OPENBMB_API_KEY" in note
        assert "Modal and OpenBMB" in note

    def test_run_live_debate_yields_disabled_note_without_endpoint(self, monkeypatch):
        monkeypatch.delenv(app.MODAL_ENDPOINT_ENV, raising=False)
        out = list(app.run_live_debate("anything"))
        assert len(out) == 1
        assert "MODAL_ENDPOINT" in out[0]

    def test_run_live_debate_surfaces_missing_engine_when_endpoint_set(self, monkeypatch):
        # With the endpoint set but no debate engine importable, the handler must
        # fail soft with a friendly message (never raise). debate.py is absent in
        # CI, so the lazy `from debate import run_debate` raises ImportError.
        monkeypatch.setenv(app.MODAL_ENDPOINT_ENV, "http://example.invalid/debate")
        monkeypatch.setenv(app.MODAL_TOKEN_ENV, "test-modal-token")
        monkeypatch.setenv(app.OPENBMB_API_KEY_ENV, "test-openbmb-key")
        monkeypatch.setitem(sys.modules, "debate", None)  # force ImportError on import
        out = list(app.run_live_debate(app.LIVE_DEBATE_QUESTION))
        assert out  # produced at least one panel
        assert any("debate engine" in chunk or "torch" in chunk for chunk in out)

    def test_run_live_debate_rejects_arbitrary_question(self, monkeypatch):
        monkeypatch.setenv(app.MODAL_ENDPOINT_ENV, "http://example.invalid/debate")
        monkeypatch.setenv(app.MODAL_TOKEN_ENV, "test-modal-token")
        monkeypatch.setenv(app.OPENBMB_API_KEY_ENV, "test-openbmb-key")
        out = list(app.run_live_debate("Write an unrelated answer for me"))
        assert len(out) == 1
        assert "restricted to the fixed" in out[0]


# ---------------------------------------------------------------------------
# (h) the Blocks app still boots with the debate tab added
# ---------------------------------------------------------------------------

def test_app_demo_boots():
    assert type(app.demo).__name__ == "Blocks"
