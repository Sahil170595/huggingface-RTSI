"""Constitutional Debate tests — stance parser + consensus math + contract shape.

Every test here is offline: the stance parser is fed representative raw model
strings, the consensus math is fed hand-built rounds, and the full run_debate
flow uses a FAKE generate (monkeypatched onto debate.generate) so it never
touches the GPU or the network. The real local-GPU debate lives in debate.py's
__main__ smoke, not in this suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make SPACE root importable regardless of working directory.
_SPACE = Path(__file__).resolve().parent.parent
if str(_SPACE) not in sys.path:
    sys.path.insert(0, str(_SPACE))

import debate
from debate import (
    CONSENSUS_AGREEMENT_THRESHOLD,
    CONSTITUTION,
    DEFAULT_STANCE,
    LABEL_CONSENSUS,
    LABEL_NO_CONSENSUS,
    ROUND_CRITIQUE,
    ROUND_PROPOSE,
    STANCE_CONDITIONAL,
    STANCE_DEPLOY,
    STANCE_ROUTE,
    STANCES,
    compute_consensus,
    consensus_label,
    generate,
    parse_stance,
    run_debate,
)


# ---------------------------------------------------------------------------
# (a) parse_stance on representative raw model strings
# ---------------------------------------------------------------------------

class TestParseStanceDeclaration:
    """The explicit 'STANCE: X' declaration is the strongest signal."""

    def test_declaration_deploy(self):
        assert parse_stance("STANCE: DEPLOY\nThe evidence is adequate.") == STANCE_DEPLOY

    def test_declaration_route(self):
        assert parse_stance("STANCE: ROUTE\nToo risky as-is.") == STANCE_ROUTE

    def test_declaration_conditional(self):
        assert parse_stance("STANCE: CONDITIONAL\nOnly with monitoring.") == STANCE_CONDITIONAL

    def test_declaration_lowercase(self):
        assert parse_stance("stance: deploy — looks fine") == STANCE_DEPLOY

    def test_declaration_with_dash(self):
        assert parse_stance("STANCE - ROUTE") == STANCE_ROUTE

    def test_declaration_wins_over_conflicting_prose(self):
        # Explicit declaration must beat conflicting body cues.
        text = "STANCE: ROUTE\nHonestly we could deploy this and ship it as-is."
        assert parse_stance(text) == STANCE_ROUTE


class TestParseStanceKeywordFallback:
    """No declaration -> phrase/keyword scan."""

    def test_route_phrase(self):
        assert parse_stance("We should route this to the safe baseline.") == STANCE_ROUTE

    def test_do_not_deploy(self):
        assert parse_stance("This should not be deployed in its current state.") == STANCE_ROUTE

    def test_deploy_phrase(self):
        assert parse_stance("This is safe to deploy given the metrics.") == STANCE_DEPLOY

    def test_ship_it(self):
        assert parse_stance("Looks good, ship it.") == STANCE_DEPLOY

    def test_conditional_only_if(self):
        assert parse_stance("Deploy only if we add extra monitoring.") == STANCE_CONDITIONAL

    def test_conditional_beats_deploy_when_hedged(self):
        # "deploy ... with safeguards" must read CONDITIONAL, not DEPLOY.
        text = "We can deploy it, but only with safeguards and rollback ready."
        assert parse_stance(text) == STANCE_CONDITIONAL

    def test_conditional_keyword(self):
        assert parse_stance("My answer is conditional approval.") == STANCE_CONDITIONAL


class TestParseStanceDefault:
    """Unreadable / empty answers default to CONDITIONAL (the safe middle)."""

    def test_empty_string(self):
        assert parse_stance("") == DEFAULT_STANCE

    def test_none_like_whitespace(self):
        assert parse_stance("   \n  ") == DEFAULT_STANCE

    def test_no_signal(self):
        assert parse_stance("The weather is pleasant and unrelated.") == DEFAULT_STANCE

    def test_default_is_conditional(self):
        # Pin the documented default so a future change to DEFAULT_STANCE is loud.
        assert DEFAULT_STANCE == STANCE_CONDITIONAL

    def test_generation_error_text_defaults(self):
        # The error placeholder run_debate inserts on a failed model.
        assert parse_stance("[generation error: CUDA out of memory]") == DEFAULT_STANCE

    def test_tie_breaks_toward_route(self):
        # Equal deploy/route signal -> ROUTE (constitution prefers safe baseline).
        text = "We could deploy it. Or route it. Hard call."
        assert parse_stance(text) == STANCE_ROUTE

    def test_all_results_in_vocab(self):
        for s in (parse_stance("STANCE: DEPLOY"), parse_stance("route it"), parse_stance("")):
            assert s in STANCES


# ---------------------------------------------------------------------------
# (b) consensus: majority vote + agreement math on hand-built rounds
# ---------------------------------------------------------------------------

class TestComputeConsensus:
    def test_unanimous_route(self):
        final = [
            {"model": "a", "stance": STANCE_ROUTE, "text": "x"},
            {"model": "b", "stance": STANCE_ROUTE, "text": "y"},
        ]
        out = compute_consensus(final)
        assert out["verdict"] == STANCE_ROUTE
        assert out["agreement"] == 1.0
        assert out["vote_breakdown"] == {STANCE_DEPLOY: 0, STANCE_ROUTE: 2, STANCE_CONDITIONAL: 0}

    def test_clear_majority(self):
        final = [
            {"model": "a", "stance": STANCE_DEPLOY, "text": ""},
            {"model": "b", "stance": STANCE_DEPLOY, "text": ""},
            {"model": "c", "stance": STANCE_ROUTE, "text": ""},
        ]
        out = compute_consensus(final)
        assert out["verdict"] == STANCE_DEPLOY
        assert out["agreement"] == pytest.approx(2 / 3)

    def test_tie_breaks_toward_route_over_deploy(self):
        # 1 DEPLOY vs 1 ROUTE -> ROUTE wins (safety-first tie-break).
        final = [
            {"model": "a", "stance": STANCE_DEPLOY, "text": ""},
            {"model": "b", "stance": STANCE_ROUTE, "text": ""},
        ]
        out = compute_consensus(final)
        assert out["verdict"] == STANCE_ROUTE
        assert out["agreement"] == 0.5

    def test_tie_breaks_route_over_conditional(self):
        final = [
            {"model": "a", "stance": STANCE_CONDITIONAL, "text": ""},
            {"model": "b", "stance": STANCE_ROUTE, "text": ""},
        ]
        out = compute_consensus(final)
        assert out["verdict"] == STANCE_ROUTE

    def test_tie_breaks_conditional_over_deploy(self):
        final = [
            {"model": "a", "stance": STANCE_CONDITIONAL, "text": ""},
            {"model": "b", "stance": STANCE_DEPLOY, "text": ""},
        ]
        out = compute_consensus(final)
        assert out["verdict"] == STANCE_CONDITIONAL

    def test_missing_stance_defaults(self):
        # A record with no 'stance' key counts as DEFAULT_STANCE.
        final = [{"model": "a", "text": ""}, {"model": "b", "stance": STANCE_CONDITIONAL, "text": ""}]
        out = compute_consensus(final)
        assert out["verdict"] == STANCE_CONDITIONAL
        assert out["agreement"] == 1.0

    def test_empty_final_round(self):
        out = compute_consensus([])
        assert out["verdict"] == DEFAULT_STANCE
        assert out["agreement"] == 0.0

    def test_vote_breakdown_sums_to_n(self):
        final = [
            {"model": "a", "stance": STANCE_DEPLOY, "text": ""},
            {"model": "b", "stance": STANCE_ROUTE, "text": ""},
            {"model": "c", "stance": STANCE_CONDITIONAL, "text": ""},
        ]
        out = compute_consensus(final)
        assert sum(out["vote_breakdown"].values()) == 3


# ---------------------------------------------------------------------------
# (b2) consensus_label: a tie-broken verdict must not render as CONSENSUS
# ---------------------------------------------------------------------------

class TestConsensusLabel:
    def test_two_model_tie_is_no_consensus(self):
        # 1-1 split (2 models) -> 0.5 agreement: the verdict comes from the
        # safety-first tie-break, NOT from agreement. Must not say CONSENSUS.
        cons = compute_consensus([
            {"model": "a", "stance": STANCE_DEPLOY, "text": ""},
            {"model": "b", "stance": STANCE_ROUTE, "text": ""},
        ])
        assert cons["agreement"] == 0.5
        out = consensus_label(cons)
        assert out["label"] == LABEL_NO_CONSENSUS
        # The explanation names the safety-first tie-break.
        assert "tie-break" in out["explanation"]
        assert "ROUTE > CONDITIONAL > DEPLOY" in out["explanation"]

    def test_two_thirds_is_consensus(self):
        # 2-1 split (3 models) -> agreement exactly 2/3: at the bar -> CONSENSUS.
        cons = compute_consensus([
            {"model": "a", "stance": STANCE_ROUTE, "text": ""},
            {"model": "b", "stance": STANCE_ROUTE, "text": ""},
            {"model": "c", "stance": STANCE_DEPLOY, "text": ""},
        ])
        assert cons["agreement"] == pytest.approx(2 / 3)
        out = consensus_label(cons)
        assert out["label"] == LABEL_CONSENSUS

    def test_unanimous_is_consensus(self):
        cons = compute_consensus([
            {"model": "a", "stance": STANCE_ROUTE, "text": ""},
            {"model": "b", "stance": STANCE_ROUTE, "text": ""},
        ])
        assert cons["agreement"] == 1.0
        out = consensus_label(cons)
        assert out["label"] == LABEL_CONSENSUS

    def test_returns_exactly_label_and_explanation(self):
        out = consensus_label({"verdict": STANCE_ROUTE, "agreement": 1.0})
        assert set(out.keys()) == {"label", "explanation"}
        assert isinstance(out["explanation"], str) and out["explanation"]

    def test_does_not_mutate_the_consensus_dict(self):
        # Pure helper: the stored consensus shape (cached substrate included)
        # must pass through untouched.
        cons = {
            "verdict": STANCE_ROUTE,
            "vote_breakdown": {STANCE_DEPLOY: 1, STANCE_ROUTE: 1, STANCE_CONDITIONAL: 0},
            "agreement": 0.5,
        }
        snapshot = {**cons, "vote_breakdown": dict(cons["vote_breakdown"])}
        consensus_label(cons)
        assert cons == snapshot

    def test_junk_agreement_reads_as_no_consensus(self):
        out = consensus_label({"verdict": STANCE_DEPLOY, "agreement": "n/a"})
        assert out["label"] == LABEL_NO_CONSENSUS

    def test_threshold_is_two_thirds(self):
        # Pin the documented bar so a future change is loud.
        assert CONSENSUS_AGREEMENT_THRESHOLD == pytest.approx(2 / 3)

    def test_cached_substrate_example_reaches_consensus(self):
        # The bundled debate example is the SOTA 3-model cohort (Qwen3-8B +
        # Phi-4-mini + SmolLM3-3B): a genuine 2/3 majority for CONDITIONAL, so
        # it labels CONSENSUS with no safety-first tie-break. An odd cohort
        # guarantees a strict majority. Read-only: the cache is NOT edited.
        cached = json.loads(
            (_SPACE / "substrate" / "debate_examples.json").read_text(encoding="utf-8")
        )
        consensus = cached["consensus"]
        assert consensus["agreement"] >= 2 / 3
        out = consensus_label(consensus)
        assert out["label"] == LABEL_CONSENSUS
        assert "consensus bar" in out["explanation"]


# ---------------------------------------------------------------------------
# (c) run_debate end-to-end with a FAKE generate (no GPU / no network)
# ---------------------------------------------------------------------------

def _make_fake_generate(script: dict[tuple[str, int], str], default: str = "STANCE: CONDITIONAL"):
    """Build a fake generate keyed by (model_id, round_inferred_from_prompt).

    The prompt for round 1 contains 'Question under debate' but NOT 'Other
    adjudicators'; round 2+ contains 'Other adjudicators'. We infer the round
    from that marker so a single fake can return different text per round.
    """

    def _fake(model_id, prompt, backend="local", max_new_tokens=220):
        rnd = 2 if "Other adjudicators" in prompt else 1
        return script.get((model_id, rnd), default)

    return _fake


class TestRunDebateContract:
    def test_full_contract_shape(self, monkeypatch):
        script = {
            ("m1", 1): "STANCE: DEPLOY\nLooks adequate.",
            ("m2", 1): "STANCE: ROUTE\nToo risky.",
            ("m1", 2): "STANCE: ROUTE\nThe peer convinced me; route it.",
            ("m2", 2): "STANCE: ROUTE\nStill route.",
        }
        monkeypatch.setattr(debate, "generate", _make_fake_generate(script))

        out = run_debate("Deploy or route?", models=["m1", "m2"], backend="local", rounds=2)

        # Top-level keys exactly per the contract.
        assert set(out.keys()) == {
            "question", "models", "backend", "rounds", "consensus",
            "final_verdict", "elapsed_s",
        }
        assert out["question"] == "Deploy or route?"
        assert out["models"] == ["m1", "m2"]
        assert out["backend"] == "local"
        assert isinstance(out["elapsed_s"], float)
        assert out["elapsed_s"] >= 0.0

        # Two rounds, correctly typed.
        assert len(out["rounds"]) == 2
        assert out["rounds"][0]["round"] == 1
        assert out["rounds"][0]["round_type"] == ROUND_PROPOSE
        assert out["rounds"][1]["round"] == 2
        assert out["rounds"][1]["round_type"] == ROUND_CRITIQUE

        # Each response has model/stance/text and a stance in the vocab.
        for rnd in out["rounds"]:
            assert len(rnd["responses"]) == 2
            for resp in rnd["responses"]:
                assert set(resp.keys()) == {"model", "stance", "text"}
                assert resp["stance"] in STANCES

        # Consensus over the FINAL round (both ROUTE) -> ROUTE, agreement 1.0.
        assert out["consensus"]["verdict"] == STANCE_ROUTE
        assert out["final_verdict"] == STANCE_ROUTE
        assert out["consensus"]["agreement"] == 1.0
        assert out["consensus"]["vote_breakdown"][STANCE_ROUTE] == 2

    def test_consensus_uses_final_round_not_first(self, monkeypatch):
        # Round 1 leans DEPLOY; round 2 flips to ROUTE. Verdict must follow round 2.
        script = {
            ("m1", 1): "STANCE: DEPLOY",
            ("m2", 1): "STANCE: DEPLOY",
            ("m1", 2): "STANCE: ROUTE",
            ("m2", 2): "STANCE: ROUTE",
        }
        monkeypatch.setattr(debate, "generate", _make_fake_generate(script))
        out = run_debate("q", models=["m1", "m2"], rounds=2)
        assert out["final_verdict"] == STANCE_ROUTE

    def test_single_round(self, monkeypatch):
        script = {("m1", 1): "STANCE: DEPLOY", ("m2", 1): "STANCE: DEPLOY"}
        monkeypatch.setattr(debate, "generate", _make_fake_generate(script))
        out = run_debate("q", models=["m1", "m2"], rounds=1)
        assert len(out["rounds"]) == 1
        assert out["rounds"][0]["round_type"] == ROUND_PROPOSE
        assert out["final_verdict"] == STANCE_DEPLOY

    def test_rounds_floor_to_one(self, monkeypatch):
        # rounds=0 is clamped to a single PROPOSE round.
        script = {("m1", 1): "STANCE: ROUTE"}
        monkeypatch.setattr(debate, "generate", _make_fake_generate(script))
        out = run_debate("q", models=["m1"], rounds=0)
        assert len(out["rounds"]) == 1

    def test_model_failure_degrades_not_crashes(self, monkeypatch):
        # A generate that raises must not abort the debate; the model gets a
        # default stance and the run still returns a full contract.
        def _boom(model_id, prompt, backend="local", max_new_tokens=220):
            if model_id == "bad":
                raise RuntimeError("CUDA OOM")
            return "STANCE: ROUTE"

        monkeypatch.setattr(debate, "generate", _boom)
        out = run_debate("q", models=["bad", "good"], rounds=1)
        assert len(out["rounds"][0]["responses"]) == 2
        bad_resp = next(r for r in out["rounds"][0]["responses"] if r["model"] == "bad")
        assert bad_resp["stance"] == DEFAULT_STANCE
        assert "generation error" in bad_resp["text"]
        # The healthy model still voted ROUTE; consensus is well-formed.
        assert out["final_verdict"] in STANCES


class TestRunDebateOnEvent:
    def test_on_event_fires_all_event_types(self, monkeypatch):
        script = {
            ("m1", 1): "STANCE: DEPLOY",
            ("m2", 1): "STANCE: ROUTE",
            ("m1", 2): "STANCE: ROUTE",
            ("m2", 2): "STANCE: ROUTE",
        }
        monkeypatch.setattr(debate, "generate", _make_fake_generate(script))

        events: list[dict] = []
        run_debate("q", models=["m1", "m2"], rounds=2, on_event=events.append)

        types = [e["type"] for e in events]
        # 2 round_start + 4 model_response + 1 consensus.
        assert types.count("round_start") == 2
        assert types.count("model_response") == 4
        assert types.count("consensus") == 1
        # Ordering: a round_start precedes that round's model_responses;
        # consensus is last.
        assert types[0] == "round_start"
        assert types[-1] == "consensus"

    def test_model_response_event_payload(self, monkeypatch):
        script = {("m1", 1): "STANCE: DEPLOY\nbody"}
        monkeypatch.setattr(debate, "generate", _make_fake_generate(script))
        events: list[dict] = []
        run_debate("q", models=["m1"], rounds=1, on_event=events.append)

        mr = next(e for e in events if e["type"] == "model_response")
        assert mr["model"] == "m1"
        assert mr["round"] == 1
        assert mr["round_type"] == ROUND_PROPOSE
        assert mr["stance"] == STANCE_DEPLOY
        assert len(mr["text"]) <= 400

    def test_event_text_is_truncated(self, monkeypatch):
        long_text = "STANCE: ROUTE " + ("x" * 5000)
        script = {("m1", 1): long_text}
        monkeypatch.setattr(debate, "generate", _make_fake_generate(script))
        events: list[dict] = []
        run_debate("q", models=["m1"], rounds=1, on_event=events.append)
        mr = next(e for e in events if e["type"] == "model_response")
        assert len(mr["text"]) == 400

    def test_consensus_event_matches_return(self, monkeypatch):
        script = {("m1", 1): "STANCE: ROUTE", ("m2", 1): "STANCE: ROUTE"}
        monkeypatch.setattr(debate, "generate", _make_fake_generate(script))
        events: list[dict] = []
        out = run_debate("q", models=["m1", "m2"], rounds=1, on_event=events.append)
        cons = next(e for e in events if e["type"] == "consensus")
        assert cons["verdict"] == out["final_verdict"]
        assert cons["agreement"] == out["consensus"]["agreement"]
        assert cons["vote_breakdown"] == out["consensus"]["vote_breakdown"]

    def test_broken_callback_does_not_abort(self, monkeypatch):
        # A callback that raises must be swallowed; the debate still completes.
        script = {("m1", 1): "STANCE: ROUTE"}
        monkeypatch.setattr(debate, "generate", _make_fake_generate(script))

        def _bad_cb(ev):
            raise ValueError("UI exploded")

        out = run_debate("q", models=["m1"], rounds=1, on_event=_bad_cb)
        assert out["final_verdict"] in STANCES


# ---------------------------------------------------------------------------
# (d) backend contract: unknown backend + dead-dep errors are clear
# ---------------------------------------------------------------------------

class TestBackendContract:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            generate("m1", "p", backend="banana")

    def test_modal_without_env_raises(self, monkeypatch):
        monkeypatch.delenv("MODAL_ENDPOINT", raising=False)
        with pytest.raises(EnvironmentError, match="MODAL_ENDPOINT"):
            generate("m1", "p", backend="modal")

    def test_constitution_is_nonempty_constant(self):
        assert isinstance(CONSTITUTION, str)
        assert "DEPLOY" in CONSTITUTION and "ROUTE" in CONSTITUTION and "CONDITIONAL" in CONSTITUTION


# ---------------------------------------------------------------------------
# (e) Modal client contract: auth header, detail surfacing, quantization,
#     300 s timeout — all against a FAKE requests module (no network)
# ---------------------------------------------------------------------------

class _FakeResp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("response body is not JSON")
        return self._payload


class _FakeRequests:
    """Stands in for the lazily-imported ``requests`` module in _generate_modal."""

    def __init__(self, resp: _FakeResp):
        self.resp = resp
        self.calls: list[dict] = []

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002 — mirrors requests' kwarg
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return self.resp


class TestModalClient:
    def _install(self, monkeypatch, resp: _FakeResp) -> _FakeRequests:
        fake = _FakeRequests(resp)
        monkeypatch.setitem(sys.modules, "requests", fake)
        monkeypatch.setenv("MODAL_ENDPOINT", "http://modal.test/generate")
        return fake

    def test_success_parses_text_and_records_quantization(self, monkeypatch):
        fake = self._install(
            monkeypatch,
            _FakeResp(200, {"text": "  STANCE: ROUTE\nToo risky.  ", "quantization": "nf4-4bit"}),
        )
        monkeypatch.setenv("MODAL_TOKEN", "sekret-token")
        out = generate("m1", "p", backend="modal")
        assert out == "STANCE: ROUTE\nToo risky."
        # The quantization disclosure is surfaced for the UI.
        assert debate.LAST_MODAL_QUANTIZATION == "nf4-4bit"
        call = fake.calls[0]
        assert call["url"] == "http://modal.test/generate"
        assert call["headers"]["Authorization"] == "Bearer sekret-token"
        assert call["timeout"] == 300  # cold start can exceed 120 s
        assert call["json"]["model"] == "m1"
        assert call["json"]["max_new_tokens"] == 220
        # The constitutional frame rides along to the remote model.
        assert call["json"]["prompt"].startswith(CONSTITUTION)

    def test_no_token_sends_no_auth_header(self, monkeypatch):
        fake = self._install(monkeypatch, _FakeResp(200, {"text": "x"}))
        monkeypatch.delenv("MODAL_TOKEN", raising=False)
        generate("m1", "p", backend="modal")
        assert "Authorization" not in (fake.calls[0]["headers"] or {})

    def test_401_surfaces_detail_as_runtimeerror(self, monkeypatch):
        self._install(
            monkeypatch, _FakeResp(401, {"detail": "Missing or invalid bearer token"})
        )
        with pytest.raises(RuntimeError, match="Missing or invalid bearer token"):
            generate("m1", "p", backend="modal")

    def test_400_bad_input_surfaces_detail(self, monkeypatch):
        self._install(monkeypatch, _FakeResp(400, {"detail": "unknown model 'zzz'"}))
        with pytest.raises(RuntimeError, match="unknown model"):
            generate("m1", "p", backend="modal")

    def test_non_json_error_body_falls_back_to_text(self, monkeypatch):
        self._install(monkeypatch, _FakeResp(502, None, text="Bad Gateway"))
        with pytest.raises(RuntimeError, match="Bad Gateway"):
            generate("m1", "p", backend="modal")

    def test_run_debate_surfaces_quantization(self, monkeypatch):
        # Full modal-backed debate (fake transport): the result carries the
        # endpoint's precision disclosure for the UI.
        self._install(
            monkeypatch, _FakeResp(200, {"text": "STANCE: ROUTE", "quantization": "bf16"})
        )
        out = run_debate("q", models=["m1", "m2"], backend="modal", rounds=1)
        assert out["quantization"] == "bf16"
        assert out["final_verdict"] == STANCE_ROUTE

    def test_run_debate_omits_quantization_when_endpoint_silent(self, monkeypatch):
        self._install(monkeypatch, _FakeResp(200, {"text": "STANCE: ROUTE"}))
        out = run_debate("q", models=["m1"], backend="modal", rounds=1)
        assert "quantization" not in out
