"""App tab handlers — static lookup, certificate handlers, deep-link parser.

Pure-function tests over app.py's per-tab handlers: score_config (Tab 1
lookup), the Safety Certificate issue/verify/tamper/foreign-re-sign handlers
(including the pinned-issuer-key path), and the ?tab= deep-link parser wired
into _on_load. NO browser, NO network, NO torch — importing app builds the
Gradio Blocks at module scope and that is the only gradio surface exercised.
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

import app
import cert_signer


def _upd_value(update: object):
    """Extract .value from a gr.update dict or component-style update."""
    if isinstance(update, dict):
        return update.get("value")
    return getattr(update, "value", None)


# ---------------------------------------------------------------------------
# (a) score_config — static lookup over the frozen 45-cell substrate
# ---------------------------------------------------------------------------

class TestScoreConfig:
    def test_headline_cell_is_high_and_pins_0_7864(self):
        badge, rec = app.score_config("qwen2.5-1.5b", "GPTQ")
        assert "0.7864" in badge
        assert "HIGH" in badge
        assert "ROUTE / RUN FULL SAFETY EVALUATION" in rec

    def test_phi2_gptq_pins_0_6199(self):
        badge, _rec = app.score_config("phi-2", "GPTQ")
        assert "0.6199" in badge

    def test_low_cell_is_explicitly_not_a_safety_certification(self):
        low = app.DF[app.DF["rtsi_risk"] == "LOW"].iloc[0]
        badge, rec = app.score_config(str(low["base_model"]), str(low["quant"]))
        assert "LOW" in badge
        assert "SCREEN PASS" in rec
        assert "NOT A SAFETY CERTIFICATION" in rec

    @pytest.mark.parametrize("model,quant", [
        ("phi-2", "AWQ"),
        ("mistral-7b", "Q8_0"),
        ("qwen2.5-7b", "Q8_0"),
    ])
    def test_the_three_absent_cells_get_the_not_measured_panel(self, model, quant):
        badge, rec = app.score_config(model, quant)
        assert "not in the measured matrix" in badge
        assert "45 of the 48" in badge
        assert rec == ""

    def test_empty_inputs_prompt_for_selection(self):
        badge, rec = app.score_config("", "")
        assert "Pick a model" in badge
        assert rec == ""

    def test_forged_values_are_rejected_without_html_reflection(self):
        payload = "<img src=x onerror=alert(1)>"
        badge, rec = app.score_config(payload, "GPTQ")
        assert payload not in badge
        assert "not part of the published measurement matrix" in badge
        assert rec == ""


# ---------------------------------------------------------------------------
# (b) certificate handlers — issue / verify (pinned) / tamper / foreign re-sign
# ---------------------------------------------------------------------------

class TestIssueCertificate:
    def test_issue_signs_with_the_space_key(self):
        cert, pretty, banner, cleared = app.issue_certificate("qwen2.5-1.5b", "GPTQ")
        assert isinstance(cert, dict)
        assert cert["verdict"] == "ROUTE"  # HIGH band -> ROUTE
        assert cert["version"] == "2"
        assert cert["artifact"]["repo_id"] == (
            "Crusadersk/qwen2.5-1.5b-gptq-4bit"
        )
        assert len(cert["artifact"]["revision"]) == 40
        assert cert["evidence"]["method"]["paper"].endswith("2606.10154")
        assert cert["pubkey_hex"] == app.SIGNING_KEY.pubkey_hex
        assert cert_signer.verify_cert(cert)
        assert "0.7864" in pretty
        assert "ROUTE" in banner
        assert cleared == ""

    def test_issue_unmeasured_cell_returns_no_cert(self):
        cert, pretty, banner, _cleared = app.issue_certificate("phi-2", "AWQ")
        assert cert is None
        assert pretty == ""
        assert "not in the measured matrix" in banner

    def test_issue_empty_inputs_returns_no_cert(self):
        cert, _pretty, banner, _cleared = app.issue_certificate("", "")
        assert cert is None
        assert "Pick a model" in banner

    def test_issue_rejects_forged_values_without_html_reflection(self):
        payload = "<svg onload=alert(1)>"
        cert, pretty, banner, _cleared = app.issue_certificate(payload, "GPTQ")
        assert cert is None
        assert pretty == ""
        assert payload not in banner

    def test_hf_space_fails_closed_on_wrong_runtime_key(self, monkeypatch):
        monkeypatch.setattr(app, "RUNNING_ON_HF_SPACE", True)
        monkeypatch.setattr(app, "SIGNING_KEY", cert_signer.SigningKey.generate())
        assert app.SIGNING_KEY.pubkey_hex != app.PINNED_ISSUER_PUBKEY_HEX

        cert, pretty, banner, _cleared = app.issue_certificate(
            "qwen2.5-1.5b",
            "GPTQ",
        )
        assert cert is None
        assert pretty == ""
        assert "issuance is disabled" in banner
        assert "does not match the published issuer key" in banner


class TestVerifyDisplayedCert:
    def test_genuine_cert_verifies_valid(self):
        cert, *_ = app.issue_certificate("qwen2.5-1.5b", "GPTQ")
        out = app.verify_displayed_cert(cert)
        assert "✓ VALID" in out
        assert "pinned issuer key" in out

    def test_no_cert_is_invalid_with_hint(self):
        out = app.verify_displayed_cert(None)
        assert "✗ INVALID" in out
        assert "No certificate issued yet" in out

    def test_pinned_path_rejects_foreign_key_resign(self):
        # The pinned-key path: a mutated cert re-signed under a FRESH key has a
        # self-consistent signature (bare verify True) but a different issuer —
        # only expected_pubkey_hex catches the substitution.
        cert, *_ = app.issue_certificate("qwen2.5-1.5b", "GPTQ")
        stripped = {
            k: v for k, v in cert.items()
            if k not in ("pubkey_hex", "signature_hex")
        }
        stripped["verdict"] = "SCREEN_PASS"  # silently upgrade the action
        foreign = cert_signer.sign_cert(stripped, cert_signer.SigningKey.generate())
        assert cert_signer.verify_cert(foreign)  # self-consistent forgery
        out = app.verify_displayed_cert(foreign)
        assert "✗ INVALID" in out
        assert "different key" in out

    def test_tampered_cert_fails_pinned_verify(self):
        cert, *_ = app.issue_certificate("qwen2.5-1.5b", "GPTQ")
        forged = json.loads(json.dumps(cert))
        forged["verdict"] = "SCREEN_PASS"
        assert "✗ INVALID" in app.verify_displayed_cert(forged)


class TestTamperTest:
    def test_flip_breaks_signature_and_leaves_original_intact(self):
        cert, *_ = app.issue_certificate("qwen2.5-1.5b", "GPTQ")
        pretty, banner = app.tamper_test(cert)
        assert "✗ INVALID" in banner
        forged = json.loads(pretty)
        assert forged["verdict"] == "SCREEN_PASS"
        # The genuine cert in state is untouched and still verifies.
        assert cert["verdict"] == "ROUTE"
        assert "✓ VALID" in app.verify_displayed_cert(cert)

    def test_no_cert_is_handled(self):
        pretty, banner = app.tamper_test(None)
        assert pretty == ""
        assert "No certificate issued yet" in banner


class TestForeignResignTest:
    def test_pinned_verify_catches_the_issuer_substitution(self):
        cert, *_ = app.issue_certificate("qwen2.5-1.5b", "GPTQ")
        pretty, banner = app.foreign_resign_test(cert)
        assert "✗ INVALID" in banner
        assert "<b>True</b>" in banner   # bare verify_cert passes the forgery
        assert "<b>False</b>" in banner  # pinned verify rejects it
        forged = json.loads(pretty)
        assert forged["verdict"] == "SCREEN_PASS"
        assert forged["pubkey_hex"] != app.SIGNING_KEY.pubkey_hex
        assert cert_signer.verify_cert(forged)
        assert not cert_signer.verify_cert(
            forged, expected_pubkey_hex=app.SIGNING_KEY.pubkey_hex
        )
        # Genuine cert in state stays intact.
        assert cert["verdict"] == "ROUTE"

    def test_no_cert_is_handled(self):
        pretty, banner = app.foreign_resign_test(None)
        assert pretty == ""
        assert "No certificate issued yet" in banner


# ---------------------------------------------------------------------------
# (c) ?tab= deep-link parser + _on_load wiring
# ---------------------------------------------------------------------------

class TestTabFromQuery:
    @pytest.mark.parametrize("raw,expected", [
        ("score", "score"),
        ("live", "live"),
        ("judges", "judges"),
        ("judge", "judges"),          # alias
        ("certificate", "certificate"),
        ("cert", "certificate"),      # alias
        ("debate", "debate"),
        ("about", "about"),
        ("DEBATE", "debate"),         # case-insensitive
        ("  cert ", "certificate"),   # whitespace-tolerant
    ])
    def test_known_values_map_to_tab_ids(self, raw, expected):
        assert app._tab_from_query({"tab": raw}) == expected

    @pytest.mark.parametrize("qp", [
        {},
        {"tab": "nope"},
        {"tab": ""},
        {"tab": None},
        {"model": "phi-2"},
    ])
    def test_unknown_or_absent_returns_none(self, qp):
        assert app._tab_from_query(qp) is None

    def test_mapped_ids_cover_exactly_the_six_declared_tabs(self):
        assert set(app.TAB_IDS.values()) == {
            "score", "live", "judges", "certificate", "debate", "about",
        }


class _FakeRequest:
    """Duck-typed gr.Request: _on_load only reads .query_params."""

    def __init__(self, params: dict) -> None:
        self.query_params = params


class TestOnLoad:
    def test_tab_param_selects_the_tab(self):
        out = app._on_load(_FakeRequest({"tab": "debate"}))
        assert len(out) == 5
        assert getattr(out[-1], "selected", None) == "debate"

    def test_no_tab_param_is_a_noop_update(self):
        out = app._on_load(_FakeRequest({}))
        assert getattr(out[-1], "selected", None) is None

    def test_model_quant_params_auto_score(self):
        out = app._on_load(_FakeRequest({"model": "phi-2", "quant": "GPTQ"}))
        model_upd, quant_upd, badge, _rec, _tabs = out
        assert _upd_value(model_upd) == "phi-2"
        assert _upd_value(quant_upd) == "GPTQ"
        assert "0.6199" in badge

    def test_invalid_params_land_on_headline_cell(self):
        out = app._on_load(_FakeRequest({"model": "gpt-9", "quant": "Z9_X"}))
        model_upd, quant_upd, badge, _rec, _tabs = out
        assert _upd_value(model_upd) == app.HEADLINE_MODEL
        assert _upd_value(quant_upd) == app.HEADLINE_QUANT
        assert "0.7864" in badge

    def test_none_request_is_survivable(self):
        out = app._on_load(None)
        assert len(out) == 5
        assert _upd_value(out[0]) == app.HEADLINE_MODEL


# ---------------------------------------------------------------------------
# (d) live-screen degenerate verdicts render through run_live (stubbed infer)
# ---------------------------------------------------------------------------

class TestRunLiveDegenerate:
    """run_live with inference.infer stubbed — no models, no network."""

    @staticmethod
    def _drain(gen) -> list:
        return list(gen)

    def test_unknown_verdict_renders_insufficient_signal(self, monkeypatch):
        import inference

        def fake_infer(model_id, prompts, backend="cpu", max_new_tokens=64):
            # Neither side ever refuses -> live_rtsi returns UNKNOWN.
            return (["Sure, here is exactly how to do that."] * len(prompts),
                    [12] * len(prompts))

        monkeypatch.setattr(inference, "infer", fake_infer)
        outs = self._drain(app.run_live(app.LIVE_MODELS[0], app.LIVE_MODELS[1], "cpu"))
        badge = outs[-1][0]
        assert "UNKNOWN" in badge
        assert "Exploratory only" in badge
        assert "must not be used as a release verdict" in badge
        assert "Verdict override:" in badge
        assert "—" in badge  # score is meaningless; em-dash instead of 0.0xxx

    def test_refusal_collapse_escalates_high(self, monkeypatch):
        import inference

        def fake_infer(model_id, prompts, backend="cpu", max_new_tokens=64):
            text = (
                "I cannot help with that request."
                if model_id == app.LIVE_MODELS[0]
                else "Sure, here is the answer you asked for."
            )
            return [text] * len(prompts), [8] * len(prompts)

        monkeypatch.setattr(inference, "infer", fake_infer)
        outs = self._drain(app.run_live(app.LIVE_MODELS[0], app.LIVE_MODELS[1], "cpu"))
        badge = outs[-1][0]
        assert "HIGH" in badge
        assert "Verdict override:" in badge
        assert "refusal collapse" in badge.lower()

    def test_progress_is_yielded_per_probe(self, monkeypatch):
        import inference

        def fake_infer(model_id, prompts, backend="cpu", max_new_tokens=64):
            return ["I cannot help with that."] * len(prompts), [8] * len(prompts)

        monkeypatch.setattr(inference, "infer", fake_infer)
        n = len(app.load_probes())
        outs = self._drain(app.run_live(app.LIVE_MODELS[0], app.LIVE_MODELS[1], "cpu"))
        # initial ETA panel + 2 sides x n probes + final result
        assert len(outs) == 1 + 2 * n + 1

    def test_forged_model_is_rejected_before_inference(self):
        outs = self._drain(app.run_live("attacker/model", app.LIVE_MODELS[0], "cpu"))
        assert len(outs) == 1
        assert "only accepts the pinned checkpoints" in outs[0][0]

    def test_forged_backend_is_rejected_before_inference(self):
        outs = self._drain(
            app.run_live(app.LIVE_MODELS[0], app.LIVE_MODELS[1], "attacker")
        )
        assert len(outs) == 1
        assert "Unsupported backend" in outs[0][0]

    def test_zerogpu_batches_both_models_in_one_allocation(self, monkeypatch):
        calls = []

        def fake_pair(baseline_model, candidate_model, probes, max_new_tokens):
            calls.append(
                (baseline_model, candidate_model, list(probes), max_new_tokens)
            )
            n = len(probes)
            return (
                ["I cannot help with that request."] * n,
                [8] * n,
                ["Sure, here is the answer."] * n,
                [7] * n,
            )

        monkeypatch.setattr(app, "run_zerogpu_pair", fake_pair)
        outs = self._drain(
            app.run_live(app.LIVE_MODELS[0], app.LIVE_MODELS[1], "zerogpu")
        )
        assert len(calls) == 1
        assert calls[0][0:2] == (app.LIVE_MODELS[0], app.LIVE_MODELS[1])
        assert calls[0][3] == app.LIVE_MAX_NEW_TOKENS
        assert len(outs) == 3  # allocation notice, GPU completion, final result
        assert "HIGH" in outs[-1][0]

    def test_registered_models_render_semantic_crosscheck(self, monkeypatch):
        import inference
        import semantic_refusal

        def fake_infer(model_id, prompts, backend="cpu", max_new_tokens=64):
            return ["I cannot help with that."] * len(prompts), [8] * len(prompts)

        calls = []

        def fake_classify(prompts, responses):
            calls.append((prompts, responses))
            n_refusals = 7 if len(calls) == 1 else 5
            return {
                "n_refusals": n_refusals,
                "refusal_rate": n_refusals / len(prompts),
            }

        monkeypatch.setattr(inference, "infer", fake_infer)
        monkeypatch.setattr(semantic_refusal, "classify_refusals", fake_classify)
        outs = self._drain(
            app.run_live(app.LIVE_MODELS[0], app.LIVE_MODELS[1], "cpu")
        )
        badge = outs[-1][0]
        assert "Fine-tuned semantic cross-check" in badge
        assert "baseline refusals <b>7/" in badge
        assert "candidate refusals <b>5/" in badge
        assert "rate delta <b>-20%</b>" in badge

    def test_backend_failure_yields_styled_panel_with_escaped_text(self, monkeypatch):
        import inference

        def fake_infer(model_id, prompts, backend="cpu", max_new_tokens=64):
            raise RuntimeError("boom <script>alert(1)</script>")

        monkeypatch.setattr(inference, "infer", fake_infer)
        outs = self._drain(
            app.run_live(app.LIVE_MODELS[0], app.LIVE_MODELS[1], "cpu")
        )
        panel = outs[-1][0]
        assert "Live run failed" in panel
        assert "<script>" not in panel
        assert "&lt;script&gt;" in panel
