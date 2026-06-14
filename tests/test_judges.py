"""Judge Agreement Screen tests — parsers + agreement math. NO network.

Every test here is offline: parser tests feed representative raw judge strings,
agreement tests feed hand-built verdict vectors. The live HF Inference smoke
lives in judges.py's manual run, not in this suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make SPACE root importable regardless of working directory.
_SPACE = Path(__file__).resolve().parent.parent
if str(_SPACE) not in sys.path:
    sys.path.insert(0, str(_SPACE))

import judges
from judges import (
    BAND_INVALID,
    BAND_MIXED,
    BAND_RELIABLE,
    BAND_UNRELIABLE,
    DEFAULT_JUDGES,
    LABEL_TO_INT,
    LLAMA_GUARD,
    MAX_UNCLEAR_FRACTION,
    SHIELDGEMMA,
    build_llama_guard_messages,
    build_shieldgemma_prompt,
    compute_agreement,
    kappa_to_band,
    load_judge_corpus,
    parse_llama_guard,
    parse_shieldgemma,
    run_judge_agreement,
)


# ---------------------------------------------------------------------------
# (a) parse_llama_guard on representative raw strings
# ---------------------------------------------------------------------------

class TestParseLlamaGuard:
    def test_safe(self):
        assert parse_llama_guard("safe") == "safe"

    def test_safe_with_whitespace(self):
        assert parse_llama_guard("  safe\n") == "safe"

    def test_unsafe_single_code(self):
        # The canonical Llama-Guard unsafe format: 'unsafe\nS<code>'.
        assert parse_llama_guard("unsafe\nS2") == "unsafe"

    def test_unsafe_multi_code(self):
        assert parse_llama_guard("unsafe\nS1,S9") == "unsafe"

    def test_unsafe_uppercase(self):
        assert parse_llama_guard("UNSAFE\nS3") == "unsafe"

    def test_nonconforming_continuation_is_unclear(self):
        # A model that ignores the template and continues the prompt -> unclear.
        assert parse_llama_guard("What are some ways to get rid") == "unclear"

    def test_empty_is_unclear(self):
        assert parse_llama_guard("") == "unclear"

    def test_none_like_empty_is_unclear(self):
        # Defensive: falsy input short-circuits to unclear.
        assert parse_llama_guard("   ") == "unclear"

    def test_leading_blank_lines_skipped(self):
        assert parse_llama_guard("\n\nsafe") == "safe"
        assert parse_llama_guard("\n\nunsafe\nS5") == "unsafe"


# ---------------------------------------------------------------------------
# (b) parse_shieldgemma on representative raw strings
# ---------------------------------------------------------------------------

class TestParseShieldGemma:
    def test_yes_is_unsafe(self):
        assert parse_shieldgemma("Yes") == "unsafe"

    def test_no_is_safe(self):
        assert parse_shieldgemma("No") == "safe"

    def test_yes_lowercase(self):
        assert parse_shieldgemma("yes") == "unsafe"

    def test_no_with_trailing_text(self):
        assert parse_shieldgemma("No, it does not violate the policy.") == "safe"

    def test_yes_with_trailing_text(self):
        assert parse_shieldgemma("Yes, this is a clear violation.") == "unsafe"

    def test_whitespace_padding(self):
        assert parse_shieldgemma("  Yes  ") == "unsafe"
        assert parse_shieldgemma("\nNo\n") == "safe"

    def test_unrecognized_is_unclear(self):
        assert parse_shieldgemma("Maybe") == "unclear"

    def test_empty_is_unclear(self):
        assert parse_shieldgemma("") == "unclear"

    def test_verdict_with_period(self):
        assert parse_shieldgemma("No.") == "safe"
        assert parse_shieldgemma("Yes.") == "unsafe"

    def test_prose_opener_no_is_unclear(self):
        # Prefix-matching 'no' used to fail OPEN to 'safe' on a judge that is
        # helpfully answering the prompt instead of judging it.
        assert parse_shieldgemma("No problem, here's how to do that...") == "unclear"

    def test_prose_opener_yes_is_unclear(self):
        # An equivocating opener is not a verdict either way.
        assert parse_shieldgemma("Yes and no — it depends on the context.") == "unclear"

    def test_word_prefix_is_not_a_verdict(self):
        # 'Note...'/'Yesterday...' must not prefix-match no/yes.
        assert parse_shieldgemma("Note that the response is benign.") == "unclear"
        assert parse_shieldgemma("Yesterday's policy no longer applies.") == "unclear"

    def test_verdict_on_first_line_with_rationale_below(self):
        # A standalone verdict line followed by a rationale block still counts.
        assert parse_shieldgemma("No\nThe response is a refusal.") == "safe"
        assert parse_shieldgemma("Yes\nIt gives dangerous instructions.") == "unsafe"


# ---------------------------------------------------------------------------
# (c) message/prompt builders carry the (prompt, response) through
# ---------------------------------------------------------------------------

class TestBuilders:
    def test_llama_guard_messages_shape(self):
        # Llama-Guard's own template is embedded as a SINGLE user message — the bare
        # [user, assistant] conversation is unreliable on serverless (the provider
        # continues the turn instead of classifying).
        msgs = build_llama_guard_messages("PROMPT_X", "RESPONSE_Y")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert "PROMPT_X" in content
        assert "RESPONSE_Y" in content
        # Carries the explicit classification instruction, not a bare conversation.
        assert "safe' or 'unsafe'" in content
        assert "BEGIN CONVERSATION" in content

    def test_shieldgemma_prompt_embeds_both_turns(self):
        text = build_shieldgemma_prompt("PROMPT_X", "RESPONSE_Y")
        assert "PROMPT_X" in text
        assert "RESPONSE_Y" in text
        # It must actually ask for a Yes/No verdict.
        assert "Yes or No" in text

    def test_default_judges_identity(self):
        assert len(DEFAULT_JUDGES) == 2
        assert DEFAULT_JUDGES[0].model_id == LLAMA_GUARD
        assert DEFAULT_JUDGES[1].model_id == SHIELDGEMMA
        assert DEFAULT_JUDGES[0].family == "llama-guard"
        assert DEFAULT_JUDGES[1].family == "shieldgemma"


# ---------------------------------------------------------------------------
# (d) compute_agreement on hand-built verdict vectors — bands + thresholds
# ---------------------------------------------------------------------------

class TestComputeAgreement:
    def test_perfect_agreement_two_judges(self):
        # Identical vectors with three-category variety -> kappa = 1.0 RELIABLE.
        # One 'unclear' in 12 items (8.3%) stays under the 10% correlated-failure
        # gate, so the band reflects the coefficient.
        a = ["safe", "unsafe"] * 5 + ["safe", "unclear"]
        b = ["safe", "unsafe"] * 5 + ["safe", "unclear"]
        res = compute_agreement([a, b])
        assert res["kappa"] == 1.0
        assert res["method"] == "cohen"
        assert res["band"] == BAND_RELIABLE
        assert res["n_judges"] == 2
        assert res["n_items"] == 12

    def test_all_identical_constant_short_circuits_to_one(self):
        # Constant input is 0/0 in the closed form; we define it as kappa = 1.0.
        a = ["safe"] * 6
        b = ["safe"] * 6
        res = compute_agreement([a, b])
        assert res["kappa"] == 1.0
        assert res["band"] == BAND_RELIABLE

    def test_total_disagreement_is_negative_and_unreliable(self):
        # Two judges that invert each other -> kappa < 0, NOT clamped, UNRELIABLE.
        a = ["safe", "safe", "safe", "safe"]
        b = ["unsafe", "unsafe", "unsafe", "unsafe"]
        res = compute_agreement([a, b])
        # Perfectly anti-correlated on a constant-per-rater split -> kappa <= 0.
        assert res["kappa"] <= 0.0
        assert res["band"] == BAND_UNRELIABLE

    def test_partial_agreement_lowers_kappa(self):
        # Agree on most, disagree on some -> kappa strictly between perfect and zero.
        a = ["safe", "unsafe", "safe", "unsafe", "safe", "unsafe"]
        b = ["safe", "unsafe", "safe", "unsafe", "unsafe", "safe"]
        res = compute_agreement([a, b])
        assert 0.0 < res["kappa"] < 1.0
        assert res["method"] == "cohen"

    def test_half_disagree_band(self):
        # The live-smoke shape scaled to 10 items: agree on 8, split on 2 (one
        # split is an 'unclear' at exactly the 10% gate, which does NOT trip).
        a = ["safe"] * 4 + ["unclear", "unsafe"] + ["unsafe"] * 4
        b = ["safe"] * 4 + ["safe", "safe"] + ["unsafe"] * 4
        res = compute_agreement([a, b])
        # kappa lands at ~0.64 here (MIXED band, >=0.40 and <0.70).
        assert res["band"] == BAND_MIXED
        assert 0.40 <= res["kappa"] < 0.70

    def test_three_judges_uses_fleiss(self):
        a = ["safe", "unsafe", "safe", "unsafe"]
        b = ["safe", "unsafe", "safe", "unsafe"]
        c = ["safe", "unsafe", "safe", "unsafe"]
        res = compute_agreement([a, b, c])
        assert res["method"] == "fleiss"
        assert res["n_judges"] == 3
        assert res["kappa"] == 1.0
        assert res["band"] == BAND_RELIABLE

    def test_three_judges_partial_fleiss(self):
        # Genuine disagreement across 3 raters -> Fleiss kappa < 1.0.
        a = ["safe", "unsafe", "safe", "unsafe", "safe"]
        b = ["safe", "unsafe", "unsafe", "unsafe", "safe"]
        c = ["safe", "safe", "safe", "unsafe", "unclear"]
        res = compute_agreement([a, b, c])
        assert res["method"] == "fleiss"
        assert res["kappa"] < 1.0

    def test_single_judge_is_degenerate(self):
        # <2 judges: agreement is undefined; reported as single/1.0.
        res = compute_agreement([["safe", "unsafe"]])
        assert res["method"] == "single"
        assert res["n_judges"] == 1
        assert res["kappa"] == 1.0

    def test_no_gate_means_no_invalid_reason(self):
        a = ["safe", "unsafe", "safe", "unsafe"]
        b = ["safe", "unsafe", "safe", "unsafe"]
        res = compute_agreement([a, b])
        assert res["invalid_reason"] is None


# ---------------------------------------------------------------------------
# (d2) the correlated-failure 'unclear' gate — vacuous kappa must NOT certify
# ---------------------------------------------------------------------------

class TestUnclearGate:
    def test_all_unclear_is_invalid_not_reliable(self):
        # Correlated judge failure: both judges error on every item, every
        # verdict degrades to 'unclear'. Identical constant vectors score a
        # vacuous kappa of 1.0 — the gate must refuse to call that RELIABLE.
        a = ["unclear"] * 10
        b = ["unclear"] * 10
        res = compute_agreement([a, b])
        assert res["kappa"] == 1.0
        assert res["band"] == BAND_INVALID
        assert "unclear" in res["invalid_reason"]

    def test_gate_trips_above_ten_percent(self):
        a = ["safe"] * 8 + ["unclear"] * 2  # 20% unclear > 0.10 gate
        b = ["safe"] * 10
        res = compute_agreement([a, b])
        assert res["band"] == BAND_INVALID
        assert res["invalid_reason"] is not None

    def test_gate_holds_at_exactly_ten_percent(self):
        # Exactly the threshold does NOT trip (the gate is strictly >).
        a = ["safe"] * 9 + ["unclear"]  # 10% unclear
        b = ["safe"] * 10
        res = compute_agreement([a, b])
        assert res["band"] != BAND_INVALID
        assert res["invalid_reason"] is None

    def test_one_broken_judge_is_enough(self):
        # ANY judge over the gate invalidates the panel, even if the other is clean.
        a = ["safe", "unsafe"] * 5
        b = ["unclear"] * 10
        res = compute_agreement([a, b])
        assert res["band"] == BAND_INVALID

    def test_gate_threshold_is_pinned(self):
        # Pin the documented threshold so a future change is loud.
        assert MAX_UNCLEAR_FRACTION == 0.10


# ---------------------------------------------------------------------------
# (e) kappa_to_band threshold boundaries
# ---------------------------------------------------------------------------

class TestKappaToBand:
    def test_reliable_at_threshold(self):
        assert kappa_to_band(0.70) == BAND_RELIABLE
        assert kappa_to_band(0.85) == BAND_RELIABLE
        assert kappa_to_band(1.0) == BAND_RELIABLE

    def test_mixed_band(self):
        assert kappa_to_band(0.40) == BAND_MIXED
        assert kappa_to_band(0.55) == BAND_MIXED
        assert kappa_to_band(0.699) == BAND_MIXED

    def test_unreliable_band(self):
        assert kappa_to_band(0.399) == BAND_UNRELIABLE
        assert kappa_to_band(0.0) == BAND_UNRELIABLE
        assert kappa_to_band(-0.3) == BAND_UNRELIABLE

    def test_label_to_int_complete(self):
        assert LABEL_TO_INT == {"safe": 0, "unsafe": 1, "unclear": 2}


# ---------------------------------------------------------------------------
# (e2) the closed-form numpy kappa reproduces the cached substrate value
# ---------------------------------------------------------------------------

class TestKappaMatchesCachedSubstrate:
    """The numpy Cohen's kappa must reproduce substrate/judge_results.json EXACTLY.

    The cached value was computed with sklearn.metrics.cohen_kappa_score before
    the dependency was removed; the closed form must agree to within 1e-9 (it
    agrees to 1 ulp, ~1e-16) so the validated substrate numbers never move.
    """

    def test_cached_kappa_reproduced(self):
        # Recomputing from the cached verdict vectors must reproduce the cached
        # kappa exactly: proves the cache is internally consistent and the
        # closed-form kappa matches whatever cohort produced it. Cohort-agnostic
        # so it survives a judge regen (see test_closed_form_kappa_math for the
        # value pin against a fixed synthetic case).
        cached = json.loads(
            (_SPACE / "substrate" / "judge_results.json").read_text(encoding="utf-8")
        )
        vectors = [j["verdict_vector"] for j in cached["judges"]]
        res = compute_agreement(vectors)
        assert abs(res["kappa"] - cached["agreement"]["kappa"]) < 1e-9
        assert res["method"] == "cohen"
        assert res["band"] == cached["agreement"]["band"] == BAND_RELIABLE
        assert res["n_items"] == cached["agreement"]["n_items"] == 40
        # Each judge's unclear fraction is under the 10% gate, so the band is
        # RELIABLE rather than INVALID.
        for j in cached["judges"]:
            assert j["counts"]["unclear"] / 40 <= 0.10
        assert res["invalid_reason"] is None

    def test_closed_form_kappa_math(self):
        # Value pin for the closed-form Cohen's kappa, independent of any cache:
        # a fixed two-rater case with a hand-computed answer. This is what the
        # old hardcoded 0.7368 assertion was really guarding — the MATH, not the
        # substrate — so it stays pinned here while the cache is free to change.
        # 20 items: 16 agree (8 safe/safe, 8 unsafe/unsafe), 4 disagree.
        a = ["safe"] * 8 + ["unsafe"] * 8 + ["safe"] * 2 + ["unsafe"] * 2
        b = ["safe"] * 8 + ["unsafe"] * 8 + ["unsafe"] * 2 + ["safe"] * 2
        res = compute_agreement([a, b])
        # p_o = 16/20 = 0.8; marginals are 10/10 each -> p_e = 0.5;
        # kappa = (0.8 - 0.5) / (1 - 0.5) = 0.6 exactly.
        assert abs(res["kappa"] - 0.6) < 1e-12
        assert res["method"] == "cohen"

    def test_no_sklearn_or_statsmodels_imports(self):
        # Regression pin: neither dependency is in requirements.txt, so neither
        # may be imported anywhere in judges.py (not even lazily). Docstrings
        # may mention them; import statements may not.
        src = Path(judges.__file__).read_text(encoding="utf-8")
        for forbidden in ("import sklearn", "from sklearn", "import statsmodels", "from statsmodels"):
            assert forbidden not in src, f"judges.py must not contain {forbidden!r}"


# ---------------------------------------------------------------------------
# (f) SafetyJudge.judge + run_judge_agreement with a FAKE client (no network)
# ---------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeClient:
    """Routes chat_completion by model id to a canned raw verdict string."""

    def __init__(self, by_model: dict):
        self.by_model = by_model
        self.calls: list[tuple[str, list]] = []

    def chat_completion(self, messages, model, max_tokens, **kwargs):  # noqa: D401
        self.calls.append((model, messages))
        return _FakeCompletion(self.by_model[model])


class TestSafetyJudgeAndDriver:
    def test_judge_parses_fake_completion(self):
        client = _FakeClient({LLAMA_GUARD: "unsafe\nS2"})
        verdict = DEFAULT_JUDGES[0].judge(client, "p", "r")
        assert verdict == "unsafe"

    def test_judge_degrades_error_to_unclear(self):
        class _Boom:
            def chat_completion(self, **kw):
                raise RuntimeError("inference down")

        verdict = DEFAULT_JUDGES[0].judge(_Boom(), "p", "r")
        assert verdict == "unclear"

    def test_run_judge_agreement_contract_shape(self):
        # Both fake judges return 'safe' for every item -> perfect agreement.
        client = _FakeClient({LLAMA_GUARD: "safe", SHIELDGEMMA: "No"})
        corpus = [
            {"prompt": "p1", "response": "r1"},
            {"prompt": "p2", "response": "r2"},
            {"prompt": "p3", "response": "r3"},
        ]
        out = run_judge_agreement(corpus, client=client)

        # Top-level contract keys.
        assert set(out.keys()) == {"judges", "agreement", "certifier_pass", "elapsed_s"}
        assert len(out["judges"]) == 2
        for jr in out["judges"]:
            assert set(jr.keys()) == {"model", "counts", "verdict_vector"}
            assert set(jr["counts"].keys()) == {"safe", "unsafe", "unclear"}
            assert len(jr["verdict_vector"]) == 3

        # Agreement block.
        ag = out["agreement"]
        assert set(ag.keys()) == {
            "kappa", "method", "band", "invalid_reason", "n_judges", "n_items",
        }
        assert ag["n_items"] == 3
        assert ag["n_judges"] == 2

        # Both judges said safe everywhere -> RELIABLE -> certifier passes.
        assert ag["band"] == BAND_RELIABLE
        assert out["certifier_pass"] is True
        assert isinstance(out["elapsed_s"], float)

    def test_run_judge_agreement_disagreement_fails_certifier(self):
        # Llama-Guard says unsafe everywhere, ShieldGemma says safe everywhere.
        client = _FakeClient({LLAMA_GUARD: "unsafe\nS1", SHIELDGEMMA: "No"})
        corpus = [{"prompt": f"p{i}", "response": f"r{i}"} for i in range(4)]
        out = run_judge_agreement(corpus, client=client)
        assert out["judges"][0]["counts"]["unsafe"] == 4
        assert out["judges"][1]["counts"]["safe"] == 4
        # Total disagreement -> not RELIABLE -> certifier fails.
        assert out["certifier_pass"] is False

    def test_correlated_judge_failure_fails_certifier(self):
        # Both judges error on EVERY call (e.g. the inference provider is down):
        # every verdict degrades to 'unclear', the vectors agree perfectly, and
        # kappa is a vacuous 1.0 — the gate must mark the band INVALID and the
        # certifier must NOT pass.
        class _Down:
            def chat_completion(self, **kwargs):
                raise RuntimeError("inference provider down")

        corpus = [{"prompt": f"p{i}", "response": f"r{i}"} for i in range(5)]
        out = run_judge_agreement(corpus, client=_Down())
        for jr in out["judges"]:
            assert jr["counts"]["unclear"] == 5
        assert out["agreement"]["kappa"] == 1.0
        assert out["agreement"]["band"] == BAND_INVALID
        assert out["certifier_pass"] is False
        assert "unclear" in out["agreement"]["invalid_reason"]

    def test_corpus_text_not_echoed_in_contract(self):
        # The output contract must never carry raw prompt/response text.
        client = _FakeClient({LLAMA_GUARD: "safe", SHIELDGEMMA: "No"})
        secret_prompt = "SECRET_PROMPT_TOKEN"
        secret_response = "SECRET_RESPONSE_TOKEN"
        corpus = [{"prompt": secret_prompt, "response": secret_response}]
        out = run_judge_agreement(corpus, client=client)
        blob = repr(out)
        assert secret_prompt not in blob
        assert secret_response not in blob


# ---------------------------------------------------------------------------
# (g) load_judge_corpus across JSON array / {"items"} / JSONL
# ---------------------------------------------------------------------------

class TestLoadJudgeCorpus:
    def test_json_array(self, tmp_path):
        p = tmp_path / "corpus.json"
        p.write_text(
            '[{"prompt": "p1", "response": "r1"}, {"prompt": "p2", "response": "r2"}]',
            encoding="utf-8",
        )
        items = load_judge_corpus(str(p))
        assert len(items) == 2
        assert items[0] == {"prompt": "p1", "response": "r1"}

    def test_items_wrapper(self, tmp_path):
        p = tmp_path / "corpus.json"
        p.write_text(
            '{"items": [{"prompt": "p1", "response": "r1"}]}',
            encoding="utf-8",
        )
        items = load_judge_corpus(str(p))
        assert len(items) == 1
        assert items[0]["prompt"] == "p1"

    def test_jsonl(self, tmp_path):
        p = tmp_path / "corpus.jsonl"
        p.write_text(
            '{"prompt": "p1", "response": "r1"}\n{"prompt": "p2", "response": "r2"}\n',
            encoding="utf-8",
        )
        items = load_judge_corpus(str(p))
        assert len(items) == 2
        assert items[1]["response"] == "r2"

    def test_empty_file(self, tmp_path):
        p = tmp_path / "corpus.json"
        p.write_text("", encoding="utf-8")
        assert load_judge_corpus(str(p)) == []


# ---------------------------------------------------------------------------
# 2025/26 judge cohort parsers — Qwen3Guard-Gen + Granite Guardian
# ---------------------------------------------------------------------------

class TestParseQwen3Guard:
    """parse_qwen3guard keys off the 'Safety:' line of the structured block."""

    def test_safe(self):
        raw = "Safety: Safe\nCategories: None\nRefusal: No"
        assert judges.parse_qwen3guard(raw) == "safe"

    def test_unsafe(self):
        raw = "Safety: Unsafe\nCategories: Violent\nRefusal: No"
        assert judges.parse_qwen3guard(raw) == "unsafe"

    def test_controversial_maps_to_unclear(self):
        raw = "Safety: Controversial\nCategories: Legal\nRefusal: No"
        assert judges.parse_qwen3guard(raw) == "unclear"

    def test_case_insensitive_and_offset(self):
        raw = "Analysis complete.\nsafety: UNSAFE\nCategories: Weapons"
        assert judges.parse_qwen3guard(raw) == "unsafe"

    def test_missing_safety_line_is_unclear(self):
        assert judges.parse_qwen3guard("The response looks fine to me.") == "unclear"

    def test_empty_is_unclear(self):
        assert judges.parse_qwen3guard("") == "unclear"

    def test_prose_safety_word_does_not_count(self):
        # 'Safety' appearing without the label colon-format must not match.
        assert judges.parse_qwen3guard("Safety is important to consider.") == "unclear"


class TestParseGraniteGuardian:
    """parse_granite_guardian reads the <score> tag; yes = risk present = unsafe."""

    def test_yes_is_unsafe(self):
        assert judges.parse_granite_guardian("<score> yes </score>") == "unsafe"

    def test_no_is_safe(self):
        assert judges.parse_granite_guardian("<score> no </score>") == "safe"

    def test_think_preamble_ignored(self):
        raw = "<think>The response refuses the request, no harm.</think>\n<score> no </score>"
        assert judges.parse_granite_guardian(raw) == "safe"

    def test_tight_whitespace(self):
        assert judges.parse_granite_guardian("<score>yes</score>") == "unsafe"

    def test_case_insensitive(self):
        assert judges.parse_granite_guardian("<SCORE> Yes </SCORE>") == "unsafe"

    def test_missing_score_is_unclear(self):
        assert judges.parse_granite_guardian("yes") == "unclear"

    def test_empty_is_unclear(self):
        assert judges.parse_granite_guardian("") == "unclear"


class TestSotaJudgeCohort:
    """SOTA_JUDGES is wired with the right ids, families, and parsers."""

    def test_two_distinct_families(self):
        families = {j.family for j in judges.SOTA_JUDGES}
        assert families == {"qwen3guard", "granite-guardian"}

    def test_model_ids(self):
        ids = {j.model_id for j in judges.SOTA_JUDGES}
        assert ids == {"Qwen/Qwen3Guard-Gen-0.6B", "ibm-granite/granite-guardian-3.3-8b"}

    def test_build_fn_yields_conversation(self):
        for j in judges.SOTA_JUDGES:
            messages = j.build_fn("the prompt", "the response")
            assert messages == [
                {"role": "user", "content": "the prompt"},
                {"role": "assistant", "content": "the response"},
            ]


class TestGoldValidation:
    def test_classification_metrics_penalize_unclear(self):
        result = judges.classification_metrics(
            ["safe", "safe", "unsafe", "unsafe"],
            ["safe", "unclear", "unsafe", "safe"],
        )
        assert result["n_correct"] == 2
        assert result["accuracy"] == 0.5
        assert result["coverage"] == 0.75
        assert 0.0 < result["macro_f1"] < 1.0

    def test_selective_consensus_reports_coverage_and_accuracy(self):
        result = judges.selective_consensus_metrics(
            ["safe", "unsafe", "safe", "unsafe"],
            [
                ["safe", "unsafe", "safe", "safe"],
                ["safe", "unsafe", "unsafe", "safe"],
            ],
        )
        assert result["n_covered"] == 3
        assert result["n_correct"] == 2
        assert result["coverage"] == 0.75
        assert abs(result["accuracy"] - (2 / 3)) < 1e-12

    def test_cached_metrics_match_vectors_when_present(self):
        cached = json.loads(
            (_SPACE / "substrate" / "judge_results.json").read_text(encoding="utf-8")
        )
        corpus = json.loads(
            (_SPACE / "substrate" / "judge_corpus.json").read_text(encoding="utf-8")
        )["items"]
        expected = [item["expected"] for item in corpus]
        for report in cached["judges"]:
            if "metrics" not in report:
                continue
            recomputed = judges.classification_metrics(expected, report["verdict_vector"])
            assert abs(recomputed["accuracy"] - report["metrics"]["accuracy"]) < 1e-12
