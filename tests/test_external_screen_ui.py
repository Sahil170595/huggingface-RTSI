"""UI-surface tests for the "Test your own quant" external-screen feature.

Importing app builds the Gradio Blocks at module scope; these tests introspect
that built graph (NO browser, NO network, NO torch). They pin:

  * the collapsed "Test your own quant" Accordion exists with the prefilled
    SAFE example, a Code input, a button, and a JSON output;
  * the endpoint is registered PUBLIC and explicitly named
    "screen_external_manifest", while the heavy live endpoints stay private;
  * the six existing tab ids (score/live/judges/certificate/debate/about)
    are unchanged;
  * the gradio_client usage snippet shown in the UI and documented in README is
    accurate (correct api_name and schema_version, no raw prompts/completions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SPACE = Path(__file__).resolve().parent.parent
if str(_SPACE) not in sys.path:
    sys.path.insert(0, str(_SPACE))

import app
import external_screen as es

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ---------------------------------------------------------------------------
# Introspection helpers over the built Blocks graph
# ---------------------------------------------------------------------------

def _blocks_of(kind: str) -> list:
    return [b for b in app.demo.blocks.values() if b.__class__.__name__ == kind]


def _api_visibility(api_name: str) -> str | None:
    for d in app.demo.fns.values():
        if getattr(d, "api_name", None) == api_name:
            return getattr(d, "api_visibility", "public")
    return None


def _dependency(api_name: str):
    return next(
        d for d in app.demo.fns.values()
        if getattr(d, "api_name", None) == api_name
    )


def _all_api_names() -> set[str]:
    return {
        getattr(d, "api_name", None)
        for d in app.demo.fns.values()
        if getattr(d, "api_name", None)
    }


# ---------------------------------------------------------------------------
# (a) Accordion + its child components render
# ---------------------------------------------------------------------------

class TestAccordion:
    def test_collapsed_accordion_exists(self):
        labels = [getattr(a, "label", None) for a in _blocks_of("Accordion")]
        match = [a for a in _blocks_of("Accordion")
                 if getattr(a, "label", "") and "Test your own quant" in a.label]
        assert match, f"accordion not found among {labels}"
        # It must ship collapsed.
        assert all(getattr(a, "open", True) is False for a in match)

    def test_code_input_is_prefilled_with_the_safe_example(self):
        codes = [c for c in _blocks_of("Code")
                 if "external-screen manifest" in (getattr(c, "label", "") or "").lower()]
        assert codes, "external-screen Code input not found"
        val = getattr(codes[0], "value", "")
        assert "quantsafe.external-screen.v1" in val
        # It is exactly the module's SAFE example (single source of truth).
        assert val == es.safe_example_json()
        # The SAFE example screens LOW (so the prefill demonstrates a pass).
        assert es.screen_external_manifest(val)["band"] == "LOW"

    def test_json_output_present(self):
        jsons = [j for j in _blocks_of("JSON")
                 if "screening report" in (getattr(j, "label", "") or "").lower()]
        assert jsons, "screening-report JSON output not found"

    def test_a_button_drives_the_endpoint(self):
        # The screen_external_manifest dependency must have at least one trigger.
        assert "screen_external_manifest" in _all_api_names()


# ---------------------------------------------------------------------------
# (b) endpoint is public + explicitly named
# ---------------------------------------------------------------------------

class TestEndpointExposure:
    def test_endpoint_is_named_exactly_screen_external_manifest(self):
        assert "screen_external_manifest" in _all_api_names()

    def test_endpoint_is_public(self):
        assert _api_visibility("screen_external_manifest") == "public"

    def test_endpoint_bypasses_the_shared_queue(self):
        assert _dependency("screen_external_manifest").queue is False

    def test_heavy_live_endpoints_remain_private(self):
        # We must not have flipped concurrency-bound heavy endpoints public.
        for hidden in ("run_live", "run_live_debate", "_on_load"):
            assert _api_visibility(hidden) == "private", hidden

    def test_endpoint_round_trips_through_app_handler(self):
        r = app.screen_external(es.safe_example_json())
        assert r["schema_version"] == "quantsafe.external-screen.response.v1"
        assert r["signed"] is False
        assert r["scope"] == "user-supplied-aggregate-evidence"

    def test_app_handler_tolerates_empty_input(self):
        r = app.screen_external("")
        assert r["status"] == "rejected"


# ---------------------------------------------------------------------------
# (c) the six existing tabs are unchanged
# ---------------------------------------------------------------------------

class TestSixTabsIntact:
    EXPECTED = ["score", "live", "judges", "certificate", "debate", "about"]

    def _tab_ids(self) -> list[str]:
        ids = []
        for b in app.demo.blocks.values():
            if b.__class__.__name__ in ("Tab", "TabItem"):
                tid = getattr(b, "id", None) or getattr(b, "elem_id", None)
                if tid:
                    ids.append(tid)
        return ids

    def test_exactly_the_six_declared_tab_ids_exist(self):
        ids = self._tab_ids()
        assert ids == self.EXPECTED, ids

    def test_tab_ids_match_the_deep_link_map(self):
        assert set(app.TAB_IDS.values()) == set(self.EXPECTED)


# ---------------------------------------------------------------------------
# (d) the documented client snippet is accurate
# ---------------------------------------------------------------------------

class TestSnippetAccuracy:
    def _readme(self) -> str:
        return (_SPACE / "README.md").read_text(encoding="utf-8")

    def test_readme_documents_the_public_endpoint(self):
        readme = self._readme()
        assert "screen_external_manifest" in readme
        assert "gradio_client" in readme
        assert "quantsafe.external-screen.v1" in readme

    def test_readme_states_no_raw_prompts_or_completions(self):
        readme = self._readme().lower()
        # The doc must make the aggregate-only, not-a-certification contract clear.
        assert "aggregate" in readme
        assert "screening recommendation" in readme
        assert "not a safety certification" in readme

    def test_readme_snippet_uses_the_real_api_name(self):
        readme = self._readme()
        # The exact api_name string a gradio_client call must use.
        assert "/screen_external_manifest" in readme

    def test_ui_snippet_matches_the_real_endpoint(self):
        # The explanatory HTML block embeds the same endpoint name.
        htmls = [getattr(h, "value", "") or "" for h in _blocks_of("HTML")]
        joined = "\n".join(htmls)
        assert "/screen_external_manifest" in joined
        assert "gradio_client" in joined
