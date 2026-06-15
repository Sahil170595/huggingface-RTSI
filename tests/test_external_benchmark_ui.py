from __future__ import annotations

import app


def test_external_benchmark_renders_minicpm_as_separate_cross_check():
    html = app._build_external_benchmark_html()

    assert "OpenBMB" in html
    assert "MiniCPM4.1-8B" in html
    assert "74.5%" in html
    assert "general-reasoning moderation cross-check" in html
    assert "three specialist guards" in html


def test_live_debate_uses_hybrid_three_model_cohort():
    assert app.LIVE_DEBATE_MODELS == [
        "Qwen/Qwen3-8B",
        "openbmb/MiniCPM4.1-8B",
        "HuggingFaceTB/SmolLM3-3B",
    ]
