"""
End-to-end tests: fire real queries through the full pipeline.

Marked @pytest.mark.e2e — skipped automatically unless GROQ_API_KEY is set.
Requires real API keys in .env.

Run with:
    pytest tests/e2e/ -v -m e2e
"""

import os
import pytest
from dotenv import load_dotenv
from unittest.mock import patch

load_dotenv()

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"),
        reason="Real API keys required — set GROQ_API_KEY in .env",
    ),
]


# ── Helper ─────────────────────────────────────────────────────────────────

def _run_pipeline(query: str, mode: str = "suggest") -> dict | None:
    """
    Run a single query through the full pipeline and return the result dict.

    run_query() renders to the terminal and returns None.
    We intercept the result by patching render_algorithm_results, which is
    the last call made before run_query() exits in suggest mode.
    """
    from src.ui.app import AlgoRAGApp

    app = AlgoRAGApp()
    app.initialize()

    captured = {}

    with patch(
        "src.ui.app.render_algorithm_results",
        side_effect=lambda r: captured.update(r),
    ):
        app.run_query(query, mode=mode)

    return captured if captured else None


# ── Tests ──────────────────────────────────────────────────────────────────

def test_dijkstra_query_returns_valid_answer():
    """Full pipeline: real Dijkstra query → real retrieval → real generation."""
    result = _run_pipeline("How does Dijkstra's shortest path algorithm work?")

    assert result is not None, (
        "Pipeline returned no result — likely refused by the pre-retrieval gate"
    )

    # ── Real response shape from suggest_algorithms ──
    # Keys: understanding, algorithms, comparison, caveats, _confidence
    assert "understanding" in result, (
        f"Missing 'understanding' key. Got: {list(result.keys())}"
    )
    assert "algorithms" in result, (
        f"Missing 'algorithms' key. Got: {list(result.keys())}"
    )
    assert isinstance(result["algorithms"], list), "'algorithms' should be a list"
    assert len(result["algorithms"]) > 0, "No algorithms returned"

    # Each algorithm entry must have name and code
    first = result["algorithms"][0]
    assert "name" in first, f"Algorithm entry missing 'name'. Got: {list(first.keys())}"
    assert "code" in first, f"Algorithm entry missing 'code'. Got: {list(first.keys())}"
    assert len(first["code"]) > 20, "Code snippet is suspiciously short"

    # Dijkstra should be mentioned somewhere in the result
    result_text = str(result).lower()
    assert "dijkstra" in result_text, (
        "Expected 'dijkstra' to appear somewhere in the result"
    )

    # Confidence check
    confidence_report = result.get("_confidence")
    assert confidence_report is not None, "Missing _confidence report"
    assert confidence_report.overall_confidence > 0.5, (
        f"Confidence too low: {confidence_report.overall_confidence:.2f} "
        f"(level={confidence_report.level})"
    )
    assert confidence_report.source_verified is True, (
        "Source files cited by LLM were not found in the index"
    )


def test_hallucination_guard_fires_on_nonsense_query():
    """
    A query for a non-existent algorithm must be refused by the
    pre-retrieval gate — not hallucinated by the LLM.

    When the gate fires, run_query() prints a refusal panel and returns
    without ever calling render_algorithm_results, so _run_pipeline
    returns None.
    """
    result = _run_pipeline(
        "Explain the FizzBuzzinator quantum sorting algorithm "
        "invented by Dr. Fictional in 2087"
    )

    assert result is None, (
        f"Hallucination guard did not fire — pipeline returned: {str(result)[:200]}"
    )


def test_explain_mode_returns_explanation():
    """explain mode should produce a long-form text explanation, not a JSON dict."""
    from src.ui.app import AlgoRAGApp
    from src.transparency.tracer import PipelineTracer

    app = AlgoRAGApp()
    app.initialize()

    captured = {}
    original_finish = PipelineTracer.finish_trace

    def capture_finish(self, final_answer=None):
        if final_answer:
            captured.update(final_answer)
        return original_finish(self, final_answer=final_answer)

    with patch.object(PipelineTracer, "finish_trace", capture_finish):
        app.run_query("Explain binary search", mode="explain")

    assert "explanation" in captured, (
        f"explain mode should produce 'explanation' key. Got: {list(captured.keys())}"
    )
    assert len(captured["explanation"]) > 100, "Explanation is suspiciously short"


def test_pipeline_uses_fallback_on_groq_failure():
    """
    When Groq fails, _call_with_fallback should catch the exception and
    transparently retry with Cohere Command A+.

    Patch _call_groq (not suggest_algorithms) so the real fallback chain
    in _call_with_fallback can catch the error and invoke Cohere.
    """
    from src.ui.app import AlgoRAGApp
    from src.generation.generator import AlgoGenerator

    app = AlgoRAGApp()
    app.initialize()

    def groq_always_fails(self, chain_input, prompt):
        raise RuntimeError("rate limit exceeded — 429")

    captured = {}

    with patch.object(AlgoGenerator, "_call_groq", groq_always_fails):
        with patch(
            "src.ui.app.render_algorithm_results",
            side_effect=lambda r: captured.update(r),
        ):
            app.run_query("How does merge sort work?")

    assert captured, (
        "Cohere fallback should have returned a result after Groq failed"
    )
    assert "algorithms" in captured, (
        f"Fallback result missing 'algorithms'. Got keys: {list(captured.keys())}"
    )
    assert len(captured["algorithms"]) > 0, "Fallback returned empty algorithms list"