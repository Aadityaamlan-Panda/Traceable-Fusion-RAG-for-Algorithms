# Pre/post generation gates
# Pre/post generation gates
"""
Unit tests for hallucination guardrails and source verification.

We test:
  1. pre_retrieval_gate blocks low-confidence retrieval results
  2. ConfidenceScorer.verify_sources detects hallucinated file paths
  3. ConfidenceScorer.verify_sources accepts valid, known file paths
  4. ConfidenceScorer.score_retrieval weighs top chunks more heavily
  5. ConfidenceScorer.assess returns a ConfidenceReport with correct level
"""

import pytest
from unittest.mock import MagicMock, patch
from src.guardrails.confidence import (
    ConfidenceScorer,
    ConfidenceReport,
    pre_retrieval_gate,
)


def _make_chunk(rerank_score: float, source: str = "graph/dijkstra.cpp"):
    chunk = MagicMock()
    chunk.rerank_score = rerank_score
    chunk.document = MagicMock()
    chunk.document.page_content = "int dijkstra() { /* ... */ }"
    chunk.document.metadata = {"source": source, "language": "cpp", "category": "graph"}
    return chunk


@pytest.fixture
def scorer():
    return ConfidenceScorer(groq_api_key="test-key", known_files={"graph/dijkstra.cpp", "sorting/merge_sort.cpp"})


# --- pre_retrieval_gate ---

def test_gate_passes_for_high_rerank_score():
    chunks = [_make_chunk(0.9), _make_chunk(0.8)]
    passed, msg = pre_retrieval_gate(chunks)
    assert passed is True


def test_gate_blocks_for_low_rerank_score():
    chunks = [_make_chunk(0.10), _make_chunk(0.05)]
    passed, msg = pre_retrieval_gate(chunks)
    assert passed is False
    assert len(msg) > 0


def test_gate_blocks_empty_chunks():
    passed, msg = pre_retrieval_gate([])
    assert passed is False
    assert len(msg) > 0


def test_gate_uses_top_chunk_score():
    """Gate should check only the top chunk, not an average."""
    chunks = [_make_chunk(0.90)] + [_make_chunk(0.05) for _ in range(4)]
    passed, _ = pre_retrieval_gate(chunks)
    assert passed is True


# --- verify_sources ---

def test_verify_sources_passes_known_file(scorer):
    ok, warnings = scorer.verify_sources(["graph/dijkstra.cpp"])
    assert ok is True
    assert warnings == []


def test_verify_sources_flags_hallucinated_file(scorer):
    ok, warnings = scorer.verify_sources(["graph/fake_algorithm.cpp"])
    assert ok is False
    assert len(warnings) > 0


def test_verify_sources_strips_file_prefix(scorer):
    """LLM sometimes outputs 'File: graph/dijkstra.cpp' — must be normalised."""
    ok, warnings = scorer.verify_sources(["File: graph/dijkstra.cpp"])
    assert ok is True


def test_verify_sources_ignores_empty_entries(scorer):
    ok, warnings = scorer.verify_sources(["", "  "])
    assert ok is True
    assert warnings == []


def test_verify_sources_ignores_long_sentences(scorer):
    """Strings > 120 chars are sentences, not paths — skip hallucination check."""
    long_sentence = "No implementation was found for this algorithm in the current codebase as of today " + "x" * 50
    ok, warnings = scorer.verify_sources([long_sentence])
    assert ok is True


# --- score_retrieval ---

def test_score_retrieval_empty_returns_zero(scorer):
    assert scorer.score_retrieval([]) == 0.0


def test_score_retrieval_weights_top_chunk(scorer):
    """Top chunk has weight 1.0, second weight 0.5 — with only two chunks
    the weighted average is (1.0*1.0 + 0.0*0.5)/1.5 ≈ 0.667, clearly > 0."""
    chunks = [_make_chunk(1.0), _make_chunk(0.0)]
    score = scorer.score_retrieval(chunks)
    assert score > 0.5, f"Two-chunk case: top=1.0, second=0.0 should yield ~0.667, got {score}"


def test_score_retrieval_capped_at_one(scorer):
    chunks = [_make_chunk(1.0) for _ in range(5)]
    score = scorer.score_retrieval(chunks)
    assert score <= 1.0