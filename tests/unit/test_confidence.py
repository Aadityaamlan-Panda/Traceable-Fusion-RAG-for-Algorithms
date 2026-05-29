# Confidence scoring formulas
"""
Unit tests for ConfidenceScorer and pre_retrieval_gate.

The scorer must be deterministic and purely functional —
no LLM calls, just score/gate logic.
"""

import pytest
from unittest.mock import MagicMock
from src.guardrails.confidence import ConfidenceScorer, pre_retrieval_gate


def _make_chunk(rerank_score: float):
    """Helper: create a mock retrieved chunk with a .rerank_score attribute."""
    chunk = MagicMock()
    chunk.rerank_score = rerank_score
    return chunk


@pytest.fixture
def scorer():
    # Fix 3: ConfidenceScorer requires groq_api_key and known_files
    return ConfidenceScorer(groq_api_key="test-key", known_files=set())


def test_score_retrieval_high_scores(scorer):
    # Fix 4: score_single doesn't exist; use score_retrieval with mock chunks
    chunks = [_make_chunk(0.95), _make_chunk(0.90), _make_chunk(0.88)]
    score = scorer.score_retrieval(chunks)
    assert score > 0.85, f"High-score chunks should yield high retrieval score, got {score}"


def test_score_retrieval_low_scores(scorer):
    # Fix 4: use score_retrieval with low-score chunks
    chunks = [_make_chunk(0.20), _make_chunk(0.18), _make_chunk(0.15)]
    score = scorer.score_retrieval(chunks)
    assert score < 0.30, f"Low-score chunks should yield low retrieval score, got {score}"


def test_gate_passes_on_high_scores():
    # Fix 5: pre_retrieval_gate takes chunk objects with .rerank_score, returns tuple[bool, str]
    high_chunks = [_make_chunk(s) for s in [0.92, 0.87, 0.91, 0.85, 0.89]]
    passed, _ = pre_retrieval_gate(high_chunks)
    assert passed is True


def test_gate_blocks_on_low_scores():
    # Fix 5: low rerank scores (below 0.25 threshold) should block
    low_chunks = [_make_chunk(s) for s in [0.10, 0.08, 0.12, 0.09, 0.11]]
    passed, _ = pre_retrieval_gate(low_chunks)
    assert passed is False


def test_gate_uses_top_score_not_mean():
    # One very high score + many low scores → should still pass (gate checks top chunk only)
    mixed_chunks = [_make_chunk(0.95)] + [_make_chunk(s) for s in [0.10, 0.11, 0.09, 0.12]]
    passed, _ = pre_retrieval_gate(mixed_chunks)
    assert passed is True


def test_gate_blocks_empty_chunks():
    passed, msg = pre_retrieval_gate([])
    assert passed is False
    assert len(msg) > 0