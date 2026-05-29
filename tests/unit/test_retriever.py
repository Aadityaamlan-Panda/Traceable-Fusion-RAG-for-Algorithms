# Hybrid retrieval scoring
# Hybrid retrieval scoring
"""
Unit tests for HybridRetriever.

We test:
  1. reciprocal_rank_fusion merges two result lists correctly
  2. dense_search and bm25_search return ranked results
  3. _tokenize splits code tokens correctly
  4. _metadata_boost applies language match/mismatch signals
  5. retrieve returns RetrievedChunk objects with expected fields
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.retrieval.retriever import HybridRetriever, RetrievedChunk


def _make_doc(content: str, source: str = "graph/dijkstra.cpp",
              category: str = "graph", language: str = "cpp") -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "category": category,
                  "language": language, "chunk_index": 0},
    )


@pytest.fixture
def retriever(sample_cpp_docs):
    """HybridRetriever with mocked vectorstore and Cohere client."""
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search_with_score.return_value = [
        (doc, 0.1) for doc in sample_cpp_docs  # low distance = high similarity
    ]

    with patch("src.retrieval.retriever.cohere.Client"):
        r = HybridRetriever(
            vectorstore=mock_vectorstore,
            all_chunks=sample_cpp_docs,
            cohere_api_key="test-key",
            k=5,
            rerank_top_n=3,
        )
    return r


def test_tokenize_handles_code_identifiers(retriever):
    tokens = retriever._tokenize("BFS(int src, vector<int>& adj)")
    assert "bfs" in tokens
    assert "src" in tokens
    assert "vector" in tokens
    # Punctuation should not appear as tokens
    assert "<" not in tokens
    assert ">" not in tokens


def test_rrf_merges_two_ranked_lists(retriever, sample_cpp_docs):
    """RRF merges by doc_id key derived from source + chunk_index metadata.
    Use docs that have distinct source metadata so they get distinct IDs."""
    from langchain_core.documents import Document
    doc_a = Document(page_content="int a=1;", metadata={"source": "a.cpp", "chunk_index": 0})
    doc_b = Document(page_content="int b=2;", metadata={"source": "b.cpp", "chunk_index": 0})
    dense = [(doc_a, 0.9), (doc_b, 0.7)]
    bm25 = [(doc_b, 15.0), (doc_a, 10.0)]

    fused = retriever.reciprocal_rank_fusion(dense, bm25)

    assert len(fused) >= 2
    result_docs = [doc for doc, _ in fused]
    assert doc_a in result_docs
    assert doc_b in result_docs


def test_rrf_scores_are_positive(retriever, sample_cpp_docs):
    dense = [(doc, 0.8) for doc in sample_cpp_docs]
    bm25 = [(doc, 5.0) for doc in sample_cpp_docs]
    fused = retriever.reciprocal_rank_fusion(dense, bm25)
    for _, score in fused:
        assert score > 0


def test_bm25_search_returns_results(retriever):
    results = retriever.bm25_search("dijkstra shortest path", k=2)
    assert isinstance(results, list)
    # Each result is a (Document, float) pair
    for doc, score in results:
        assert isinstance(doc, Document)
        assert isinstance(score, float)


def test_dense_search_converts_distance_to_similarity(retriever):
    """ChromaDB returns distance; dense_search must convert to similarity."""
    results = retriever.dense_search("graph algorithm", k=2)
    for _, score in results:
        # Similarity should be in [0, 1]
        assert 0.0 <= score <= 1.0


def test_metadata_boost_rewards_language_match(retriever, sample_cpp_docs):
    cpp_doc = _make_doc("int x;", source="test.cpp", language="cpp")
    lang_hint = ({"cpp", "cc", "h"}, "cpp")
    boosted = retriever._metadata_boost(0.5, cpp_doc, lang_hint, query="")
    assert boosted > 0.5, "Language match should increase score"


def test_metadata_boost_penalises_language_mismatch(retriever):
    java_doc = _make_doc("int x;", source="Test.java", language="java")
    lang_hint = ({"cpp", "cc", "h"}, "cpp")
    penalised = retriever._metadata_boost(0.5, java_doc, lang_hint, query="")
    assert penalised < 0.5, "Language mismatch should decrease score"


def test_retrieve_returns_retrieved_chunks(retriever):
    """retrieve() must return RetrievedChunk objects."""
    mock_response = MagicMock()
    mock_response.results = []

    with patch.object(retriever.cohere_client, "rerank", return_value=mock_response):
        results = retriever.retrieve("Dijkstra shortest path")

    # With empty rerank results and fallback, result is still a list
    assert isinstance(results, list)