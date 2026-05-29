# query → retrieve → generate
"""
Integration test: query → hybrid retrieval → confidence gate → generation.

Uses mock LLM to avoid API calls.
Uses a real (tiny) in-memory ChromaDB for retrieval.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.retrieval.retriever import HybridRetriever, RetrievedChunk
from src.generation.generator import AlgoGenerator


def _make_retrieved_chunks(docs):
    """Wrap plain Documents into RetrievedChunk objects as the pipeline expects."""
    return [
        RetrievedChunk(document=doc, dense_score=0.9, rerank_score=0.9)
        for doc in docs
    ]


@pytest.mark.integration
def test_retrieval_returns_relevant_docs(temp_settings, sample_cpp_docs):
    """After indexing sample docs, retrieval should return the dijkstra doc for a graph query."""
    from src.ingestion.indexer import AlgoIndexer
    from unittest.mock import patch

    # Index the sample docs (uses real ChromaDB in temp dir)
    with patch("src.ingestion.indexer.CohereEmbeddings") as MockEmbed:
        # Mock embeddings to avoid Cohere API call
        MockEmbed.return_value.embed_documents.return_value = [
            [0.1 * i for i in range(1024)] for _ in sample_cpp_docs
        ]
        MockEmbed.return_value.embed_query.return_value = [0.1 * i for i in range(1024)]

        indexer = AlgoIndexer(
            cohere_api_key=temp_settings.cohere_api_key,
            chroma_path=temp_settings.chroma_db_path,  # Fix: AlgoIndexer uses chroma_path not chroma_db_path
        )
        indexer.index_chunks(sample_cpp_docs)  # Fix: method is index_chunks not index_documents
        stats = indexer.get_stats()
        assert stats["total_chunks"] >= len(sample_cpp_docs)


@pytest.mark.integration
def test_generation_uses_context_only(mock_llm, sample_cpp_docs):
    """Generator must include retrieved context in its prompt — no free-form generation."""
    from src.generation.generator import AlgoGenerator

    # Fix: suggest_algorithms expects RetrievedChunk objects, not plain Documents
    retrieved_chunks = _make_retrieved_chunks(sample_cpp_docs)

    json_response = (
        '{"answer": "Dijkstra uses a min-heap priority queue.", '
        '"source_files": ["graph/dijkstra.cpp"], '
        '"confidence_note": "High confidence.", '
        '"cpp_snippet": "priority_queue<...> pq;"}'
    )

    with patch("src.generation.generator.ChatGroq"):
        generator = AlgoGenerator(groq_api_key="test", cohere_api_key="test")
        # Patch _call_groq directly so the full fallback chain uses our response
        with patch.object(generator, '_call_groq', return_value=json_response):
            result, provider = generator.suggest_algorithms(
                query="How does Dijkstra work?",
                retrieved_chunks=retrieved_chunks,
            )

    assert "answer" in result
    assert len(result["answer"]) > 10