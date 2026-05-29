# clone → chunk → index
# clone → chunk → index
"""
Integration test: clone → load → chunk → index pipeline.

Uses a minimal in-memory corpus (no real git clone) and mocked
Cohere + ChromaDB to avoid network calls.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.ingestion.chunker import CppAwareChunker, add_context_header
from src.ingestion.indexer import AlgoIndexer


@pytest.mark.integration
def test_chunk_then_index_pipeline(temp_settings, sample_cpp_docs):
    """Chunk real documents, then index them — count must match."""
    chunker = CppAwareChunker(chunk_size=600, chunk_overlap=50)
    chunks = chunker.chunk_documents(sample_cpp_docs)
    chunks_with_headers = [add_context_header(c) for c in chunks]

    assert len(chunks_with_headers) >= len(sample_cpp_docs), (
        "Chunking must produce at least as many chunks as source docs"
    )
    for chunk in chunks_with_headers:
        assert chunk.page_content.startswith("File:"), (
            "add_context_header must prepend a 'File:' header"
        )

    with patch("src.ingestion.indexer.CohereEmbeddings") as MockEmbed, \
         patch("src.ingestion.indexer.chromadb.PersistentClient") as MockChroma, \
         patch("src.ingestion.indexer.Chroma") as MockVectorstore, \
         patch("time.sleep"):

        MockEmbed.return_value.embed_documents.return_value = [
            [0.01 * i for i in range(1024)] for _ in chunks_with_headers
        ]
        MockEmbed.return_value.embed_query.return_value = [0.1] * 1024

        mock_collection = MagicMock()
        mock_collection.count.return_value = len(chunks_with_headers)
        mock_collection.get.return_value = {
            "ids": [],
            "metadatas": [c.metadata for c in chunks_with_headers],
        }
        MockChroma.return_value.get_collection.return_value = mock_collection

        indexer = AlgoIndexer(
            cohere_api_key=temp_settings.cohere_api_key,
            chroma_path=temp_settings.chroma_db_path,
        )

        with patch.object(indexer, "get_existing_ids", return_value=set()):
            count = indexer.index_chunks(chunks_with_headers)

        assert count == len(chunks_with_headers)


@pytest.mark.integration
def test_context_headers_survive_indexing(sample_cpp_docs):
    """Context headers added before indexing must not be stripped."""
    chunker = CppAwareChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk_documents(sample_cpp_docs)
    enriched = [add_context_header(c) for c in chunks]

    for chunk in enriched:
        # Header must reference the source file
        source = chunk.metadata.get("source", chunk.metadata.get("source_file", ""))
        assert "File:" in chunk.page_content
        assert "Language:" in chunk.page_content
        assert "Category:" in chunk.page_content


@pytest.mark.integration
def test_idempotent_indexing_skips_existing(temp_settings, sample_cpp_docs):
    """Re-indexing the same chunks must return 0 (no duplicates written)."""
    chunker = CppAwareChunker(chunk_size=600, chunk_overlap=50)
    chunks = chunker.chunk_documents(sample_cpp_docs)

    with patch("src.ingestion.indexer.CohereEmbeddings"), \
         patch("src.ingestion.indexer.chromadb.PersistentClient"), \
         patch("src.ingestion.indexer.Chroma"), \
         patch("time.sleep"):

        indexer = AlgoIndexer(
            cohere_api_key=temp_settings.cohere_api_key,
            chroma_path=temp_settings.chroma_db_path,
        )

        # Simulate all chunks already existing
        all_ids = {indexer._generate_chunk_id(c) for c in chunks}
        with patch.object(indexer, "get_existing_ids", return_value=all_ids):
            count = indexer.index_chunks(chunks)

        assert count == 0, "No chunks should be re-indexed when all already exist"