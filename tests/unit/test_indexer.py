# Embedding & ChromaDB writes
# Embedding & ChromaDB writes
"""
Unit tests for AlgoIndexer.

We test:
  1. index_chunks returns correct count of newly indexed chunks
  2. Already-indexed chunks are skipped (idempotency)
  3. get_stats returns expected structure
  4. _generate_chunk_id produces deterministic, unique IDs
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.ingestion.indexer import AlgoIndexer


def _make_doc(content: str, source: str = "test.cpp", category: str = "graph") -> Document:
    return Document(
        page_content=content,
        metadata={"source_file": source, "category": category, "chunk_index": 0},
    )


@pytest.fixture
def mock_indexer(tmp_path):
    """AlgoIndexer with mocked Cohere + ChromaDB so no API calls are made."""
    with patch("src.ingestion.indexer.CohereEmbeddings") as MockEmbed, \
         patch("src.ingestion.indexer.chromadb.PersistentClient") as MockChroma, \
         patch("src.ingestion.indexer.Chroma") as MockVectorstore:

        MockEmbed.return_value.embed_documents.return_value = [[0.1] * 1024]
        MockEmbed.return_value.embed_query.return_value = [0.1] * 1024

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        MockChroma.return_value.get_collection.return_value = mock_collection
        MockChroma.return_value.get_or_create_collection.return_value = mock_collection

        MockVectorstore.return_value.add_documents.return_value = None

        indexer = AlgoIndexer(
            cohere_api_key="test-key",
            chroma_path=str(tmp_path / "chroma"),
        )
        # Expose mocks for test inspection
        indexer._mock_vectorstore = MockVectorstore.return_value
        indexer._mock_collection = mock_collection
        yield indexer


def test_generate_chunk_id_is_deterministic(mock_indexer):
    doc = _make_doc("int x = 1;", "algo.cpp")
    id1 = mock_indexer._generate_chunk_id(doc)
    id2 = mock_indexer._generate_chunk_id(doc)
    assert id1 == id2, "Chunk IDs must be deterministic for the same content"


def test_generate_chunk_id_differs_for_different_content(mock_indexer):
    doc_a = _make_doc("int x = 1;", "a.cpp")
    doc_b = _make_doc("int y = 2;", "b.cpp")
    assert mock_indexer._generate_chunk_id(doc_a) != mock_indexer._generate_chunk_id(doc_b)


def test_index_chunks_returns_count(mock_indexer):
    docs = [_make_doc(f"int x{i} = {i};", f"file{i}.cpp") for i in range(3)]
    with patch.object(mock_indexer, "get_existing_ids", return_value=set()), \
         patch("time.sleep"):
        count = mock_indexer.index_chunks(docs)
    assert count == 3


def test_index_chunks_skips_already_indexed(mock_indexer):
    doc = _make_doc("int x = 1;", "already.cpp")
    existing_id = mock_indexer._generate_chunk_id(doc)

    with patch.object(mock_indexer, "get_existing_ids", return_value={existing_id}), \
         patch("time.sleep"):
        count = mock_indexer.index_chunks([doc])

    assert count == 0, "Already-indexed chunk must be skipped"


def test_index_chunks_deduplicates_within_batch(mock_indexer):
    """Two identical docs in the same call should only be indexed once."""
    doc = _make_doc("int dup = 1;", "dup.cpp")

    with patch.object(mock_indexer, "get_existing_ids", return_value=set()), \
         patch("time.sleep"):
        count = mock_indexer.index_chunks([doc, doc])

    assert count == 1


def test_get_stats_returns_expected_keys(mock_indexer):
    mock_indexer._mock_collection.count.return_value = 5
    mock_indexer._mock_collection.get.return_value = {
        "ids": ["a", "b"],
        "metadatas": [{"category": "graph"}, {"category": "sorting"}],
    }
    with patch.object(
        mock_indexer.chroma_client, "get_collection",
        return_value=mock_indexer._mock_collection
    ):
        stats = mock_indexer.get_stats()

    assert "total_chunks" in stats
    assert "categories" in stats
    assert "db_path" in stats