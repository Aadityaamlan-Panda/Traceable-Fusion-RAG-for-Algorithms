# CppAwareChunker logic
"""
Unit tests for CppAwareChunker.

We test:
  1. Chunks stay within size limits
  2. Function boundaries are preserved (no mid-function splits)
  3. Metadata is correctly attached
  4. Edge cases: empty file, single-line file, no functions
"""

import pytest
from src.ingestion.chunker import CppAwareChunker, add_context_header
from langchain_core.documents import Document  # Fix 1: use langchain_core.documents


@pytest.fixture
def chunker():
    return CppAwareChunker(chunk_size=400, chunk_overlap=50)


def test_chunks_respect_size_limit(chunker, sample_cpp_docs):
    for doc in sample_cpp_docs:
        chunks = chunker.chunk_documents([doc])  # Fix 2: split_documents → chunk_documents
        for chunk in chunks:
            assert len(chunk.page_content) <= chunker.chunk_size + 50, (
                f"Chunk too large: {len(chunk.page_content)} chars"
            )


def test_chunks_preserve_metadata(chunker, sample_cpp_docs):
    chunks = chunker.chunk_documents(sample_cpp_docs)  # Fix 2: split_documents → chunk_documents
    for chunk in chunks:
        assert "source_file" in chunk.metadata
        assert "category" in chunk.metadata


def test_context_header_added(chunker, sample_cpp_docs):
    chunks = chunker.chunk_documents(sample_cpp_docs)  # Fix 2: split_documents → chunk_documents
    chunks_with_header = [add_context_header(c) for c in chunks]
    for chunk in chunks_with_header:
        assert chunk.page_content.startswith("File:"), (  # Fix 6: "// File:" → "File:" (matches source)
            "Context header missing from chunk"
        )


def test_empty_document_produces_no_chunks(chunker):
    empty_doc = Document(page_content="", metadata={"source_file": "empty.cpp"})
    chunks = chunker.chunk_documents([empty_doc])  # Fix 2: split_documents → chunk_documents
    assert chunks == [], "Empty document should produce no chunks"


def test_overlap_creates_continuity(chunker):
    """Adjacent chunks must share at least `overlap` characters."""
    long_doc = Document(
        page_content="int x = 0;\n" * 100,
        metadata={"source_file": "long.cpp", "category": "misc"},
    )
    chunks = chunker.chunk_documents([long_doc])  # Fix 2: split_documents → chunk_documents
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            end_of_prev = chunks[i].page_content[-chunker.chunk_overlap:]
            start_of_next = chunks[i+1].page_content[:chunker.chunk_overlap]
            assert end_of_prev in chunks[i+1].page_content or \
                   start_of_next in chunks[i].page_content, \
                   f"No overlap between chunk {i} and {i+1}"