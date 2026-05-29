# src/config.py
"""
CONCEPT: Centralised config prevents hardcoded values scattered everywhere.
Using Pydantic's BaseSettings auto-reads from .env files.
"""
import os
# Disable ChromaDB's PostHog telemetry before ChromaDB is imported anywhere.
# Cannot go in .env — Pydantic Settings uses extra="forbid" and rejects
# unknown keys. Setting via os.environ bypasses Pydantic entirely.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from pydantic_settings import BaseSettings
from pathlib import Path

# ---------------------------------------------------------------------------
# Multi-repo knowledge base
# Each entry: repo_url, local_path, language
#
# Languages supported: "cpp" | "python" | "java" | "markdown"
#
# Repos included and why:
#
#   TheAlgorithms/C-Plus-Plus  — original corpus, C++, 370+ files
#   TheAlgorithms/Python       — same family, identical dir structure,
#                                Python, ~900 files
#   TheAlgorithms/Java         — same family, identical dir structure,
#                                Java, ~600 files, zero code changes needed
#   keon/algorithms            — Python, better docstrings than TheAlgorithms,
#                                cleaner complexity annotations, ~200 files
#   williamfiset/Algorithms    — Java, best graph/DP quality on GitHub,
#                                competitive-programmer-written, ~300 files
#   trekhleb/javascript-algorithms — JS code + rich markdown explanations
#                                    per algorithm; indexed as "markdown" so
#                                    /explain retrieves human-written text
#                                    not just code comments
#
# To add more: append a dict here. Nothing else needs to change.
# ---------------------------------------------------------------------------
ALGO_REPOS = [
    {
        "url":      "https://github.com/TheAlgorithms/C-Plus-Plus",
        "path":     "./data/raw/C-Plus-Plus",
        "language": "cpp",
    },
    {
        "url":      "https://github.com/TheAlgorithms/Python",
        "path":     "./data/raw/Python",
        "language": "python",
    },
    {
        "url":      "https://github.com/TheAlgorithms/Java",
        "path":     "./data/raw/Java",
        "language": "java",
    },
    {
        "url":      "https://github.com/keon/algorithms",
        "path":     "./data/raw/keon-algorithms",
        "language": "python",
    },
    {
        "url":      "https://github.com/williamfiset/Algorithms",
        "path":     "./data/raw/williamfiset-algorithms",
        "language": "java",
    },
    {
        "url":      "https://github.com/trekhleb/javascript-algorithms",
        "path":     "./data/raw/trekhleb-js-algorithms",
        "language": "markdown",   # index the .md explanation docs, not the JS code
    },
]


class Settings(BaseSettings):
    # API Keys
    groq_api_key: str
    cohere_api_key: str
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "false"
    langchain_project: str = "Traceable Fusion RAG for Algorithms"

    # LLM Config
    primary_llm: str = "llama-3.3-70b-versatile"   # Groq
    fallback_llm: str = "command-a-03-2025"          # Cohere Command A+
    llm_temperature: float = 0.0
    max_tokens: int = 2048

    # Embedding Config
    embedding_model: str = "embed-english-v3.0"
    embedding_dim: int = 1024

    # Retrieval Config
    retrieval_k: int = 8
    rerank_top_n: int = 5
    similarity_threshold: float = 0.35

    # Confidence Thresholds
    high_confidence: float = 0.80
    medium_confidence: float = 0.60
    low_confidence: float = 0.40

    # Paths — kept for back-compat; pipeline uses ALGO_REPOS above
    chroma_db_path: str = "./data/chroma_db"
    cpp_repo_path:  str = "./data/raw/C-Plus-Plus"
    cpp_repo_url:   str = "https://github.com/TheAlgorithms/C-Plus-Plus"

    # Logging
    log_level: str = "INFO"

    # Chunking
    chunk_size:    int = 1000
    chunk_overlap: int = 200

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
