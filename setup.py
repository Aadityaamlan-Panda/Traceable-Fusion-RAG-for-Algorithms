from setuptools import setup, find_packages

setup(
    name="Traceable Fusion RAG for Algorithms",
    version="0.1.0",
    description="A RAG system for querying algorithm implementations across multiple languages",
    author="",
    author_email="",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core LangChain Stack
        "langchain==0.3.25",
        "langchain-community==0.3.24",
        "langchain-chroma==0.2.4",
        "langchain-groq==0.3.2",
        "langchain-cohere==0.4.4",
        # Vector Database
        "chromadb==1.0.12",
        # Embeddings / Reranking
        "cohere==5.15.0",
        # LLM APIs
        "groq==0.15.0",
        # Retrieval
        "rank-bm25==0.2.2",
        "numpy>=1.26",              # used directly in retriever.py (import numpy as np)
        # Terminal UI
        "rich==13.9.4",
        "textual==1.0.0",
        # Utilities
        "python-dotenv==1.0.1",
        "pydantic==2.10.6",
        "pydantic-settings==2.8.1",
        "gitpython==3.1.44",
        "tqdm==4.67.1",
        "httpx==0.28.1",
    ],
    extras_require={
        # pip install algo-rag[tracing]
        # Only needed when LANGCHAIN_TRACING_V2=true in .env.
        # LangChain imports langsmith internally when tracing is enabled;
        # your source code never imports it directly.
        "tracing": [
            "langsmith==0.2.10",
        ],
        # pip install algo-rag[eval]
        "eval": [
            "ragas==0.2.14",
            "datasets==3.2.0",
        ],
        # pip install algo-rag[dev]
        "dev": [
            "pytest==8.3.4",
            "pytest-asyncio==0.25.3",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            # scripts/verify_setup.py is removed from here because scripts/ has
            # no __init__.py and is not a Python package. Run it directly with:
            #   python scripts/verify_setup.py
            "algo-rag=ui.app:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)