
from pathlib import Path

# Base structure
structure = {
    "tests": {
        "conftest.py": "# Shared fixtures (settings, sample docs, mock LLM)\n",
        "unit": {
            "test_chunker.py": "# CppAwareChunker logic\n",
            "test_indexer.py": "# Embedding & ChromaDB writes\n",
            "test_retriever.py": "# Hybrid retrieval scoring\n",
            "test_confidence.py": "# Confidence scoring formulas\n",
            "test_guardrails.py": "# Pre/post generation gates\n",
        },
        "integration": {
            "test_ingestion_pipeline.py": "# clone → chunk → index\n",
            "test_query_pipeline.py": "# query → retrieve → generate\n",
        },
        "e2e": {
            "test_full_system.py": "# Known question → expected answer shape\n",
        },
    }
}


def create_structure(base_path: Path, tree: dict):
    """
    Recursively create folders and files from a nested dictionary.
    """
    for name, content in tree.items():
        current_path = base_path / name

        if isinstance(content, dict):
            # Create directory
            current_path.mkdir(parents=True, exist_ok=True)
            create_structure(current_path, content)
        else:
            # Create file with starter content
            current_path.write_text(content, encoding="utf-8")
            print(f"Created file: {current_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    create_structure(project_root, structure)

    print("\n✅ Test directory structure generated successfully!")

