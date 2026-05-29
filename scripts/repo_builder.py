from pathlib import Path

# Root project folder
ROOT = Path("Traceable Fusion RAG for Algorithms")

# Directory structure
directories = [
    "src/ingestion",
    "src/retrieval",
    "src/generation",
    "src/transparency",
    "src/guardrails",
    "src/ui",
    "data/raw",
    "data/chroma_db",
    "tests",
    "eval",
]

# Files to create
files = [
    # Root files
    ".env",
    ".env.example",
    "requirements.txt",
    "setup.py",
    "README.md",

    # Ingestion
    "src/ingestion/__init__.py",
    "src/ingestion/loader.py",
    "src/ingestion/chunker.py",
    "src/ingestion/indexer.py",

    # Retrieval
    "src/retrieval/__init__.py",
    "src/retrieval/retriever.py",
    "src/retrieval/reranker.py",

    # Generation
    "src/generation/__init__.py",
    "src/generation/prompt_builder.py",
    "src/generation/generator.py",

    # Transparency
    "src/transparency/__init__.py",
    "src/transparency/tracer.py",
    "src/transparency/visualiser.py",

    # Guardrails
    "src/guardrails/__init__.py",
    "src/guardrails/confidence.py",
    "src/guardrails/hallucination_guard.py",

    # UI
    "src/ui/__init__.py",
    "src/ui/app.py",
    "src/ui/components.py",

    # Tests
    "tests/test_ingestion.py",
    "tests/test_retrieval.py",
    "tests/test_e2e.py",

    # Eval
    "eval/ragas_eval.py",
]

def create_project_structure():
    # Create root folder
    ROOT.mkdir(exist_ok=True)

    # Create directories
    for directory in directories:
        dir_path = ROOT / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

    # Create files
    for file in files:
        file_path = ROOT / file
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            file_path.touch()
            print(f"📄 Created file: {file_path}")
        else:
            print(f"⚠️ File already exists: {file_path}")

    print("\n✅ Algo-RAG project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()