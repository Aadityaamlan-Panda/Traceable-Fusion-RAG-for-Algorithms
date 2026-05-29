"""
CONCEPT: GitPython clones repos programmatically. We then walk
the directory tree, loading source files as LangChain Documents.

We enrich each Document with metadata by:
1. Parsing the directory structure (sorting/, graph/, dynamic_programming/)
2. Extracting complexity annotations from comments
3. Identifying algorithm type from filename patterns

Supported languages: cpp | python | java | markdown
Adding a new language = add one entry to _LANGUAGE_CONFIG below.
"""

import re
from pathlib import Path
from typing import Optional, Callable
import git
from langchain_core.documents import Document
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repo management
# ---------------------------------------------------------------------------

def clone_or_update_repo(repo_url: str, local_path: str) -> str:
    """Clone if absent, pull if present. Returns local_path."""
    path = Path(local_path)
    if path.exists() and (path / ".git").exists():
        logger.info(f"Repo exists at {local_path}, pulling latest...")
        git.Repo(local_path).remotes.origin.pull()
        logger.info("Pull complete.")
    else:
        logger.info(f"Cloning {repo_url} → {local_path}...")
        path.mkdir(parents=True, exist_ok=True)
        git.Repo.clone_from(repo_url, local_path, depth=1)
        logger.info("Clone complete.")
    return local_path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def extract_complexity(content: str, ctype: str) -> Optional[str]:
    """Extract O() complexity notation from comments."""
    match = re.search(rf'(?i){ctype}[^:]*:\s*(O\([^)]+\))', content)
    if match:
        return match.group(1)
    matches = re.findall(r'O\([^)]+\)', content)
    return matches[0] if matches else None


def _common_metadata(file_path: Path, repo_root: Path, language: str) -> dict:
    relative_path = str(file_path.relative_to(repo_root)).replace("\\", "/")
    parts    = relative_path.split("/")
    category = parts[0] if len(parts) > 1 else "misc"
    algo_name = file_path.stem.replace("_", " ").replace("-", " ").title()
    return {
        "source":         relative_path,
        "category":       category,
        "algorithm_name": algo_name,
        "file_name":      file_path.name,
        "language":       language,
    }


# ---------------------------------------------------------------------------
# C++ metadata
# ---------------------------------------------------------------------------

def _fn_signatures_cpp(content: str) -> list:
    pattern = r'(?:^|\n)[\w\s\*&<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?{'
    blacklist = {'if', 'while', 'for', 'switch', 'else', 'catch', 'try'}
    return [m for m in re.findall(pattern, content) if m not in blacklist][:5]


def extract_metadata_cpp(file_path: Path, repo_root: Path) -> dict:
    meta = _common_metadata(file_path, repo_root, "cpp")
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        content = ""
    meta.update({
        "time_complexity":     extract_complexity(content, "time") or "unknown",
        "space_complexity":    extract_complexity(content, "space") or "unknown",
        "has_class":           str(bool(re.search(r'\bclass\b|\bstruct\b', content))),
        "has_template":        str(bool(re.search(r'\btemplate\b', content))),
        "has_main":            str(bool(re.search(r'\bmain\s*\(', content))),
        "function_signatures": ", ".join(_fn_signatures_cpp(content)),
    })
    return meta


# ---------------------------------------------------------------------------
# Python metadata
# ---------------------------------------------------------------------------

def _fn_signatures_python(content: str) -> list:
    return re.findall(r'^(?:def|class)\s+(\w+)', content, re.MULTILINE)[:5]


def extract_metadata_python(file_path: Path, repo_root: Path) -> dict:
    meta = _common_metadata(file_path, repo_root, "python")
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        content = ""
    meta.update({
        "time_complexity":     extract_complexity(content, "time") or "unknown",
        "space_complexity":    extract_complexity(content, "space") or "unknown",
        "has_class":           str(bool(re.search(r'^\s*class\s+\w+', content, re.MULTILINE))),
        "has_template":        "False",
        "has_main":            str(bool(re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', content))),
        "function_signatures": ", ".join(_fn_signatures_python(content)),
    })
    return meta


# ---------------------------------------------------------------------------
# Java metadata
# ---------------------------------------------------------------------------

def _fn_signatures_java(content: str) -> list:
    # Match: [modifiers] returnType methodName(
    pattern = r'(?:public|private|protected|static|final|\s)+[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+\s*)?\{'
    blacklist = {'if', 'while', 'for', 'switch', 'else', 'catch', 'try', 'new'}
    return [m for m in re.findall(pattern, content) if m not in blacklist][:5]


def extract_metadata_java(file_path: Path, repo_root: Path) -> dict:
    meta = _common_metadata(file_path, repo_root, "java")
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        content = ""
    meta.update({
        "time_complexity":     extract_complexity(content, "time") or "unknown",
        "space_complexity":    extract_complexity(content, "space") or "unknown",
        "has_class":           str(bool(re.search(r'\bclass\s+\w+', content))),
        "has_template":        "False",  # Java uses generics, not templates
        "has_main":            str(bool(re.search(r'public\s+static\s+void\s+main', content))),
        "function_signatures": ", ".join(_fn_signatures_java(content)),
    })
    return meta


# ---------------------------------------------------------------------------
# Markdown metadata  (trekhleb-style: one .md per algorithm with O() info)
# ---------------------------------------------------------------------------

def extract_metadata_markdown(file_path: Path, repo_root: Path) -> dict:
    meta = _common_metadata(file_path, repo_root, "markdown")
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        content = ""
    meta.update({
        "time_complexity":     extract_complexity(content, "time") or "unknown",
        "space_complexity":    extract_complexity(content, "space") or "unknown",
        "has_class":           "False",
        "has_template":        "False",
        "has_main":            "False",
        "function_signatures": "",
    })
    return meta


def _is_useful_markdown(file_path: Path, content: str) -> bool:
    """
    Filter out boilerplate READMEs (root-level, CI badges, etc.).
    Keep only algorithm-specific docs that contain complexity info or
    substantial technical content (>400 chars, has a heading).
    """
    # Skip root-level and shallow READMEs that are just repo intros
    depth = len(file_path.parts)
    if depth <= 2 and file_path.name.upper() == "README.MD":
        return False
    if len(content.strip()) < 400:
        return False
    # Must have at least one markdown heading
    if not re.search(r'^#{1,3}\s+\w+', content, re.MULTILINE):
        return False
    return True


# ---------------------------------------------------------------------------
# Dispatch table — add new languages here only
# ---------------------------------------------------------------------------

_LANGUAGE_CONFIG: dict = {
    "cpp": {
        "extensions":   {".cpp", ".h", ".hpp", ".cc"},
        "extract_meta": extract_metadata_cpp,
        "min_size":     50,
        "extra_filter": None,
    },
    "python": {
        "extensions":   {".py"},
        "extract_meta": extract_metadata_python,
        "min_size":     50,
        "extra_filter": None,
    },
    "java": {
        "extensions":   {".java"},
        "extract_meta": extract_metadata_java,
        "min_size":     50,
        "extra_filter": None,
    },
    "markdown": {
        "extensions":   {".md", ".markdown"},
        "extract_meta": extract_metadata_markdown,
        "min_size":     400,
        "extra_filter": _is_useful_markdown,   # extra quality gate
    },
}

_SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".github",
    "test", "tests", "spec", "specs", ".idea", "build", "dist",
}


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_repository(repo_path: str, language: str) -> list:
    """
    Load all source files from a single cloned repo.
    Returns a list of LangChain Documents.
    """
    if language not in _LANGUAGE_CONFIG:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Choose from: {list(_LANGUAGE_CONFIG)}"
        )

    cfg        = _LANGUAGE_CONFIG[language]
    extensions = cfg["extensions"]
    meta_fn    = cfg["extract_meta"]
    min_size   = cfg["min_size"]
    extra_filt = cfg["extra_filter"]
    repo_root  = Path(repo_path)
    documents  = []

    source_files = [
        f for f in repo_root.rglob("*")
        if f.is_file()
        and f.suffix.lower() in extensions
        and not any(skip in f.parts for skip in _SKIP_DIRS)
    ]

    logger.info(f"[{language}] Found {len(source_files)} files in {repo_path}")

    for file_path in tqdm(source_files, desc=f"Loading {language} files"):
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if len(content.strip()) < min_size:
                continue
            if extra_filt and not extra_filt(file_path, content):
                continue
            metadata = meta_fn(file_path, repo_root)
            documents.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue

    logger.info(f"[{language}] Loaded {len(documents)} documents")
    return documents


def load_all_repositories(algo_repos: list) -> list:
    """
    Clone/update and load every repo in ALGO_REPOS.
    Returns the combined document list.
    """
    all_docs = []
    for repo_cfg in algo_repos:
        url      = repo_cfg["url"]
        path     = repo_cfg["path"]
        language = repo_cfg["language"]
        label    = url.split("/")[-1]

        print(f"\n  📦 [{language.upper():8s}] {label}")
        clone_or_update_repo(url, path)
        docs = load_repository(path, language)
        all_docs.extend(docs)
        print(f"     → {len(docs)} documents loaded")

    print(f"\n  📚 Total across all repos: {len(all_docs)} documents")
    return all_docs


# ---------------------------------------------------------------------------
# Back-compat alias
# ---------------------------------------------------------------------------

def load_cpp_repository(repo_path: str) -> list:
    """Kept for backward compatibility."""
    return load_repository(repo_path, language="cpp")


def get_category_summary(documents: list) -> dict:
    from collections import Counter
    return dict(Counter(d.metadata["category"] for d in documents).most_common())
