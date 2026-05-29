"""
CONCEPT: Chunking transforms whole files into retrievable units.

Strategy per language:
  cpp      — brace-matched function extraction, fallback to LangChain CPP splitter
  python   — indent-tracked def/class extraction, fallback to LangChain PYTHON splitter
  java     — brace-matched method extraction, fallback to LangChain JAVA splitter
  markdown — LangChain MARKDOWN splitter (preserve heading hierarchy)
  default  — generic RecursiveCharacterTextSplitter

chunk_overlap bridges adjacent chunks so a call-site and its definition
don't end up in completely separate chunks with no shared context.
"""

import re
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
import logging

logger = logging.getLogger(__name__)


class CppAwareChunker:
    """
    Multi-language code chunker. Despite the class name (kept for back-compat)
    it handles C++, Python, Java, and Markdown.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        # LangChain language-aware splitters
        def _make(lang):
            return RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        self._splitters = {
            "cpp":      _make(Language.CPP),
            "python":   _make(Language.PYTHON),
            "java":     _make(Language.JAVA),
            "markdown": _make(Language.MARKDOWN),
            "default":  RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        ),
        }

    # ------------------------------------------------------------------
    # Function / block extractors
    # ------------------------------------------------------------------

    def _extract_functions_cpp(self, content: str) -> list:
        """Brace-matched C++ function extraction."""
        blacklist = {'if', 'while', 'for', 'switch', 'else', 'catch', 'try'}
        pattern = re.compile(
            r'(?:^|\n)'
            r'(?:\/\*[\s\S]*?\*\/\s*)?'
            r'(?:\/\/[^\n]*\n\s*)*'
            r'(?:template\s*<[^>]*>\s*)?'
            r'(?:[\w:*&<>\[\]\s]+)\s+'
            r'(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{',
            re.MULTILINE
        )
        return self._brace_extract(content, pattern, blacklist)

    def _extract_functions_java(self, content: str) -> list:
        """Brace-matched Java method extraction."""
        blacklist = {'if', 'while', 'for', 'switch', 'else', 'catch', 'try', 'new'}
        pattern = re.compile(
            r'(?:public|private|protected|static|final|\s)+'
            r'[\w<>\[\]]+\s+'
            r'(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+\s*)?\{',
            re.MULTILINE
        )
        return self._brace_extract(content, pattern, blacklist)

    def _brace_extract(self, content: str, pattern, blacklist: set) -> list:
        """Shared brace-counting extractor for C-family languages."""
        functions = []
        for match in pattern.finditer(content):
            name = match.group(1)
            if name in blacklist:
                continue
            start       = match.start()
            brace_count = 0
            end         = start
            for i, ch in enumerate(content[start:], start):
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            if end > start:
                body = content[start:end]
                if 50 < len(body) < self.chunk_size * 2:
                    functions.append((name, body))
        return functions

    def _extract_functions_python(self, content: str) -> list:
        """Indent-tracked Python def/class extraction."""
        functions = []
        lines     = content.splitlines(keepends=True)
        i         = 0
        while i < len(lines):
            m = re.match(r'^(def|class)\s+(\w+)', lines[i])
            if m:
                name  = m.group(2)
                start = i
                i += 1
                while i < len(lines):
                    l = lines[i]
                    if l.strip() == "":
                        i += 1
                        continue
                    if l[0] in (' ', '\t'):
                        i += 1
                        continue
                    break
                body = "".join(lines[start:i])
                if 50 < len(body) < self.chunk_size * 2:
                    functions.append((name, body))
            else:
                i += 1
        return functions

    # ------------------------------------------------------------------
    # Splitter + extractor dispatch
    # ------------------------------------------------------------------

    def _get_splitter(self, language: str):
        return self._splitters.get(language, self._splitters["default"])

    def _extract_functions(self, content: str, language: str) -> list:
        if language == "cpp":
            return self._extract_functions_cpp(content)
        if language == "python":
            return self._extract_functions_python(content)
        if language == "java":
            return self._extract_functions_java(content)
        return []  # markdown and others: skip function-level, use splitter directly

    # ------------------------------------------------------------------
    # Main chunking
    # ------------------------------------------------------------------

    def chunk_document(self, doc: Document) -> list:
        content  = doc.page_content
        language = doc.metadata.get("language", "cpp")
        splitter = self._get_splitter(language)
        chunks   = []

        functions = self._extract_functions(content, language)

        if len(functions) >= 2:
            # Chunk at function/method boundary (preferred)
            for func_name, func_body in functions:
                if len(func_body) > self.chunk_size:
                    for i, sub in enumerate(splitter.split_text(func_body)):
                        chunks.append(Document(
                            page_content=sub,
                            metadata={**doc.metadata,
                                      "chunk_type": "function_sub",
                                      "function_name": func_name,
                                      "chunk_index": i},
                        ))
                else:
                    chunks.append(Document(
                        page_content=func_body,
                        metadata={**doc.metadata,
                                  "chunk_type": "function",
                                  "function_name": func_name,
                                  "chunk_index": 0},
                    ))
        else:
            # Fallback: language-aware text splitting
            for i, text in enumerate(splitter.split_text(content)):
                chunks.append(Document(
                    page_content=text,
                    metadata={**doc.metadata,
                              "chunk_type": "language_split",
                              "chunk_index": i},
                ))

        return chunks

    def chunk_documents(self, documents: list) -> list:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        if all_chunks:
            avg = sum(len(c.page_content) for c in all_chunks) / len(all_chunks)
            logger.info(
                f"Chunked {len(documents)} docs → {len(all_chunks)} chunks "
                f"(avg {avg:.0f} chars)"
            )
        return all_chunks


def add_context_header(chunk: Document) -> Document:
    """
    Prepend a metadata header to each chunk before embedding.
    The header is part of the embedded text so vectors capture
    both context and content — significantly improves retrieval.
    Language tag is included so the model knows what it's reading.
    """
    meta   = chunk.metadata
    header = (
        f"File: {meta.get('source', 'unknown')} | "
        f"Language: {meta.get('language', 'unknown')} | "
        f"Category: {meta.get('category', 'unknown')} | "
        f"Algorithm: {meta.get('algorithm_name', 'unknown')} | "
        f"Function: {meta.get('function_name', 'N/A')} | "
        f"Time Complexity: {meta.get('time_complexity', 'unknown')}\n\n"
    )
    return Document(page_content=header + chunk.page_content, metadata=chunk.metadata)
