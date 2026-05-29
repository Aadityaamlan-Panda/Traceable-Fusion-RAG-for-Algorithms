"""
CONCEPT: Hybrid Search = Dense (semantic) + Sparse (BM25 keyword) search

Dense retrieval: captures semantic meaning ("fast sort" → "quicksort")
Sparse retrieval: captures exact terms ("BFS" → finds "BFS", not "DFS")

Reciprocal Rank Fusion (RRF) merges both result lists by rank position,
not raw score (which can't be directly compared across methods).

Then Cohere Reranker applies a cross-encoder to re-score the top-N combined
results with full pairwise (query, document) attention — far more accurate
than embedding dot-product.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_chroma import Chroma
import cohere
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with its scores — for transparency."""
    document: Document
    dense_score: float = 0.0        # Cosine similarity (0-1)
    bm25_score: float = 0.0         # BM25 term frequency score  
    rrf_score: float = 0.0          # Reciprocal Rank Fusion score
    rerank_score: float = 0.0       # Cohere cross-encoder score
    final_rank: int = 0
    retrieval_method: str = ""      # "dense", "bm25", "hybrid"


class HybridRetriever:
    """
    Retrieves algorithm chunks using hybrid dense + sparse search,
    followed by cross-encoder reranking.
    """
    
    RRF_K = 60  # RRF constant — standard value, controls rank impact
    
    def __init__(self,
                 vectorstore: Chroma,
                 all_chunks: list[Document],
                 cohere_api_key: str,
                 k: int = 8,
                 rerank_top_n: int = 5):
        
        self.vectorstore = vectorstore
        self.cohere_client = cohere.Client(cohere_api_key)
        self.k = k
        self.rerank_top_n = rerank_top_n
        
        # Build BM25 index from all chunks
        # CONCEPT: BM25 (Best Match 25) is a probabilistic ranking function
        # based on term frequency and inverse document frequency.
        # It's the backbone of Elasticsearch.
        logger.info("Building BM25 index...")
        self.all_chunks = all_chunks
        tokenized = [self._tokenize(c.page_content) for c in all_chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built over {len(all_chunks)} chunks")
    
    def _tokenize(self, text: str) -> list[str]:
        """
        CONCEPT: BM25 works on tokens. We keep identifiers intact
        (quicksort, BFS, O(n)) by splitting on whitespace and punctuation
        but keeping alphanumeric runs together.
        """
        import re
        # Split on spaces/punctuation but keep alphanumeric runs
        tokens = re.findall(r'[a-zA-Z0-9_]+', text.lower())
        return tokens
    
    def dense_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Semantic vector search via ChromaDB."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: 1 - (distance/2) → 0 to 1 range
        return [(doc, 1.0 - score/2.0) for doc, score in results]
    
    # Common filler words that hurt BM25 precision — stripped from queries only,
    # not from the index (we still want to find them in code comments if needed).
    _BM25_STOPWORDS = {
        "the", "a", "an", "for", "of", "in", "to", "and", "or", "is", "are",
        "was", "be", "with", "that", "this", "it", "its", "from", "as", "on",
        "by", "at", "do", "get", "give", "me", "my", "i", "we", "you", "show",
        "return", "provide", "generate", "create", "write", "find", "make",
        "code", "implement", "implementation", "algorithm", "alg", "function",
        "program", "example", "using", "use", "can", "how", "what", "which",
    }

    def bm25_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Keyword-based BM25 search with soft language boost."""
        query_tokens = [t for t in self._tokenize(query) if t not in self._BM25_STOPWORDS]
        if not query_tokens:
            query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Soft language boost: if query contains an explicit language hint,
        # multiply BM25 scores for matching-language docs by 1.4.
        # This keeps BM25 diversity (Python/README still show up) but
        # surfaces the right language higher in the candidate list.
        lang_hint = self._extract_language_hint(query)
        if lang_hint is not None:
            _, lang_tag = lang_hint
            for idx, chunk in enumerate(self.all_chunks):
                if chunk.metadata.get("language", "") == lang_tag:
                    scores[idx] *= 1.4

        top_indices = np.argsort(scores)[::-1][:k]
        return [
            (self.all_chunks[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]
    
    def reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[Document, float]],
        bm25_results: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        """
        CONCEPT: RRF merges ranked lists by position, not score.
        
        Formula: RRF(d) = Σ 1/(k + rank(d))
        
        Why not just add scores? Dense scores are cosine similarities (0-1),
        BM25 scores are unbounded. They're incomparable directly.
        RRF only uses rank position — robust and scale-invariant.
        
        Standard k=60 was empirically found to work well across many tasks.
        """
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        
        # Process dense results
        for rank, (doc, _) in enumerate(dense_results, 1):
            doc_id = doc.metadata.get("source", "") + str(doc.metadata.get("chunk_index", 0))
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (self.RRF_K + rank)
            docs[doc_id] = doc
        
        # Process BM25 results
        for rank, (doc, _) in enumerate(bm25_results, 1):
            doc_id = doc.metadata.get("source", "") + str(doc.metadata.get("chunk_index", 0))
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (self.RRF_K + rank)
            docs[doc_id] = doc
        
        # Sort by RRF score (descending)
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [(docs[id_], scores[id_]) for id_ in sorted_ids[:self.k * 2]]
    
    # Maps query language keywords to file extensions and metadata language tags
    _LANGUAGE_HINTS: dict[str, tuple[set, str]] = {
        "cpp":    ({"cpp", "cc", "h", "hpp"}, "cpp"),
        "c++":    ({"cpp", "cc", "h", "hpp"}, "cpp"),
        "python": ({"py"},                    "python"),
        "java":   ({"java"},                  "java"),
        "js":     ({"js", "ts"},              "js"),
        "javascript": ({"js", "ts"},          "js"),
    }

    def _extract_language_hint(self, query: str) -> tuple[set, str] | None:
        """
        Detect an explicit language constraint in the query.
        Returns (extensions_set, language_tag) or None.
        e.g. "Prim's MST in Java" → ({"java"}, "java")
        """
        q = query.lower()
        for keyword, (exts, lang) in self._LANGUAGE_HINTS.items():
            # Match "in java", "java implementation", "using java", etc.
            if (f" in {keyword}" in q or
                f"in {keyword} " in q or
                f"{keyword} implementation" in q or
                f"using {keyword}" in q or
                q.startswith(keyword + " ")):
                return exts, lang
        return None

    @staticmethod
    def _query_tokens(query: str) -> list[str]:
        """Extract meaningful lowercase tokens from the query for symbolic matching."""
        import re
        raw = re.findall(r'[a-zA-Z0-9]+', query.lower())
        noise = {"code", "implement", "write", "give", "show", "the", "in",
                 "a", "an", "for", "using", "with", "all", "pairs", "path",
                 "algorithm", "java", "python", "cpp", "c", "me", "please"}
        return [t for t in raw if t not in noise and len(t) > 1]

    def _symbolic_name_boost(self, query: str, doc) -> float:
        """
        Filename / algorithm-name overlap bonus.

        The cross-encoder scores SteinerTree.java and FloydWarshallSolver.java
        identically when both chunks contain Floyd-Warshall code, because it
        sees only the chunk body text.  This bonus rewards files whose *name*
        directly matches the query tokens, and penalizes files whose name
        belongs to a different algorithm family not mentioned in the query.

          +0.12 per query token found in filename (capped +0.20)
          -0.10 per foreign algorithm name in filename (capped -0.20)
        """
        import re
        source = doc.metadata.get("source", "")
        filename = source.split("/")[-1].rsplit(".", 1)[0].lower() if source else ""
        algo_name = doc.metadata.get("algorithm_name", "").lower()
        name_text = filename + " " + algo_name

        # Split PascalCase / camelCase: FloydWarshallSolver -> floyd warshall solver
        name_tokens = set(re.findall(r'[a-z]+',
                                     re.sub(r'([A-Z])', r' \1', name_text).lower()))

        q_tokens = set(self._query_tokens(query))

        # Positive: query tokens found in filename/algo-name
        boost = min(0.20, len(q_tokens & name_tokens) * 0.12)

        # Negative: filename contains a known algorithm family NOT in the query
        _KNOWN_ALGO_NAMES = {
            "steiner", "dijkstra", "bellman", "prim", "kruskal", "kosaraju",
            "tarjan", "topological", "astar", "bfs", "dfs", "kmp",
            "quicksort", "mergesort", "heapsort", "lcs", "knapsack",
        }
        foreign = (name_tokens & _KNOWN_ALGO_NAMES) - q_tokens
        penalty = min(0.20, len(foreign) * 0.10)

        return boost - penalty

    # Algorithms commonly used as subroutines inside unrelated files.
    # When a chunk's filename/category does NOT match the query but the code
    # calls one of these as a helper, the cross-encoder gives it a high score
    # because the algorithm name appears in the code.  We detect this pattern
    # and apply a subroutine penalty so only files *about* the algorithm rank high.
    _SUBROUTINE_NAMES = {
        "binary_search", "binarysearch", "quicksort", "quick_sort",
        "mergesort", "merge_sort", "heapsort", "heap_sort",
        "lower_bound", "upper_bound",
    }

    def _subroutine_penalty(self, query: str, doc) -> float:
        """
        Return a negative score adjustment when a chunk clearly uses the queried
        algorithm as a subroutine rather than being the primary implementation.

        Detection: the query names algorithm A, but the chunk's category/filename
        belongs to a completely different domain (e.g. query='binary search',
        chunk is maths/perfect_square.py which calls binary search internally).

        Penalty: -0.30, enough to push subroutine-use chunks below true
        implementations without hard-excluding them (they may still be useful
        if no primary implementation is found).
        """
        import re
        q_tokens = set(self._query_tokens(query))
        source = doc.metadata.get("source", "").lower()
        category = doc.metadata.get("category", "").lower()
        filename = source.split("/")[-1].rsplit(".", 1)[0].lower() if source else ""

        for name in self._SUBROUTINE_NAMES:
            parts = set(name.split("_"))
            if parts <= q_tokens:           # query IS about this algorithm
                # Check if the chunk's home file is NOT about this algorithm
                if not any(p in filename for p in parts):
                    # File name doesn't match — likely a subroutine use
                    return -0.30
        return 0.0

    def _readme_bonus(self, doc) -> float:
        """
        Small bonus for README / markdown explanation chunks.

        Code-only chunks retrieved for questions like "how does X work?" or
        "when should I use X?" often score well on the algorithm name but badly
        on the conceptual content.  README chunks that explain the algorithm in
        prose are highly relevant for such questions and deserve a slight lift.
        """
        source = doc.metadata.get("source", "").lower()
        lang = doc.metadata.get("language", "").lower()
        algo = doc.metadata.get("algorithm_name", "").lower()
        if lang == "markdown" or source.endswith(".md") or "readme" in source:
            return 0.08
        return 0.0

    def _metadata_boost(
        self,
        rerank_score: float,
        doc,
        lang_hint: tuple | None,
        query: str = "",
    ) -> float:
        """
        Metadata-aware score adjustment combining four signals:

        1. Language match / mismatch  (from explicit query hint, e.g. "in Java")
             +0.15  exact language match
             -0.25  language mismatch

        2. Filename / algorithm-name overlap  (_symbolic_name_boost)
             +0.12 per overlapping query token (capped +0.20)
             -0.10 per foreign algorithm name in filename (capped -0.20)

        3. Subroutine-use penalty  (_subroutine_penalty)
             -0.30 when file uses queried algorithm as helper, not primary subject
             Fixes: binary search query retrieving perfect_square.py

        4. README/explanation bonus  (_readme_bonus)
             +0.08 for markdown/README chunks
             Helps context_recall: prose explanation chunks contain the conceptual
             content that ground-truth answers are based on.
        """
        score = rerank_score

        # Signal 1: language
        if lang_hint is not None:
            exts, lang_tag = lang_hint
            source = doc.metadata.get("source", "").lower()
            doc_lang = doc.metadata.get("language", "").lower()
            ext = source.rsplit(".", 1)[-1] if "." in source else ""
            if ext in exts or doc_lang == lang_tag:
                score = min(1.0, score + 0.15)
            elif ext and ext not in exts and doc_lang and doc_lang != lang_tag:
                score = max(0.0, score - 0.25)

        # Signal 2: symbolic filename/algo overlap
        # NOTE: We intentionally do NOT clamp to 1.0 here so that symbolic bonuses
        # can push scores above 1.0 for sorting purposes.  The display layer caps
        # the shown percentage at 100%; what matters is relative ordering.
        if query:
            score = max(0.0, score + self._symbolic_name_boost(query, doc))

        # Signal 3: subroutine-use penalty
        if query:
            score = max(0.0, score + self._subroutine_penalty(query, doc))

        # Signal 4: README/explanation bonus
        score += self._readme_bonus(doc)

        return score

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Document, float]],
    ) -> list[RetrievedChunk]:
        """
        CONCEPT: Reranking uses a cross-encoder model.

        Embedding models encode query and document INDEPENDENTLY then compare vectors.
        Cross-encoders see BOTH query and document together — much richer attention.

        Pipeline:
        1. Retrieve many candidates fast (embedding search)
        2. Rerank top candidates precisely (cross-encoder)
        3. Apply metadata boosts for explicit language/type constraints
           so "in Java" queries strongly prefer .java files.

        Cohere Rerank 3.5 is state-of-the-art and has a free tier.
        """
        if not candidates:
            return []

        lang_hint = self._extract_language_hint(query)
        docs_text = [doc.page_content for doc, _ in candidates]

        try:
            response = self.cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=docs_text,
                top_n=min(self.rerank_top_n * 3, len(candidates)),
            )

            results = []
            for item in response.results:
                doc, rrf_score = candidates[item.index]
                boosted = self._metadata_boost(item.relevance_score, doc, lang_hint, query=query)
                results.append(RetrievedChunk(
                    document=doc,
                    rrf_score=rrf_score,
                    rerank_score=boosted,
                    final_rank=0,
                    retrieval_method="hybrid+rerank"
                ))

            # Re-sort by boosted score and take top_n
            results.sort(key=lambda c: c.rerank_score, reverse=True)
            results = results[:self.rerank_top_n]
            for i, r in enumerate(results):
                r.final_rank = i + 1

            return results

        except Exception as e:
            logger.warning(f"Reranking failed ({e}), using RRF order")
            return [
                RetrievedChunk(
                    document=doc,
                    rrf_score=score,
                    rerank_score=self._metadata_boost(score, doc, lang_hint, query=query),
                    final_rank=i + 1,
                    retrieval_method="hybrid_fallback"
                )
                for i, (doc, score) in enumerate(candidates[:self.rerank_top_n])
            ]
    
    def retrieve(
        self,
        query: str,
        category_filter: Optional[str] = None
    ) -> list[RetrievedChunk]:
        """
        Full retrieval pipeline:
        Dense + BM25 → RRF Fusion → Cohere Rerank
        """
        logger.debug(f"Retrieving for: '{query}'")
        
        # Dense search
        dense_results = self.dense_search(query, k=self.k * 2)
        
        # BM25 search (category-aware)
        bm25_results = self.bm25_search(query, k=self.k * 2)
        
        # Apply category filter if specified
        if category_filter:
            dense_results = [(d, s) for d, s in dense_results 
                           if d.metadata.get("category") == category_filter]
            bm25_results = [(d, s) for d, s in bm25_results
                          if d.metadata.get("category") == category_filter]
        
        # Fuse with RRF
        fused = self.reciprocal_rank_fusion(dense_results, bm25_results)
        
        # Rerank
        final = self.rerank(query, fused)
        
        return final