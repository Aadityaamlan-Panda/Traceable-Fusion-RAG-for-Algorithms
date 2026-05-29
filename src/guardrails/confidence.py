"""
CONCEPT: Confidence scoring combines multiple signals:
1. Retrieval score (how relevant are the retrieved chunks?)
2. Source verification (do cited files actually exist?)
3. Coverage score (how much of the answer is grounded in retrieved context?)

The final confidence score is shown to the user, creating transparency
about the system's certainty — a hallmark of trustworthy AI systems.
"""

import re
import json
import logging
from dataclasses import dataclass
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceReport:
    """Detailed confidence breakdown."""
    retrieval_score: float      # Average rerank score of top chunks
    source_verified: bool       # Cited sources exist in our index
    coverage_score: float       # Fraction of answer claims supported by context
    overall_confidence: float   # Weighted average
    level: str                  # HIGH / MEDIUM / LOW / INSUFFICIENT
    should_proceed: bool        # Whether to generate or refuse
    warnings: list[str]         # Specific concerns


class ConfidenceScorer:
    """
    Multi-signal confidence scorer for RAG outputs.
    """
    
    # Thresholds
    HIGH_THRESHOLD = 0.75
    MEDIUM_THRESHOLD = 0.50
    LOW_THRESHOLD = 0.30
    REFUSE_THRESHOLD = 0.25
    
    _REFUSAL_PHRASES = ["rate limit", "Rate limit", "429", "quota", "Resource exhausted"]

    def __init__(self, groq_api_key: str, known_files: set[str],
                 cohere_api_key: str = "", **kwargs):
        """
        known_files: set of all file paths in our index.
        LLM fallback order: Groq 70B → Cohere Command A+
        """
        self.known_files = known_files
        self._cohere_api_key = cohere_api_key

        # Primary: Groq
        self.groq_llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=256,
            request_timeout=12,
            max_retries=1,
        )

        # Fallback: Cohere Command A+ (lazy init)
        self._cohere_client = None

    def _get_cohere_client(self):
        if self._cohere_client is None:
            import cohere
            self._cohere_client = cohere.Client(self._cohere_api_key)
        return self._cohere_client

    def _call_coverage_llm(self, coverage_prompt, context: str, answer: str) -> str:
        """Call coverage LLM with Groq → Cohere Command A+ fallback."""
        chain_input = {"context": context, "answer": answer}

        # Groq
        try:
            chain = coverage_prompt | self.groq_llm | StrOutputParser()
            result = chain.invoke(chain_input)
            # Refusal: short response containing a rate-limit phrase.
            # Long responses are always returned — they cannot be pure refusals.
            is_refusal = len(result) < 500 and any(p in result for p in self._REFUSAL_PHRASES)
            if not is_refusal:
                return result
            logger.warning("Coverage Groq returned refusal/rate-limit, trying Cohere Command A+...")
        except Exception as e:
            logger.warning(f"Coverage Groq failed ({e}), trying Cohere Command A+...")

        # Cohere Command A+
        if self._cohere_api_key:
            try:
                client = self._get_cohere_client()
                msgs = coverage_prompt.format_messages(**chain_input)
                system_text = next((m.content for m in msgs if m.type == "system"), "")
                human_text = next((m.content for m in msgs if m.type == "human"), "")
                resp = client.chat(
                    model="command-a-03-2025",
                    preamble=system_text,
                    message=human_text,
                    temperature=0,
                    max_tokens=256,
                )
                return resp.text
            except Exception as e:
                logger.warning(f"Coverage Cohere Command A+ failed ({e})")

        raise RuntimeError("All coverage LLM providers failed")
    
    def score_retrieval(self, retrieved_chunks) -> float:
        """
        Score based on how relevant the retrieved chunks are.
        Uses the reranker scores from the retrieval phase.
        """
        if not retrieved_chunks:
            return 0.0
        
        # Weighted average: top chunk has most weight
        scores = [c.rerank_score for c in retrieved_chunks]
        weights = [1.0 / (i + 1) for i in range(len(scores))]  # 1, 0.5, 0.33, ...
        
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        
        return min(weighted_sum / total_weight, 1.0)
    
    def verify_sources(self, cited_files: list[str]) -> tuple[bool, list[str]]:
        """
        Check that files cited in the LLM's response actually exist in our index.
        This is the primary hallucination detection layer for code RAG.

        Handles common LLM formatting quirks:
          - "File: graph/kosaraju.cpp"  → strips "File: " prefix
          - Leading slashes
          - Long explanatory strings (LLM wrote a sentence instead of a path)
          - Duplicate citations
        """
        warnings = []
        known_lower = {f.lower() for f in self.known_files}

        for raw_cited in cited_files:
            if not raw_cited or not raw_cited.strip():
                continue

            # Strip common LLM artefacts
            cited = raw_cited.strip()
            for prefix in ("File: ", "file: ", "FILE: ", "[File: ", "[file: "):
                if cited.startswith(prefix):
                    cited = cited[len(prefix):].rstrip("]").strip()
                    break

            cited = cited.lstrip("/").strip()

            # If the string is very long (> 120 chars) it's a sentence, not a path
            # — skip the hallucination check entirely for these
            if len(cited) > 120:
                continue

            # Skip strings that contain spaces but no path separator
            # (e.g. "No Java implementation provided...")
            if " " in cited and "/" not in cited and "\\" not in cited:
                continue

            normalised = cited.lower()
            exact_match   = normalised in known_lower
            # Partial: either the cited path is a suffix of a known path,
            # or the filename alone matches a known file
            partial_match = (
                any(normalised in f for f in known_lower) or
                any(f.endswith(normalised) for f in known_lower)
            )

            if not exact_match and not partial_match:
                warnings.append(
                    f"⚠️ Cited file '{raw_cited}' not found in knowledge base "
                    f"— potential hallucination"
                )

        return len(warnings) == 0, warnings
    
    def score_coverage(
        self,
        generated_answer: str,
        retrieved_chunks,
        sample_claims: int = 3
    ) -> tuple:
        """
        CONCEPT: Faithfulness scoring.

        We ask the LLM to extract factual claims from the answer,
        then check if each claim is supported by retrieved context.

        This is a simplified version of RAGAS Faithfulness metric.

        Returns (score: float, scored_successfully: bool).
        scored_successfully=False means both providers failed and 0.5 is a
        neutral placeholder — the caller surfaces a visible warning so the
        user knows coverage scoring was skipped rather than silently wrong.
        """
        if not generated_answer or not retrieved_chunks:
            return 0.0, False

        context = "\n\n".join([c.document.page_content[:500] for c in retrieved_chunks[:3]])

        coverage_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are assessing whether a generated algorithm answer is grounded in the retrieved source code context.

Score the fraction of factual claims in the answer that are DIRECTLY SUPPORTED by the context below.

IMPORTANT CALIBRATION RULES:
- If the answer's code appears verbatim (or nearly verbatim) in the context: coverage should be 0.85-1.0
- If the complexity (e.g. O(V^3)) matches what the context code structure shows: that claim is supported
- If the source file cited is present in the context: that claim is supported
- Only mark claims as unsupported if they introduce facts NOT present in the context at all
- Do NOT penalize paraphrasing or light rewording of context content

Return ONLY a JSON object (no markdown, no extra text):
{{"coverage": 0.0-1.0, "unsupported_claims": ["claim1", ...]}}

Context:
{context}"""),
            ("human", "Generated answer: {answer}")
        ])

        try:
            raw = self._call_coverage_llm(coverage_prompt, context, generated_answer[:1000])

            # Parse — guard against empty response before JSON decode
            raw = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`')
            if not raw:
                raise ValueError("Empty response from coverage LLM")
            # Extract JSON object if wrapped in extra text
            start, end = raw.find('{'), raw.rfind('}') + 1
            if start != -1 and end > start:
                raw = raw[start:end]
            data = json.loads(raw)
            return float(data.get("coverage", 0.5)), True

        except Exception as e:
            logger.warning(f"Coverage scoring failed: {e}")
            return 0.5, False  # neutral placeholder; caller emits a warning
    
    def assess(
        self,
        retrieved_chunks,
        generated_result: dict,
        query: str
    ) -> ConfidenceReport:
        """
        Full confidence assessment — call after generation.

        Signal weights:
          40%  retrieval quality   (reranker scores of top chunks)
          30%  coverage score      (LLM-judged faithfulness to context)
          30%  source verification (cited files exist in knowledge base)

        Additional penalties applied after weighting:
          -0.15  answer caveats mention missing methods/implementations
          -0.10  answer contains only a docstring/header, no real code
          -0.10  generation was from memory (hallucination guard triggered)
        """
        warnings = []

        # 1. Retrieval quality
        retrieval_score = self.score_retrieval(retrieved_chunks)

        # 2. Source verification
        cited_files = [
            algo.get("source_file", "")
            for algo in generated_result.get("algorithms", [])
        ]
        sources_ok, source_warnings = self.verify_sources(cited_files)
        warnings.extend(source_warnings)

        # 3. Coverage scoring
        coverage_ok = False
        coverage_score = None
        if retrieval_score > self.MEDIUM_THRESHOLD:
            answer_text = " ".join([
                algo.get("reason", "") + " " + algo.get("code", "")[:200]
                for algo in generated_result.get("algorithms", [])
            ])
            coverage_score, coverage_ok = self.score_coverage(answer_text, retrieved_chunks)
            if not coverage_ok:
                warnings.append(
                    "⚠️ Coverage scoring unavailable (both LLM providers failed) "
                    "— excluded from confidence calculation"
                )

        # 4. Weighted overall confidence
        # When coverage is unavailable, redistribute its 30% weight equally
        # between retrieval and sources so the score stays meaningful.
        source_multiplier = 1.0 if sources_ok else 0.6
        source_score = 1.0 if sources_ok else 0.0

        if coverage_ok and coverage_score is not None:
            overall = (
                0.40 * retrieval_score +
                0.30 * coverage_score +
                0.30 * source_score
            ) * source_multiplier
        else:
            # No coverage signal — 55/45 retrieval/sources split
            overall = (
                0.55 * retrieval_score +
                0.45 * source_score
            ) * source_multiplier
            coverage_score = retrieval_score  # proxy for breakdown display

        # 5. Caveat-based penalty
        # If the generated answer itself warns about missing methods or
        # incomplete implementations, that directly contradicts HIGH confidence.
        _INCOMPLETENESS_SIGNALS = [
            "does not include", "not included", "missing", "not implemented",
            "not provided", "not shown", "assumed", "not defined",
            "written based on", "from memory", "no java", "no c++", "no python",
            "no implementation", "adapt from", "may need to be adjusted",
        ]
        caveats_text = (
            generated_result.get("caveats", "") + " " +
            " ".join(a.get("source_file", "") for a in generated_result.get("algorithms", []))
        ).lower()

        incompleteness_hits = sum(
            1 for sig in _INCOMPLETENESS_SIGNALS if sig in caveats_text
        )
        if incompleteness_hits > 0:
            penalty = min(0.25, incompleteness_hits * 0.08)
            overall = max(0.0, overall - penalty)
            warnings.append(
                f"⚠️ Answer completeness penalty applied "
                f"({incompleteness_hits} incompleteness signal(s) in caveats)"
            )

        # Code-only-docstring check: if every code block is < 80 chars it's
        # probably just a header/signature, not a full implementation
        all_codes = [a.get("code", "") for a in generated_result.get("algorithms", [])]
        if all_codes and all(len(c.strip()) < 80 for c in all_codes):
            overall = max(0.0, overall - 0.10)
            warnings.append("⚠️ Generated code appears truncated or incomplete")

        # Mixed-language penalty: if the top-5 retrieved chunks span multiple
        # languages, the reranker may have conflated semantically similar code
        # across languages, reducing our trust in source selection.
        retrieved_langs = set()
        if retrieved_chunks:
            retrieved_langs = {
                c.document.metadata.get("language", "unknown")
                for c in retrieved_chunks[:5]
            }
        if len(retrieved_langs) > 1:
            overall = max(0.0, overall - 0.05)
            warnings.append(
                f"⚠️ Mixed-language candidates in retrieval pool "
                f"({', '.join(sorted(retrieved_langs))}) — source selection less certain"
            )

        # Source-mismatch penalty: if the LLM cited a file that is not the
        # highest-ranked retrieved file, it may have picked a tangentially
        # related source (e.g. SteinerTree.java instead of FloydWarshall.java).
        if retrieved_chunks and generated_result.get("algorithms"):
            top_retrieved = retrieved_chunks[0].document.metadata.get("source", "")
            cited_files = [
                a.get("source_file", "").lower()
                for a in generated_result["algorithms"]
            ]
            top_base = top_retrieved.split("/")[-1].lower()
            if top_base and not any(top_base in cf or cf in top_base for cf in cited_files):
                overall = max(0.0, overall - 0.05)
                warnings.append(
                    f"⚠️ Cited source does not match top-ranked retrieval result "
                    f"(top: {top_retrieved.split('/')[-1]}) — reranker may have over-weighted semantic associations"
                )

        # 6. Build human-readable confidence breakdown
        reranker_entropy = ""
        if retrieved_chunks:
            scores = [c.rerank_score for c in retrieved_chunks[:5]]
            spread = max(scores) - min(scores) if len(scores) > 1 else 1.0
            if spread < 0.05:
                reranker_entropy = "reranker scores tightly clustered (ambiguous candidates)"
            elif spread < 0.15:
                reranker_entropy = "moderate reranker score spread"

        breakdown_parts = [f"retrieval={retrieval_score:.0%}",
                           f"coverage={'N/A (unavailable)' if not coverage_ok else f'{coverage_score:.0%}'}",
                           f"sources={'✓' if sources_ok else '✗'}"]
        if reranker_entropy:
            breakdown_parts.append(reranker_entropy)
        if len(retrieved_langs) > 1:
            breakdown_parts.append(f"mixed language candidates ({', '.join(sorted(retrieved_langs))})")
        if incompleteness_hits > 0:
            breakdown_parts.append(f"{incompleteness_hits} incompleteness signal(s)")

        warnings.insert(0, "📊 Confidence breakdown: " + " | ".join(breakdown_parts))

        # Hard-cap at 0.95: 100% confidence is never justified in a RAG system
        # that has mixed-language retrieval pools or semantic-only reranking.
        overall = min(overall, 0.95)

        # 7. Determine level
        if overall >= self.HIGH_THRESHOLD:
            level = "HIGH"
            should_proceed = True
        elif overall >= self.MEDIUM_THRESHOLD:
            level = "MEDIUM"
            should_proceed = True
            warnings.append("💡 Medium confidence — recommend verifying with additional sources")
        elif overall >= self.LOW_THRESHOLD:
            level = "LOW"
            should_proceed = True
            warnings.append("⚠️ Low confidence — answer may be incomplete or imprecise")
        else:
            level = "INSUFFICIENT"
            should_proceed = False
            warnings.append("❌ Confidence too low — insufficient context to answer reliably")

        return ConfidenceReport(
            retrieval_score=retrieval_score,
            source_verified=sources_ok,
            coverage_score=coverage_score,
            overall_confidence=overall,
            level=level,
            should_proceed=should_proceed,
            warnings=warnings
        )


def pre_retrieval_gate(
    retrieved_chunks,
    min_rerank_score: float = 0.25
) -> tuple[bool, str]:
    """
    GATE: Run BEFORE generation.
    If top chunk's rerank score is below threshold, refuse to generate.
    This prevents "hallucination from nothing" when the codebase 
    doesn't contain what the user asked for.
    """
    if not retrieved_chunks:
        return False, "No relevant algorithms found in the knowledge base."
    
    top_score = retrieved_chunks[0].rerank_score
    
    if top_score < min_rerank_score:
        return False, (
            f"Best match score ({top_score:.2f}) is below confidence threshold ({min_rerank_score}). "
            f"The knowledge base may not contain algorithms relevant to your query. "
            f"Try: rephrasing, using algorithm names (e.g., 'Dijkstra', 'quicksort'), "
            f"or category terms (e.g., 'graph', 'sorting', 'dynamic programming')."
        )
    
    return True, ""
