"""
CONCEPT: Hypothetical Document Embedding (HyDE)
Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"

Problem: User queries are short and vague.
  "fast sorting" → sparse query vector

HyDE solution:
  1. Ask LLM to write a HYPOTHETICAL answer/document
  2. Embed that hypothetical document (not the query)
  3. Search with the hypothetical embedding

The hypothetical doc is more similar to actual docs than the short query!
This is especially powerful for code queries.

Provider fallback: Groq 70B → Cohere Command A+
"""

import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a C++ algorithm expert. 
Given a question about algorithms, write a SHORT hypothetical C++ code snippet 
(50-150 words) that would answer the question.
Include: function signature, key logic, complexity comment.
Output ONLY the code snippet, no explanation."""),
    ("human", "Question: {query}")
])

# Refusal/rate-limit phrases to detect silent failures
_REFUSAL_PHRASES = ["rate limit", "Rate limit", "429", "quota", "Resource exhausted", "I cannot"]


def _is_refusal(text: str) -> bool:
    """Only flag short responses or hard rate-limit signals to avoid false positives."""
    if len(text) < 300:
        return any(p in text for p in _REFUSAL_PHRASES)
    return "429" in text or text.strip().startswith("Rate limit")


def _call_cohere_hyde(cohere_api_key: str, query: str) -> str:
    """Cohere Command A+ fallback for HyDE generation."""
    import cohere
    client = cohere.Client(cohere_api_key)
    system = HYDE_PROMPT.messages[0].prompt.template
    response = client.chat(
        model="command-a-03-2025",
        preamble=system,
        message=f"Question: {query}",
        temperature=0.3,
        max_tokens=300,
    )
    return response.text


class HyDEChain:
    """
    Resilient HyDE query transformer with 2-provider fallback.

    Provider order:
      1. Groq Llama 3.3 70B
      2. Cohere Command A+

    request_timeout=12 prevents LangChain's retry wrapper from silently
    hanging 2-4 minutes on rate-limit errors. If Groq doesn't respond
    in 12s it raises immediately and we fall over to Cohere.
    """

    def __init__(self, groq_api_key: str, cohere_api_key: str = "", **kwargs):
        self.groq = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=300,
            request_timeout=12,
            max_retries=1,
        )
        self._cohere_api_key = cohere_api_key
        self._parser = StrOutputParser()

    def invoke(self, input_dict: dict) -> str:
        query = input_dict["query"]

        # Provider 1: Groq
        try:
            result = (HYDE_PROMPT | self.groq | self._parser).invoke(input_dict)
            if not _is_refusal(result):
                return result
            logger.warning("HyDE: Groq returned refusal, trying Cohere Command A+...")
        except Exception as e:
            logger.warning(f"HyDE: Groq failed ({e}), trying Cohere Command A+...")

        # Provider 2: Cohere Command A+
        try:
            return _call_cohere_hyde(self._cohere_api_key, query)
        except Exception as e:
            logger.error(f"HyDE: All providers failed. Last: {e}")
            # Last resort: return the raw query so retrieval still works
            return query


def create_hyde_chain(groq_api_key: str, cohere_api_key: str = "", **kwargs) -> HyDEChain:
    """
    Factory — returns a HyDEChain with full 2-provider fallback.
    """
    return HyDEChain(
        groq_api_key=groq_api_key,
        cohere_api_key=cohere_api_key,
    )


# Step-Back Prompting — another query transformation technique
STEP_BACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a C++ algorithm expert.
Given a specific question, generate a MORE GENERAL version that would 
help find relevant algorithmic concepts.
Output only the generalised question."""),
    ("human", "Specific question: {query}\nGeneralised question:")
])
