"""
CONCEPT: Resilient LLM calling with:
1. Primary: Groq (Llama 3.3 70B) — ultra-fast, 1K RPD / 30K TPM free
2. Fallback: Cohere Command A+ (command-a-03-2025) — 1K API calls/month free, 128K context, RAG-optimised
3. JSON output parsing with raw-newline repair
4. Streaming support for terminal real-time display
"""

import json
import re
import time
from typing import Generator
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from .prompt_builder import (
    ALGO_SUGGESTION_PROMPT,
    EXPLANATION_PROMPT,
    format_context_for_prompt
)
import logging

logger = logging.getLogger(__name__)


def _build_cohere_client(api_key: str):
    """Lazily build a raw Cohere client (used for the fallback)."""
    import cohere
    return cohere.Client(api_key=api_key)


class AlgoGenerator:
    """
    LLM generation with 2-provider fallback and structured output.

    Provider order (best free-tier first):
      1. Groq 70B          — 1K RPD, 30K TPM, very fast
      2. Cohere Command A+ — 1K calls/month, 128K ctx, RAG-tuned
    """

    # Phrases that indicate a rate-limit or context-not-found refusal
    # returned as a 200 OK (not an exception) by the provider.
    # Only checked on SHORT responses (< 500 chars) to avoid false positives
    # where the LLM legitimately uses these words inside code/JSON output.
    _REFUSAL_PHRASES = [
        "I cannot find a suitable algorithm",
        "rate limit",
        "Rate limit",
        "429",
        "quota",
        "Resource exhausted",
    ]

    def __init__(self, groq_api_key: str, cohere_api_key: str, **kwargs):
        # Primary: Groq Llama 3.3 70B
        # request_timeout=20 prevents hanging on rate-limit retries.
        # max_retries=1: one retry, then raise so Cohere fallback fires.
        self.primary_llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=4000,
            stop_sequences=None,
            request_timeout=20,
            max_retries=1,
        )

        # Fallback: Cohere Command A+ (via raw SDK — best for RAG)
        # 1K API calls/month free, 128K context window
        self._cohere_api_key = cohere_api_key
        self._cohere_client = None  # lazy init

        self.parser = StrOutputParser()

    def _get_cohere_client(self):
        if self._cohere_client is None:
            self._cohere_client = _build_cohere_client(self._cohere_api_key)
        return self._cohere_client

    def _call_cohere(self, chain_input: dict, prompt) -> str:
        """
        Call Cohere Command A+ directly via the SDK.
        Formats the LangChain prompt messages into Cohere's chat format.
        """
        client = self._get_cohere_client()

        # Render the prompt template to get the actual messages
        messages = prompt.format_messages(**chain_input)

        system_text = ""
        human_text = ""
        for msg in messages:
            if hasattr(msg, 'type'):
                if msg.type == "system":
                    system_text = msg.content
                elif msg.type == "human":
                    human_text = msg.content
            else:
                # Fallback: stringify
                if "System" in type(msg).__name__:
                    system_text = str(msg)
                else:
                    human_text = str(msg)

        response = client.chat(
            model="command-a-03-2025",
            preamble=system_text,
            message=human_text,
            temperature=0,
            max_tokens=4000,
        )
        return response.text

    def _is_refusal(self, text: str) -> bool:
        """
        Return True if the response is a rate-limit or refusal string.

        Only checks short responses (< 500 chars) to avoid false positives
        where valid JSON/code output happens to contain these words.
        For longer responses, only the unambiguous HTTP-status marker '429'
        is checked, since that never appears in legitimate algorithm output.
        """
        if len(text) < 500:
            return any(phrase in text for phrase in self._REFUSAL_PHRASES)
        # For long responses only flag hard rate-limit signals
        return "429" in text or text.strip().startswith("Rate limit")

    def _call_with_fallback(self, chain_input: dict, prompt) -> tuple:
        """
        2-provider fallback chain.
        Groq → Cohere Command A+

        Handles both exceptions AND silent refusals (200 OK with refusal text).

        Returns (result: str, provider_name: str).
        """
        providers = [
            ("Groq Llama 3.3 70B",  self._call_groq),
            ("Cohere Command A+",   self._call_cohere),
        ]

        last_error = None
        for name, caller in providers:
            try:
                result = caller(chain_input, prompt)
                if self._is_refusal(result):
                    logger.warning(f"{name} returned a refusal/rate-limit response, trying next provider...")
                    continue
                logger.debug(f"Provider succeeded: {name}")
                return result, name
            except Exception as e:
                logger.warning(f"{name} failed ({type(e).__name__}: {e}), trying next provider...")
                last_error = e

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def _call_groq(self, chain_input: dict, prompt) -> str:
        chain = prompt | self.primary_llm | self.parser
        return chain.invoke(chain_input)

    def stream_response(self, chain_input: dict, prompt) -> Generator[str, None, None]:
        """
        Stream tokens for real-time terminal display.
        Tries Groq first, falls back to Cohere Command A+ (non-streaming).

        Also guards against Groq returning a silent rate-limit / refusal string
        as a 200 OK response — in that case the full streamed text is checked
        and Cohere is used instead.
        """
        # Try Groq streaming — collect chunks so we can check for refusals
        try:
            chain = prompt | self.primary_llm
            collected: list[str] = []
            for chunk in chain.stream(chain_input):
                if isinstance(chunk, AIMessageChunk):
                    collected.append(chunk.content)
            full_text = "".join(collected)
            if not self._is_refusal(full_text):
                yield full_text
                return
            logger.warning("Groq streaming returned refusal/rate-limit, falling back to Cohere Command A+...")
        except Exception as e:
            logger.warning(f"Groq streaming failed ({e}), falling back to Cohere Command A+ non-streaming...")

        # Cohere doesn't stream in this integration — yield full response
        result = self._call_cohere(chain_input, prompt)
        yield result

    def parse_json_response(self, text: str) -> dict:
        """
        Safely parse JSON responses from the LLM.

        LLMs frequently embed raw newlines, triple-quotes (Python docstrings),
        and unescaped characters inside JSON string values — especially in the
        code field.  We attempt several repair strategies before giving up.
        """
        # Step 1: strip markdown fences
        cleaned = re.sub(r'```(?:json|cpp|c\+\+|python|java)?\s*', '', text).strip()
        cleaned = cleaned.rstrip('`').strip()

        def _repair_and_parse(s: str) -> dict | None:
            # Try direct parse first
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass

            # Repair 1: escape raw newlines/tabs inside JSON string values
            # Uses a state machine approach — replace \n/\t only inside "..."
            try:
                fixed = re.sub(
                    r'"((?:[^"\\]|\\.)*)"',
                    lambda m: '"' + m.group(1)
                        .replace('\n', '\\n')
                        .replace('\r', '\\r')
                        .replace('\t', '\\t') + '"',
                    s,
                    flags=re.DOTALL
                )
                return json.loads(fixed)
            except (json.JSONDecodeError, Exception):
                pass

            # Repair 2: replace triple-quotes with escaped single quotes
            try:
                fixed2 = s.replace('"""', '\\"\\"\\"').replace("'''", "\\'\\'\\'")
                return json.loads(fixed2)
            except (json.JSONDecodeError, Exception):
                pass

            return None

        # Step 2: try whole cleaned string
        result = _repair_and_parse(cleaned)
        if result is not None:
            return result

        # Step 3: extract outermost {...} block and retry
        start = cleaned.find('{')
        end   = cleaned.rfind('}') + 1
        if start != -1 and end > start:
            result = _repair_and_parse(cleaned[start:end])
            if result is not None:
                return result

        # Step 4: extract "code" field manually then rebuild minimal dict
        # Handles the case where the code field itself is so large/complex
        # that no repair can make it valid JSON
        understanding_m = re.search(r'"understanding"\s*:\s*"([^"]*)"', cleaned)
        code_blocks = re.findall(
            r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, re.DOTALL
        )
        name_blocks = re.findall(r'"name"\s*:\s*"([^"]*)"', cleaned)
        if code_blocks:
            algorithms = []
            for i, code in enumerate(code_blocks):
                algorithms.append({
                    "name":            name_blocks[i] if i < len(name_blocks) else "Algorithm",
                    "reason":          "Extracted from partial JSON",
                    "time_complexity": "unknown",
                    "space_complexity": "unknown",
                    "code":            code.replace('\\n', '\n').replace('\\t', '\t'),
                    "source_file":     "unknown",
                    "confidence":      0.7,
                })
            return {
                "understanding": understanding_m.group(1) if understanding_m else "See code below",
                "algorithms":    algorithms,
                "comparison":    "",
                "caveats":       "Response was partially parsed due to JSON formatting issues in code field.",
            }

        # Step 5: raw code fallback (C++/Python/Java indicators)
        code_indicators = [
            "#include", "using namespace std", "int main(",
            "def ", "class ", "public static", "import java"
        ]
        if any(ind in cleaned for ind in code_indicators):
            logger.warning("Detected raw code response instead of JSON")
            return {
                "understanding": "Generated code implementation.",
                "algorithms": [{
                    "name":            "Generated Code",
                    "time_complexity": "Unknown",
                    "space_complexity": "Unknown",
                    "reason":          "Returned directly by the LLM",
                    "confidence":      0.95,
                    "source_file":     "N/A",
                    "code":            cleaned,
                }],
                "comparison": "",
                "caveats":    "",
            }

        # Step 6: unrecoverable
        logger.error(f"JSON parse failed. Raw response: {text[:500]}")
        return {
            "understanding": "Parse error",
            "algorithms":    [],
            "comparison":    "",
            "caveats":       "Response parsing failed.",
            "parse_error":   True,
        }

    def suggest_algorithms(self, query: str, retrieved_chunks) -> tuple:
        """Main generation method — returns (structured dict, provider_name)."""
        # Use top 3 chunks with a generous per-chunk cap so full function
        # bodies aren't truncated mid-way through.
        context = format_context_for_prompt(retrieved_chunks[:3], max_chars_per_chunk=3000)

        raw_response, provider = self._call_with_fallback(
            {"query": query, "context": context},
            ALGO_SUGGESTION_PROMPT
        )

        return self.parse_json_response(raw_response), provider

    def explain_algorithm(self, query: str, retrieved_chunks) -> tuple:
        """Teaching mode — returns (explanation text, provider_name)."""
        context = format_context_for_prompt(retrieved_chunks[:3], max_chars_per_chunk=3000)

        result, provider = self._call_with_fallback(
            {"query": query, "context": context},
            EXPLANATION_PROMPT
        )
        return result, provider

    def stream_explanation(self, query: str, retrieved_chunks) -> Generator:
        """Stream explanation tokens for real-time display."""
        context = format_context_for_prompt(retrieved_chunks[:3], max_chars_per_chunk=3000)
        return self.stream_response(
            {"query": query, "context": context},
            EXPLANATION_PROMPT
        )
