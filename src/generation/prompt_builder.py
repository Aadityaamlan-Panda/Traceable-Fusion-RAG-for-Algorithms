"""
CONCEPT: Prompt engineering for RAG has specific requirements:
1. Ground strictly in context ("only use the provided code")
2. Cite sources (critical for anti-hallucination)
3. Request structured output (JSON with code blocks)
4. Chain-of-Thought: "explain reasoning step by step"
5. Negative instruction: "if unsure, say so — do not fabricate"
"""

from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# MAIN ALGORITHM SUGGESTION PROMPT
# ============================================================
ALGO_SUGGESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are AlgoRAG, an expert algorithm assistant supporting C++, Python, Java, and Markdown documentation.

## Your Core Rules (CRITICAL):
1. ONLY use code from the PROVIDED CONTEXT BELOW.
2. If the context does not contain a relevant algorithm, say:
   "I cannot find a suitable algorithm in my knowledge base for this query."
3. NEVER write code from memory — only adapt from context.
4. Always cite source files using [File: <filename>].
5. **COMPLEXITY RULE**: If the context contains a line starting with
   "⚠️  COMPLEXITY NOTE:", you MUST use THAT complexity in your answer.
   Do NOT use generic O() from algorithm descriptions or comments.
   The note reflects the ACTUAL implementation structure (matrix vs heap etc).
6. **COMPARISON RULE**: The "comparison" field must ONLY compare algorithms that appear in
   the Retrieved Context below. Do NOT compare against algorithms not in the context
   (e.g. do not say "less efficient than Bellman-Ford" if Bellman-Ford is not retrieved).
   If only one algorithm is retrieved, say "Only one implementation found — no comparison available."
7. ALWAYS copy the COMPLETE code verbatim from context — NEVER truncate or summarize.

## Output Format (STRICT):
Respond with a valid JSON object. Include AT MOST 2 algorithms — only the most relevant ones.

CRITICAL JSON RULES for the "code" field:
- Escape ALL double-quotes inside code as \\"
- Replace ALL literal newlines with \\n
- Replace ALL literal tabs with \\t
- NEVER use triple-quotes inside a JSON string value
- The entire response must be parseable by json.loads()

{{
  "understanding": "What the user needs in one sentence — derive this from the retrieved context, do not add background knowledge",
  "algorithms": [
    {{
      "name": "Algorithm Name",
      "reason": "Why this fits the query (1 sentence, based on what the code shows — not general knowledge)",
      "time_complexity": "O(...) — from COMPLEXITY NOTE if present, else from code analysis",
      "space_complexity": "O(...)",
      "code": "COMPLETE CODE with newlines as \\n and quotes escaped",
      "source_file": "filename from context",
      "confidence": 0.0-1.0
    }}
  ],
  "comparison": "1-2 sentence tradeoff summary based ONLY on what the retrieved context shows — complexity differences, data structures used. Do not invent trade-offs not visible in the code.",
  "caveats": "1-2 sentence edge-case note derived STRICTLY from what is observable in the retrieved code (e.g. no negative-weight check present in code → note that). Do NOT add general CS background knowledge that is not shown in the retrieved context. Do not contradict yourself."
}}

## Retrieved Context:
{context}
"""),
    ("human", "{query}")
])

# ============================================================
# EXPLANATION PROMPT — for teaching mode
# ============================================================
EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are AlgoRAG in TEACHING MODE.
Explain the algorithm implementation from the context below.

Rules:
- Use ONLY the provided code — don't supplement from memory
- Explain line by line for complex sections
- Relate to CS fundamentals (recursion, divide-and-conquer, etc.)
- Point out common pitfalls and edge cases
- If something in the code is unclear, say so honestly

Context:
{context}
"""),
    ("human", "Explain this algorithm: {query}")
])

# ============================================================
# CONFIDENCE SELF-ASSESSMENT PROMPT
# ============================================================
CONFIDENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Assess how well the following context answers the query.
Return ONLY a JSON:
{{
  "can_answer": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation",
  "missing_info": "what's missing if confidence < 0.8"
}}

Context:
{context}"""),
    ("human", "Query: {query}")
])


def _strip_header(page_content: str) -> str:
    """Strip the prepended 'File: ...' header line added by add_context_header()."""
    lines = page_content.splitlines()
    if lines and lines[0].startswith("File:"):
        skip = 2 if len(lines) > 1 and lines[1].strip() == "" else 1
        return "\n".join(lines[skip:])
    return page_content


def _merge_chunks(chunks: list) -> str:
    """
    Merge consecutive chunks from the same file into one continuous code block.

    Chunks overlap by chunk_overlap characters. We detect the overlap by finding
    the longest suffix of the previous chunk that matches a prefix of the next,
    then join without duplicating it. This reconstructs the original source order
    and eliminates the mid-function start problem caused by chunk boundaries.
    """
    if not chunks:
        return ""

    # Sort by chunk_index so we always stitch in file order
    sorted_chunks = sorted(chunks, key=lambda c: c.document.metadata.get("chunk_index", 0))

    merged = _strip_header(sorted_chunks[0].document.page_content)

    for chunk in sorted_chunks[1:]:
        raw = _strip_header(chunk.document.page_content)

        # Find the longest overlap between the tail of merged and the head of raw.
        # Search up to 400 chars (> chunk_overlap=200, with headroom).
        overlap_found = False
        search_len = min(400, len(merged), len(raw))
        for overlap in range(search_len, 15, -1):   # min 16 chars to avoid false matches
            if merged[-overlap:] == raw[:overlap]:
                merged = merged + raw[overlap:]
                overlap_found = True
                break

        if not overlap_found:
            # No overlap detected — just append with a separator so nothing is lost
            merged = merged + "\n" + raw

    return merged



# ============================================================
# CODE-AWARE COMPLEXITY INFERENCE
# ============================================================

def _count_max_loop_depth(code: str) -> int:
    """
    Count the maximum nesting depth of for/while loops in the code.

    Uses a brace-tracking state machine for C++/Java, and an indent-based
    fallback for Python.  This correctly distinguishes between:
      - Two sequential loops (depth 1 each  -> overall max depth 1)
      - Two nested loops    (depth 2)
      - Three nested loops  (depth 3, e.g. Floyd-Warshall k/i/j)

    The old approach (regex r'for...for' with re.DOTALL) matched ANY two
    'for' tokens anywhere in the file and treated them as nested, which caused
    Floyd-Warshall (three nested loops -> O(V^3)) to be wrongly inferred as O(V^2).

    State machine detail:
      - We push brace_depth (BEFORE the loop body '{'  opens) when we see a
        loop keyword.
      - When a '}' fires we decrement brace_depth, then pop any loop entries
        whose push-depth >= new brace_depth (their body scope has closed).
      - max_depth = max simultaneous entries on the loop stack.
    """
    import re

    # Strip line comments to avoid matching keywords inside comment text
    stripped = re.sub(r'//[^\n]*', '', code)
    stripped = re.sub(r'#[^\n]*', '', stripped)

    brace_depth = 0
    loop_stack: list[int] = []
    max_depth = 0

    tokens = re.split(r'(\{|\}|\bfor\b|\bwhile\b)', stripped)
    for tok in tokens:
        tok = tok.strip()
        if tok == '{':
            brace_depth += 1
        elif tok == '}':
            brace_depth = max(0, brace_depth - 1)
            # has its body open at D+1; when brace_depth returns to D the body
            # is over -> pop all entries with push_depth >= current brace_depth.
            loop_stack = [d for d in loop_stack if d < brace_depth]
        elif tok in ('for', 'while'):
            loop_stack.append(brace_depth)
            max_depth = max(max_depth, len(loop_stack))

    # ── Indent-based fallback for Python (no braces) ──────────────────────
    if max_depth == 0 and '{' not in code:
        indent_stack: list[int] = []
        for line in code.splitlines():
            stripped_line = line.lstrip()
            if not stripped_line:
                continue
            indent = len(line) - len(stripped_line)
            indent_stack = [i for i in indent_stack if i < indent]
            if re.match(r'(for|while)\b', stripped_line):
                indent_stack.append(indent)
                max_depth = max(max_depth, len(indent_stack))

    return max_depth

def infer_complexity_from_code(code: str, language: str) -> dict:
    """
    Inspect the retrieved code for structural markers that imply a specific
    complexity class.  This prevents the LLM from copying a generic description
    (e.g. "O(E log V)" for Prim's) when the actual implementation uses a
    linear scan instead of a heap.

    KEY FIX: We now use actual loop-depth counting (_count_max_loop_depth)
    instead of detecting whether two 'for' tokens exist anywhere in the file.
    The old regex matched sequential loops as nested, causing Floyd-Warshall
    (3 nested loops → O(V³)) to be wrongly reported as O(V²).

    Decision table:
      >= 3 nested loops                      -> O(V^3) / O(n^3)
      2 nested loops + linear min (no heap)  -> O(V^2)  [matrix Prim/Dijkstra]
      2 nested loops + matrix (no heap)      -> O(V^2)
      heap + adjacency list                  -> O((V+E) log V)
      sort call                              -> O(n log n)
      (else)                                 -> no hint (LLM decides)

    Returns a dict with keys "time_hint" and "note" to inject into context.
    The LLM is instructed to prefer this over any O() annotation in comments.
    """
    import re
    code_lower = code.lower()

    # ── Heap / priority queue usage ──────────────────────────────────────
    heap_patterns = [
        r'priority_queue', r'heapq', r'PriorityQueue', r'min_heap',
        r'heappush', r'heappop', r'pq\.', r'\.offer\(', r'\.poll\(',
    ]
    has_heap = any(re.search(p, code, re.IGNORECASE) for p in heap_patterns)

    # ── Adjacency matrix (2-D subscript access) ───────────────────────────
    has_adj_matrix = (
        "graph[u][v]" in code or "graph[i][j]" in code or
        "dist[i][j]" in code or "distance[i][j]" in code or
        bool(re.search(r'\w+\[i\]\[j\]|\w+\[u\]\[v\]|\w+\[k\]\[', code))
    )

    # ── Linear scan for minimum (no heap) ────────────────────────────────
    has_linear_min = bool(re.search(
        r'for.*min.*key|minkey|min_key|extract.min',
        code_lower
    )) and not has_heap

    # ── Sorting ───────────────────────────────────────────────────────────
    has_sort = bool(re.search(r'\.sort\(|std::sort|Arrays\.sort|sorted\(', code))

    # ── Loop depth — the decisive structural signal ───────────────────────
    loop_depth = _count_max_loop_depth(code)

    # ── Infer time complexity ─────────────────────────────────────────────
    time_hint = None
    note = None

    if loop_depth >= 3 and not has_heap:
        # Three (or more) nested loops -> cubic complexity.
        # Floyd-Warshall is the canonical example: k/i/j loops each over V.
        hint_label = "O(V\u00b3)" if has_adj_matrix else "O(n\u00b3)"
        time_hint = hint_label
        note = (
            f"Implementation contains {loop_depth} nested loops "
            f"{'over an adjacency matrix ' if has_adj_matrix else ''}"
            f"\u2014 actual complexity is {hint_label}, NOT O(V\u00b2). "
            f"Floyd-Warshall and similar all-pairs algorithms are {hint_label}; "
            f"use {hint_label} in your answer."
        )
    elif has_adj_matrix and has_linear_min and not has_heap:
        time_hint = "O(V\u00b2)"
        note = (
            "Implementation uses an adjacency matrix with a linear minKey/minDist scan \u2014 "
            "actual complexity is O(V\u00b2), NOT O(E log V). "
            "Use O(V\u00b2) in your answer, not the generic heap-based Prim/Dijkstra description."
        )
    elif has_adj_matrix and not has_heap and loop_depth >= 2:
        time_hint = "O(V\u00b2)"
        note = (
            "Implementation uses an adjacency matrix without a priority queue \u2014 "
            "complexity is O(V\u00b2), not O(E log V)."
        )
    elif has_heap and not has_adj_matrix:
        time_hint = "O((V+E) log V)"
        note = "Implementation uses a heap/priority-queue with adjacency list \u2014 complexity is O((V+E) log V)."
    elif has_sort:
        time_hint = "O(n log n) due to sorting"
        note = "Implementation contains a sort call."

    return {"time_hint": time_hint, "note": note}


def format_context_for_prompt(retrieved_chunks, max_chars_per_chunk: int = 3000) -> str:
    """
    CONCEPT: How you format the context affects LLM performance significantly.

    Key improvement over naive per-chunk formatting:
    - Chunks from the SAME source file are MERGED in chunk_index order before
      being sent to the LLM. This eliminates the mid-function start problem where
      the retrieved chunk begins inside a function body because the file was split
      at that point by the chunker.
    - Chunks from DIFFERENT files are kept separate (they are genuinely distinct
      algorithms).

    The merged block is capped at max_chars_per_chunk * 2 to stay within token
    budgets while still providing enough context for complete function bodies.
    """
    # Group chunks by source file, preserving the retrieval rank of the best
    # (highest-ranked) chunk for each file so we can order files by relevance.
    from collections import defaultdict
    file_groups: dict[str, list] = defaultdict(list)
    file_best_rank: dict[str, int] = {}

    for rank, chunk in enumerate(retrieved_chunks):
        src = chunk.document.metadata.get("source", "unknown")
        file_groups[src].append(chunk)
        if src not in file_best_rank:
            file_best_rank[src] = rank   # first occurrence = best rank

    # Sort files by their best retrieval rank (most relevant first)
    sorted_files = sorted(file_groups.keys(), key=lambda s: file_best_rank[s])

    context_parts = []
    for i, src in enumerate(sorted_files, 1):
        chunks = file_groups[src]
        meta = chunks[0].document.metadata   # metadata is same for all chunks of a file

        # Merge first — complexity inference needs the actual code
        merged_code = _merge_chunks(chunks)

        # Cap total characters — 2× per-chunk budget for merged blocks
        cap = max_chars_per_chunk * 2 if len(chunks) > 1 else max_chars_per_chunk
        code = merged_code[:cap]

        # Infer actual complexity from code structure (overrides comment annotations)
        lang = meta.get("language", "cpp")
        complexity_info = infer_complexity_from_code(code, lang)
        complexity_display = (
            complexity_info["time_hint"]
            if complexity_info["time_hint"]
            else meta.get("time_complexity", "N/A")
        )
        header = (
            f"[{i}] File: {src} | "
            f"Language: {lang} | "
            f"Category: {meta.get('category', 'unknown')} | "
            f"Algorithm: {meta.get('algorithm_name', 'unknown')} | "
            f"Inferred Complexity: {complexity_display}"
        )
        if complexity_info["note"]:
            header += f"\n⚠️  COMPLEXITY NOTE: {complexity_info['note']}"

        # Use the correct language fence so the LLM knows what it's reading
        fence = {"cpp": "cpp", "python": "python", "java": "java",
                 "markdown": "markdown"}.get(lang, "")
        context_parts.append(f"{header}\n```{fence}\n{code}\n```")

    # Prepend a numbered source file index so the LLM always has explicit
    # filenames to cite — prevents "No specific file provided" answers.
    source_index = "AVAILABLE SOURCE FILES (cite these exactly in source_file field):\n"
    for i, src in enumerate(sorted_files, 1):
        lang = file_groups[src][0].document.metadata.get("language", "")
        source_index += f"  [{i}] {src} ({lang})\n"

    return source_index + "\n---\n\n" + "\n\n---\n\n".join(context_parts)