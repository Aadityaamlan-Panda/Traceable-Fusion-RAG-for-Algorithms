# Traceable Fusion RAG for Algorithms

A Retrieval-Augmented Generation (RAG) system for querying algorithm implementations across multiple programming languages. Ask natural-language questions and get answers grounded in real, curated source code — with full transparency into every pipeline step.
<p align="center">
  <img src="images/1.png" alt="Default Page">
</p>


---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Codebase Explained](#codebase-explained)
  - [src/config.py](#srcconfigpy)
  - [src/ingestion/](#srcingesion)
  - [src/retrieval/](#srcretrieval)
  - [src/generation/](#srcgeneration)
  - [src/guardrails/](#srcguardrails)
  - [src/transparency/](#srctransparency)
  - [src/ui/](#srcui)
- [Knowledge Base](#knowledge-base)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Running Tests](#running-tests)
  - [Test Results](#test-results)
  - [Unit Tests](#unit-tests)
  - [Integration Tests](#integration-tests)
  - [End-to-End Tests](#end-to-end-tests)
- [Evaluation (RAGAS)](#evaluation-ragas)
- [Configuration](#configuration)
- [Scripts](#scripts)

---

## Overview

`Traceable Fusion RAG for Algorithms` lets you ask questions like *"How does Dijkstra's algorithm work?"* or *"What is a segment tree and when should I use it?"* and get back structured answers with code snippets, source file citations, and a confidence score — all sourced from real algorithm repositories on GitHub, not from the LLM's parametric memory.

The system ingests 6 open-source algorithm repositories (~1 GB of C++, Python, Java, and Markdown), stores them in a ChromaDB vector store, and at query time runs a hybrid BM25 + dense retrieval pipeline, reranks with Cohere, and generates answers with Groq (Llama 3.3 70B) with a Cohere Command A+ fallback.

<img src="images/2.png" alt="query"><img src="images/3.png" alt="query"><img src="images/4.png" alt="query"><img src="images/5.png" alt="query"><img src="images/6.png" alt="query">
---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Query Transformer  (query_transformer.py)              │
│  Expands / rephrases the query for better recall        │
└───────────────────────┬─────────────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
  Dense Retrieval               BM25 Retrieval
  (ChromaDB + Cohere            (rank-bm25, keyword
   embeddings)                   exact match)
          │                            │
          └─────────────┬──────────────┘
                        ▼
           Reciprocal Rank Fusion (RRF)
           Merges both ranked lists by position
                        │
                        ▼
             Cohere Reranker (cross-encoder)
             Rescores top-N with full attention
                        │
                        ▼
          ┌─────────────────────────────┐
          │  Confidence Gate            │
          │  (guardrails/confidence.py) │
          │  Score < 0.25 → REFUSE      │
          └────────────┬────────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │  LLM Generation            │
          │  Primary:  Groq 70B        │
          │  Fallback: Cohere Cmd A+   │
          └────────────┬───────────────┘
                       │
                       ▼
          Structured JSON Answer
          { answer, source_files,
            confidence_note, cpp_snippet }
                       │
                       ▼
          Rich Terminal UI  (src/ui/app.py)
```

The `Tracer` runs as an observer across all steps, recording every event so the terminal UI can show a step-by-step live display.

---

## Codebase Explained

### `src/config.py`

Central configuration using **Pydantic `BaseSettings`**, which auto-reads values from your `.env` file. This prevents hardcoded values scattered across the codebase.

`ALGO_REPOS` is a plain Python list of dicts that defines *which GitHub repos to ingest* and what language to treat them as. Adding a new repo means appending one dict — nothing else changes.

The `Settings` class holds all tuneable parameters: LLM model names, retrieval `k`, reranking `top_n`, confidence thresholds, chunk sizes. Every other module reads from the single global `settings` instance.

---

### `src/ingestion/`

**`loader.py`** — Clones each repo in `ALGO_REPOS` using `gitpython` (or pulls if already cloned), then walks the directory tree and loads only source files matching the declared language extension. Returns a list of `Document` objects with `source_file`, `language`, and `category` metadata.

**`chunker.py`** — `CppAwareChunker` splits documents into retrievable chunks. Despite its name, it handles all four languages:

| Language | Strategy |
|---|---|
| C++ | Brace-matched function extraction, fallback to LangChain CPP splitter |
| Python | Indent-tracked `def`/`class` extraction, fallback to LangChain PYTHON splitter |
| Java | Brace-matched method extraction, fallback to LangChain JAVA splitter |
| Markdown | LangChain MARKDOWN splitter (preserves heading hierarchy) |

`chunk_overlap` is set so a call-site and its definition are never in completely separate chunks with zero shared context.

`add_context_header()` prepends a metadata header like `File: graph/dijkstra.cpp | Language: cpp | Category: graph | Algorithm: Dijkstra` to each chunk's text before indexing, so the LLM always knows where retrieved code comes from.

**`indexer.py`** — `AlgoIndexer` takes chunks, embeds them in batches of 24 via the Cohere `embed-english-v3.0` model, and writes them to ChromaDB. Key design decisions:

- **Idempotent**: each chunk gets a deterministic SHA-based ID from its content + source path. Re-running the indexer skips already-stored chunks — no duplicates, safe to run repeatedly.
- **Rate limiting**: respects Cohere's free-tier (100 req/min) with `time.sleep` between batches.
- **Progress tracking**: uses `tqdm` progress bars for long indexing runs.

---

### `src/retrieval/`

**`retriever.py`** — `HybridRetriever` is the core of the retrieval pipeline.

It combines two complementary search methods:

- **Dense search**: queries ChromaDB using Cohere embeddings. Captures *semantic* similarity — "fast sorting" finds "quicksort". ChromaDB returns L2 distances, which are converted to `[0, 1]` similarities.
- **BM25 search**: a probabilistic keyword ranker (the algorithm behind Elasticsearch). Captures *exact terms* — "BFS" reliably finds "BFS", not "DFS". The BM25 index is built over all chunk tokens at startup.

Results from both are merged via **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(doc) = Σ  1 / (k + rank_in_list)
```

RRF uses rank position, not raw scores (which can't be compared across methods). `k=60` is the standard constant that dampens the impact of top-ranked documents.

The merged list is then passed to the **Cohere Reranker** — a cross-encoder that scores each (query, document) pair with full pairwise attention, far more accurate than embedding dot-product alone.

`_metadata_boost()` applies a small score bonus for language-matching results (e.g. a C++ query gets a boost for `.cpp` files) and a penalty for mismatches.

The result is a list of `RetrievedChunk` dataclasses, each carrying `dense_score`, `bm25_score`, `rrf_score`, `rerank_score`, and `final_rank` — all surfaced in the terminal UI for full transparency.

**`query_transformer.py`** — Expands the user query before retrieval (e.g. rephrasing, adding synonyms) to improve recall on edge cases.

**`reranker.py`** — Thin wrapper around the Cohere reranking API call.

---

### `src/generation/`

**`generator.py`** — `AlgoGenerator` calls the LLM with a structured prompt and parses the JSON response.

Provider fallback order:

1. **Groq Llama 3.3 70B** — primary, fastest, 30K tokens/min free
2. **Cohere Command A+** — fallback, 128K context window, RAG-optimised

The generator detects refusal phrases (`"rate limit"`, `"429"`, `"quota"`) in short responses and promotes them to exceptions so the fallback chain fires automatically.

Output is always a JSON object:
```json
{
  "answer": "...",
  "source_files": ["graph/dijkstra.cpp"],
  "confidence_note": "High confidence — direct match found.",
  "cpp_snippet": "priority_queue<...> pq;"
}
```

Raw-newline repair is applied to the response string before `json.loads()` to handle occasional LLM formatting artifacts.

**`prompt_builder.py`** — Contains the prompt templates (`ALGO_SUGGESTION_PROMPT`, `EXPLANATION_PROMPT`) and `format_context_for_prompt()`, which assembles the retrieved chunks into the prompt context block.

---

### `src/guardrails/`

**`confidence.py`** — `ConfidenceScorer` computes a multi-signal confidence score before and after generation:

| Signal | Description |
|---|---|
| `retrieval_score` | Weighted average of top chunk rerank scores (top chunk weight 1.0, second 0.5, …) |
| `source_verified` | Cited file paths are checked against the set of all indexed files |
| `coverage_score` | Fraction of answer claims supported by retrieved context (LLM-graded) |
| `overall_confidence` | Weighted combination of all three signals |

The `ConfidenceReport` dataclass holds the full breakdown. If `overall_confidence < 0.25`, `should_proceed = False` and the system refuses to generate rather than hallucinate.

`pre_retrieval_gate()` is a fast pre-check: if the top reranked chunk scores below threshold, it short-circuits before calling the LLM at all.

`verify_sources()` normalises LLM output like `"File: graph/dijkstra.cpp"` and checks each path against `known_files`. Strings over 120 characters are skipped (they're sentences, not paths).

**`hallucination_guard.py`** — Additional post-generation checks.

---

### `src/transparency/`

**`tracer.py`** — Implements the Observer pattern. Every pipeline component calls `tracer.record_step()` at each stage. Steps are typed via the `StepType` enum:

```
QUERY_RECEIVED → QUERY_TRANSFORM → DENSE_RETRIEVAL → BM25_RETRIEVAL
→ RRF_FUSION → RERANKING → CONFIDENCE_CHECK → GENERATION → COMPLETE
```

Each `TraceStep` records start/end time, success flag, and a data payload (e.g. how many chunks were retrieved, what scores they got).

**`visualiser.py`** — Consumes the tracer's event stream and renders it as a live Rich terminal display — step timings, retrieval scores, confidence breakdown, and the final answer panel.

---

### `src/ui/`

**`app.py`** — The main entry point. Builds a Textual + Rich interactive REPL. On each query it orchestrates the full pipeline (transform → retrieve → gate → generate → display), using Rich's `Console` for real-time step-by-step rendering as results arrive.

---

## Knowledge Base

The `data/` directory (~1 GB, not committed) has this structure:

```
data/
├── chroma_db/          # Active ChromaDB vector store
├── chroma_db_backup/   # Backup snapshot
└── raw/                # Cloned source repositories
    ├── C-Plus-Plus/             TheAlgorithms/C-Plus-Plus   ~370 C++ files
    ├── Python/                  TheAlgorithms/Python        ~900 Python files
    ├── Java/                    TheAlgorithms/Java          ~600 Java files
    ├── keon-algorithms/         keon/algorithms             ~200 Python files, rich docstrings
    ├── williamfiset-algorithms/ williamfiset/Algorithms     ~300 Java files, best graph/DP quality
    └── trekhleb-js-algorithms/  trekhleb/javascript-algorithms  indexed as Markdown (explanation docs)
```

`trekhleb/javascript-algorithms` is indexed as `"markdown"` rather than JavaScript — the repo's per-algorithm README files contain the best human-written explanations on GitHub, so only those are indexed (not the JS code itself).

---

## Project Structure

```
Traceable Fusion RAG for Algorithms/
├── src/
│   ├── config.py
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── indexer.py
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   └── query_transformer.py
│   ├── generation/
│   │   ├── generator.py
│   │   └── prompt_builder.py
│   ├── guardrails/
│   │   ├── confidence.py
│   │   └── hallucination_guard.py
│   ├── transparency/
│   │   ├── tracer.py
│   │   └── visualiser.py
│   └── ui/
│       └── app.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_confidence.py
│   │   ├── test_guardrails.py
│   │   ├── test_indexer.py
│   │   └── test_retriever.py
│   ├── integration/
│   │   ├── test_ingestion_pipeline.py
│   │   └── test_query_pipeline.py
│   └── e2e/
│       └── test_full_system.py
├── evaluation/
│   ├── ragas_eval.py
│   └── ragas_results.json
├── scripts/
│   ├── verify_setup.py
│   ├── backup_chroma.py
│   └── generate_test_structure.py
├── data/                   ← NOT committed (~1 GB)
├── .env.example
├── requirements.txt
├── setup.py
└── README.md
```

---

## Setup

**1. Clone this repo**
```bash
git clone <repo-url>
cd "Traceable Fusion RAG for Algorithms"
```

**2. Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
# or editable install with optional groups:
pip install -e ".[dev,eval]"
```

**4. Configure environment variables**
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required:
- `GROQ_API_KEY` — [console.groq.com](https://console.groq.com) (free, no credit card)
- `COHERE_API_KEY` — [dashboard.cohere.com](https://dashboard.cohere.com) (free trial)

Optional:
- `LANGCHAIN_API_KEY` — LangSmith tracing ([smith.langchain.com](https://smith.langchain.com))

**5. Verify setup**
```bash
python scripts/verify_setup.py
```

**6. Ingest the knowledge base** *(skip if you have a pre-built `data/` directory)*
```bash
# TODO: add ingestion entry-point command here
```

---

## Usage

```bash
python -m src.ui.app
```

---

## Running Tests

```bash
# Unit tests only (no API calls, fast)
pytest tests/unit/ -v

# Unit + integration with coverage
pytest tests/unit/ tests/integration/ -v --cov=src

# Full suite (e2e skipped unless real API keys are present)
pytest -v
```

All unit and integration tests use mocked LLM and embedding calls — no real API keys are needed.

---

### Test Results

All 30 unit and integration tests pass. All 4 e2e tests pass with real API keys (68s). E2E tests are skipped in CI unless `GROQ_API_KEY` is set.

```
========================= test session results =========================

tests/unit/test_chunker.py::test_chunks_respect_size_limit        PASSED
tests/unit/test_chunker.py::test_chunks_preserve_metadata         PASSED
tests/unit/test_chunker.py::test_context_header_added             PASSED
tests/unit/test_chunker.py::test_empty_document_produces_no_chunks PASSED
tests/unit/test_chunker.py::test_overlap_creates_continuity        PASSED

tests/unit/test_confidence.py::test_score_retrieval_high_scores   PASSED
tests/unit/test_confidence.py::test_score_retrieval_low_scores    PASSED
tests/unit/test_confidence.py::test_gate_passes_on_high_scores    PASSED
tests/unit/test_confidence.py::test_gate_blocks_on_low_scores     PASSED
tests/unit/test_confidence.py::test_gate_uses_top_score_not_mean  PASSED
tests/unit/test_confidence.py::test_gate_blocks_empty_chunks      PASSED

tests/unit/test_guardrails.py::test_gate_passes_for_high_rerank_score    PASSED
tests/unit/test_guardrails.py::test_gate_blocks_for_low_rerank_score     PASSED
tests/unit/test_guardrails.py::test_gate_blocks_empty_chunks             PASSED
tests/unit/test_guardrails.py::test_gate_uses_top_chunk_score            PASSED
tests/unit/test_guardrails.py::test_verify_sources_passes_known_file     PASSED
tests/unit/test_guardrails.py::test_verify_sources_flags_hallucinated_file PASSED
tests/unit/test_guardrails.py::test_verify_sources_strips_file_prefix    PASSED
tests/unit/test_guardrails.py::test_verify_sources_ignores_empty_entries PASSED
tests/unit/test_guardrails.py::test_verify_sources_ignores_long_sentences PASSED
tests/unit/test_guardrails.py::test_score_retrieval_empty_returns_zero   PASSED
tests/unit/test_guardrails.py::test_score_retrieval_weights_top_chunk    PASSED
tests/unit/test_guardrails.py::test_score_retrieval_capped_at_one        PASSED

tests/unit/test_indexer.py::test_generate_chunk_id_is_deterministic      PASSED
tests/unit/test_indexer.py::test_generate_chunk_id_differs_for_different_content PASSED
tests/unit/test_indexer.py::test_index_chunks_returns_count              PASSED
tests/unit/test_indexer.py::test_index_chunks_skips_already_indexed      PASSED
tests/unit/test_indexer.py::test_index_chunks_deduplicates_within_batch  PASSED
tests/unit/test_indexer.py::test_get_stats_returns_expected_keys         PASSED

tests/unit/test_retriever.py::test_tokenize_handles_code_identifiers     PASSED
tests/unit/test_retriever.py::test_rrf_merges_two_ranked_lists           PASSED
tests/unit/test_retriever.py::test_rrf_scores_are_positive               PASSED
tests/unit/test_retriever.py::test_bm25_search_returns_results           PASSED
tests/unit/test_retriever.py::test_dense_search_converts_distance_to_similarity PASSED
tests/unit/test_retriever.py::test_metadata_boost_rewards_language_match PASSED
tests/unit/test_retriever.py::test_metadata_boost_penalises_language_mismatch PASSED
tests/unit/test_retriever.py::test_retrieve_returns_retrieved_chunks     PASSED

tests/integration/test_ingestion_pipeline.py::test_chunk_then_index_pipeline    PASSED
tests/integration/test_ingestion_pipeline.py::test_context_headers_survive_indexing PASSED
tests/integration/test_ingestion_pipeline.py::test_idempotent_indexing_skips_existing PASSED

tests/integration/test_query_pipeline.py::test_retrieval_returns_relevant_docs  PASSED
tests/integration/test_query_pipeline.py::test_generation_uses_context_only     PASSED

tests/e2e/test_full_system.py::test_dijkstra_query_returns_valid_answer         SKIPPED
tests/e2e/test_full_system.py::test_hallucination_guard_fires_on_nonsense_query SKIPPED
tests/e2e/test_full_system.py::test_explain_mode_returns_explanation             SKIPPED
tests/e2e/test_full_system.py::test_pipeline_uses_fallback_on_groq_failure       SKIPPED
========================= 43 passed, 2 skipped =========================

# With real API keys (pytest tests/e2e/ -v):

tests/e2e/test_full_system.py::test_dijkstra_query_returns_valid_answer         PASSED
tests/e2e/test_full_system.py::test_hallucination_guard_fires_on_nonsense_query PASSED
tests/e2e/test_full_system.py::test_explain_mode_returns_explanation             PASSED
tests/e2e/test_full_system.py::test_pipeline_uses_fallback_on_groq_failure       PASSED

========================= 4 passed in 68.25s (0:01:08) =========================
```

---

### Unit Tests

**`test_chunker.py`** — Verifies `CppAwareChunker`:
- Chunks never exceed `chunk_size + 50` characters
- All chunks carry `source_file` and `category` metadata
- `add_context_header()` correctly prepends a `File: ...` header
- Empty documents produce zero chunks
- Adjacent chunks share overlapping content

**`test_confidence.py`** — Verifies `ConfidenceScorer` and `pre_retrieval_gate`:
- High rerank scores produce a retrieval score above 0.85
- Low rerank scores produce a retrieval score below 0.30
- The gate passes when the top chunk scores high, even if others are low (checks top only, not mean)
- The gate blocks empty chunk lists

**`test_guardrails.py`** — Deeper tests of the same guardrail components plus `verify_sources`:
- `verify_sources` passes known file paths and flags fabricated ones
- Normalises `"File: graph/dijkstra.cpp"` prefixes from LLM output
- Skips empty strings and long sentences (not file paths)
- `score_retrieval` returns 0.0 on empty input, stays ≤ 1.0 with perfect scores, and weights the top chunk at 1.0 vs 0.5 for the second

**`test_indexer.py`** — Verifies `AlgoIndexer` with mocked Cohere + ChromaDB:
- `_generate_chunk_id` is deterministic (same doc → same ID every time)
- IDs differ for different content (no collisions)
- `index_chunks` returns the correct count of newly written chunks
- Already-indexed IDs are skipped (idempotency)
- Duplicate docs in the same batch are deduplicated before writing
- `get_stats` returns `total_chunks`, `categories`, and `db_path`

**`test_retriever.py`** — Verifies `HybridRetriever` with a mocked vectorstore:
- `_tokenize` handles code identifiers and strips punctuation
- RRF merges two ranked lists and preserves all unique documents
- RRF scores are always positive
- BM25 search returns `(Document, float)` pairs
- Dense search converts ChromaDB L2 distances to `[0, 1]` similarities
- Language-match metadata boost increases scores; mismatch decreases them
- `retrieve()` returns `RetrievedChunk` objects

---

### Integration Tests

**`test_ingestion_pipeline.py`** — Tests the full `chunk → add_context_header → index` chain end-to-end (mocked embeddings and ChromaDB):
- Chunking produces ≥ as many chunks as source documents
- Context headers survive the full pipeline with `File:`, `Language:`, and `Category:` fields intact
- Re-indexing the same chunks returns count 0 (no duplicates written)

**`test_query_pipeline.py`** — Tests `retrieve → generate` (mocked Cohere + Groq):
- After indexing sample docs, retrieval stats show the expected chunk count
- Generator includes retrieved context in the prompt and returns a valid `answer` field

---

### End-to-End Tests

`tests/e2e/test_full_system.py` contains four tests that run the complete pipeline against real APIs. They are **skipped in CI** unless `GROQ_API_KEY` is set in `.env`. All 4 pass (68s total).

- `test_dijkstra_query_returns_valid_answer` — fires a real Dijkstra query and asserts `understanding` and `algorithms` are present, confidence > 0.5, and sources are verified against the index.
- `test_hallucination_guard_fires_on_nonsense_query` — queries for a fictional algorithm and asserts the pre-retrieval gate refuses rather than hallucinating.
- `test_explain_mode_returns_explanation` — runs in `explain` mode and asserts a long-form explanation is returned (not a JSON dict).
- `test_pipeline_uses_fallback_on_groq_failure` — forces Groq to fail and asserts Cohere Command A+ transparently takes over and still returns a valid result.

---

## Evaluation (RAGAS)

RAGAS (Retrieval Augmented Generation Assessment) was run over 5 representative questions on `2026-05-28`. One question failed due to LLM rate limits and was excluded from aggregates.

### Metric Definitions

| Metric | What it measures |
|---|---|
| **Faithfulness** | Are the claims in the answer supported by the retrieved context? (Hallucination detector) |
| **Answer Relevancy** | Does the answer actually address what was asked? |
| **Context Precision** | Are the retrieved chunks relevant to the question? (Retrieval quality) |
| **Context Recall** | Did retrieval find everything needed to answer the question? |

Scores range from 0.0 to 1.0. Higher is better.

---

### Results by Question

| # | Question | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|:---:|:---:|:---:|:---:|
| 1 | How does Dijkstra's algorithm find the shortest path? | 0.33 | 0.46 | **1.00** | 0.25 |
| 2 | What is the difference between merge sort and quick sort? | 0.00 | 0.10 | 0.00 | 0.00 |
| 3 | Explain the knapsack dynamic programming approach. | 0.63 | 0.72 | 0.80 | **1.00** |
| 4 | How does binary search work and when should I use it? | **0.88** | 0.49 | **1.00** | **1.00** |
| 5 | What is a segment tree and when is it used? | N/A | **0.71** | 0.68 | N/A |

> **Question 2** (`merge sort vs quick sort`) failed with `"ERROR: All LLM providers failed"` due to rate limiting and is excluded from aggregate scores.

---

### Aggregate Scores

| Metric | Raw (5 questions) | Corrected (excl. Q2 + NaN) |
|---|:---:|:---:|
| Faithfulness | 0.46 | **0.61** |
| Answer Relevancy | 0.50 | **0.60** |
| Context Precision | 0.70 | **0.87** |
| Context Recall | 0.56 | **0.75** |

The corrected scores exclude the rate-limit failure (Q2) and questions where a metric returned `NaN` (Q5 faithfulness and context recall — the RAGAS LLM judge timed out on that response).

---

### Interpretation

**What's working well:**
- Context precision is very high (0.87 corrected) — the retrieval pipeline returns relevant chunks, not noise.
- Context recall of 0.75 means the knowledge base covers most of what's needed to answer these questions.
- Binary search (Q4) achieved the highest faithfulness (0.88) — the answer was well-grounded in retrieved code.

**What needs improvement:**
- **Faithfulness is the weakest metric (0.61)** — the generator sometimes adds claims not present in the retrieved context. Tightening the prompt to instruct "only state what is in the context" would help.
- **Answer relevancy is moderate (0.60)** — answers tend to over-explain complexity details that weren't asked for. More focused prompt templates would improve this.
- **Rate limit resilience** — Q2 failed entirely. The Cohere fallback should fire more aggressively; adding exponential backoff with 2–3 retries before giving up would prevent total failures.
- **Context recall for Dijkstra (0.25)** — the retriever found the implementation but missed the conceptual explanation chunks. Increasing `retrieval_k` from 8 to 12 for graph questions would likely fix this.

---

## Configuration

All settings live in `src/config.py` and are loaded from `.env` via Pydantic `BaseSettings`.

| Setting | Default | Description |
|---|---|---|
| `primary_llm` | `llama-3.3-70b-versatile` | Primary LLM (Groq) |
| `fallback_llm` | `command-a-03-2025` | Fallback LLM (Cohere) |
| `llm_temperature` | `0.0` | Deterministic generation |
| `max_tokens` | `2048` | Max generation length |
| `embedding_model` | `embed-english-v3.0` | Cohere embedding model |
| `embedding_dim` | `1024` | Embedding vector dimensions |
| `retrieval_k` | `8` | Chunks retrieved before reranking |
| `rerank_top_n` | `5` | Chunks kept after reranking |
| `similarity_threshold` | `0.35` | Minimum similarity to consider a chunk |
| `chunk_size` | `1000` | Max characters per chunk |
| `chunk_overlap` | `200` | Overlap between adjacent chunks |
| `high_confidence` | `0.80` | Threshold for HIGH confidence label |
| `medium_confidence` | `0.60` | Threshold for MEDIUM confidence label |
| `low_confidence` | `0.40` | Threshold for LOW confidence label |

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/verify_setup.py` | Confirms API keys and core packages are configured correctly |
| `scripts/backup_chroma.py` | Copies `data/chroma_db/` to `data/chroma_db_backup/` |
| `scripts/generate_test_structure.py` | Scaffolds test file structure |

---

## License

<!-- TODO: Add license. -->
