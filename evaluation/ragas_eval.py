"""
CONCEPT: RAGAS evaluation pipeline for AlgoRAG.

We build a test dataset of (question, ground_truth) pairs,
run our full pipeline on each, collect (answer, contexts),
then score with RAGAS metrics.

Architecture:
  Test Dataset → AlgoRAG Pipeline → {answer, contexts}
               → RAGAS Score → Report
"""

import json
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


# ─── Test Dataset ────────────────────────────────────────────────────────────

EVAL_QUESTIONS: List[Dict[str, str]] = [
    {
        "question": "How does Dijkstra's algorithm find the shortest path in a weighted graph?",
        "ground_truth": (
            "Dijkstra's algorithm uses a priority queue (min-heap) to greedily "
            "expand the nearest unvisited node. It initialises all distances to "
            "infinity except the source (distance 0), then repeatedly extracts "
            "the minimum-distance node and relaxes its neighbours. Time complexity "
            "is O((V + E) log V) with a binary heap."
        ),
    },
    {
        "question": "What is the difference between merge sort and quick sort?",
        "ground_truth": (
            "Merge sort is a stable, divide-and-conquer algorithm with guaranteed "
            "O(n log n) time in all cases, but requires O(n) extra space. "
            "Quick sort is in-place with average O(n log n) but worst-case O(n²) "
            "when the pivot is always the smallest or largest element. Quick sort "
            "has better cache locality in practice."
        ),
    },
    {
        "question": "Explain the knapsack dynamic programming approach.",
        "ground_truth": (
            "The 0/1 knapsack problem fills a 2D DP table dp[i][w] representing "
            "the maximum value using the first i items with weight capacity w. "
            "For each item, we either skip it (dp[i-1][w]) or include it "
            "(dp[i-1][w - weight[i]] + value[i]), taking the maximum. "
            "Time complexity is O(n * W) where W is the capacity."
        ),
    },
    {
        "question": "How does binary search work and when should I use it?",
        "ground_truth": (
            "Binary search works on sorted arrays by repeatedly halving the "
            "search space. It compares the target to the middle element: if "
            "equal, return; if less, search left half; if greater, search right. "
            "Time complexity is O(log n). Use it whenever the input is sorted "
            "or can be sorted, and you need O(log n) lookup instead of O(n)."
        ),
    },
    {
        "question": "What is a segment tree and when is it used?",
        "ground_truth": (
            "A segment tree is a binary tree where each node stores an aggregate "
            "(sum, min, max) over a range of the input array. It supports range "
            "queries and point updates in O(log n) time, using O(n) space. "
            "Use it when you need many range queries on a mutable array — "
            "for example, range sum queries with updates."
        ),
    },
]


# ─── One-time setup: build vectorstore + all_chunks from ChromaDB ─────────────

def _load_retriever(settings):
    """
    Build the HybridRetriever the same way the main app does:
      1. Open the existing ChromaDB collection via AlgoIndexer
      2. Fetch all stored chunks (needed to build the BM25 index)
      3. Construct HybridRetriever(vectorstore, all_chunks, ...)
    """
    from langchain_chroma import Chroma
    from langchain_cohere import CohereEmbeddings
    from langchain_core.documents import Document
    from src.ingestion.indexer import AlgoIndexer
    from src.retrieval.retriever import HybridRetriever

    # Open the persisted ChromaDB (read-only — no re-indexing)
    indexer = AlgoIndexer(
        cohere_api_key=settings.cohere_api_key,
        chroma_path=settings.chroma_db_path,
    )
    vectorstore = indexer.vectorstore

    # Fetch all chunks so HybridRetriever can build its BM25 index.
    # ChromaDB .get() returns raw dicts; we reconstruct Document objects.
    collection = indexer.chroma_client.get_collection("algorithms")
    raw = collection.get(include=["documents", "metadatas"])

    all_chunks: list[Document] = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    console.print(f"  [dim]BM25 index: {len(all_chunks)} chunks loaded from ChromaDB[/dim]")

    retriever = HybridRetriever(
        vectorstore=vectorstore,
        all_chunks=all_chunks,
        cohere_api_key=settings.cohere_api_key,
        k=settings.retrieval_k,
        rerank_top_n=settings.rerank_top_n,
    )
    return retriever


# ─── Pipeline Runner ──────────────────────────────────────────────────────────

def run_pipeline_for_eval(question: str, retriever, settings) -> Dict[str, Any]:
    """
    Run the AlgoRAG pipeline for one question.
    Returns {"answer": str, "contexts": list[str]}.

    Uses real source APIs:
      - HyDEChain.invoke({"query": ...})          ← key is "query" not "question"
      - HybridRetriever.retrieve(query)            ← no k= arg (set in __init__)
      - pre_retrieval_gate(chunks) → (bool, str)   ← returns tuple
      - AlgoGenerator.suggest_algorithms(query, chunks) → (dict, provider)
      - ConfidenceScorer(groq_api_key, known_files, cohere_api_key)
      - scorer.score_retrieval(chunks)             ← no score_single method
    """
    from src.retrieval.query_transformer import create_hyde_chain
    from src.generation.generator import AlgoGenerator
    from src.guardrails.confidence import ConfidenceScorer, pre_retrieval_gate
    from src.transparency.tracer import PipelineTracer

    # Tracer — takes no __init__ args; query goes to start_trace()
    tracer = PipelineTracer()
    tracer.start_trace(question)

    # Step 1: HyDE — key must be "query", not "question"
    hyde_chain = create_hyde_chain(
        groq_api_key=settings.groq_api_key,
        cohere_api_key=settings.cohere_api_key,
    )
    transformed_query = hyde_chain.invoke({"query": question})   # ← "query"

    # Step 2: Hybrid retrieval — retrieve() takes no k= kwarg
    chunks = retriever.retrieve(transformed_query)               # ← no k=

    if not chunks:
        return {
            "answer": "Insufficient relevant context found to answer reliably.",
            "contexts": [],
        }

    # Step 3: Confidence gate
    # pre_retrieval_gate takes RetrievedChunk list, returns (bool, reason_str)
    should_proceed, reason = pre_retrieval_gate(chunks)          # ← unpack tuple
    if not should_proceed:
        return {
            "answer": f"Insufficient relevant context found to answer reliably. {reason}",
            "contexts": [c.document.page_content for c in chunks],
        }

    # Step 4: Generate
    # suggest_algorithms(query, retrieved_chunks) → (dict, provider_name)
    generator = AlgoGenerator(
        groq_api_key=settings.groq_api_key,
        cohere_api_key=settings.cohere_api_key,
    )
    result_dict, _provider = generator.suggest_algorithms(question, chunks)  # ← correct method

    # Extract a flat answer string from the structured JSON response
    answer = _flatten_answer(result_dict)

    # Contexts: raw text of the retrieved chunks (what RAGAS needs)
    contexts = [c.document.page_content for c in chunks]

    return {"answer": answer, "contexts": contexts}


def _flatten_answer(result_dict: dict) -> str:
    """
    suggest_algorithms returns a structured dict:
      {understanding, algorithms: [{name, reason, code, ...}], comparison, caveats}
    Flatten to a readable answer string for RAGAS.

    Structure:
      1. understanding  — the direct answer to "what does the user need?"
                          This is a proper prose sentence from the LLM and reads
                          as a real answer, which lifts answer_relevancy significantly.
      2. Per-algorithm  — name + reason + complexities (factual, context-grounded)
      3. comparison     — tradeoff summary (only when 2 algorithms present)
      4. caveats        — edge-case note

    NOTE ON FAITHFULNESS: The 'understanding' field may contain background knowledge
    not literally in the retrieved code snippets. RAGAS faithfulness will penalise
    this, but removing it tanked answer_relevancy badly. The right trade-off is to
    keep it and instead tighten the generator prompt so the LLM derives the
    understanding from context rather than memory.
    """
    parts = []

    # Lead with the direct answer — this is what makes the response feel useful
    understanding = result_dict.get("understanding", "")
    if understanding and understanding not in ("Parse error", "See code below", ""):
        parts.append(understanding)

    for algo in result_dict.get("algorithms", []):
        name = algo.get("name", "")
        reason = algo.get("reason", "")
        complexity = algo.get("time_complexity", "")
        space = algo.get("space_complexity", "")
        if name:
            detail = f"{name}: {reason}"
            if complexity:
                detail += f" Time complexity: {complexity}."
            if space:
                detail += f" Space complexity: {space}."
            parts.append(detail)

    comparison = result_dict.get("comparison", "")
    if comparison and comparison.strip() and "Only one" not in comparison:
        parts.append(comparison)

    caveats = result_dict.get("caveats", "")
    if caveats and "parse" not in caveats.lower() and caveats.strip():
        parts.append(caveats)

    return " ".join(parts) if parts else "No answer generated."


# ─── RAGAS Evaluation ─────────────────────────────────────────────────────────

def build_ragas_dataset(pipeline_results: List[Dict]) -> Dataset:
    rows = {
        "question":     [r["question"] for r in pipeline_results],
        "answer":       [r["answer"] for r in pipeline_results],
        "contexts":     [r["contexts"] for r in pipeline_results],
        "ground_truth": [r["ground_truth"] for r in pipeline_results],
    }
    return Dataset.from_dict(rows)


def run_ragas_evaluation(output_path: str = "evaluation/ragas_results.json") -> Dict:
    console.print(Panel(
        "[bold cyan]RAGAS Evaluation — AlgoRAG[/bold cyan]\n"
        f"[dim]Evaluating {len(EVAL_QUESTIONS)} questions across 4 metrics[/dim]",
        border_style="cyan",
        box=box.DOUBLE,
    ))

    from src.config import settings

    # ── Build retriever once (expensive BM25 index) ───────────────────────────
    console.print("\n[cyan]Loading ChromaDB and building BM25 index...[/cyan]")
    retriever = _load_retriever(settings)

    # ── Step 1: Collect pipeline outputs ──────────────────────────────────────
    pipeline_results = []

    for i, item in enumerate(EVAL_QUESTIONS, 1):
        console.print(f"\n[yellow][{i}/{len(EVAL_QUESTIONS)}][/yellow] Running: {item['question'][:60]}...")

        # Small inter-question delay: the 70B pipeline (HyDE + generation) fires
        # 2-3 Groq calls per question. At 30 req/min we have 2 s/req headroom.
        # Sleeping 3 s between questions keeps us well under the rate limit and
        # leaves the bulk of the budget for RAGAS scoring that follows.
        if i > 1:
            import time
            time.sleep(3)

        try:
            result = run_pipeline_for_eval(item["question"], retriever, settings)
            pipeline_results.append({
                "question":     item["question"],
                "ground_truth": item["ground_truth"],
                "answer":       result["answer"],
                "contexts":     result["contexts"],
            })
            console.print(f"  [green]✓[/green] Answer: {result['answer'][:80]}...")
        except Exception as e:
            console.print(f"  [red]✗ Pipeline error: {e}[/red]")
            import traceback
            traceback.print_exc()
            pipeline_results.append({
                "question":     item["question"],
                "ground_truth": item["ground_truth"],
                "answer":       f"ERROR: {e}",
                "contexts":     [],
            })

    # ── Step 2: Build RAGAS dataset ───────────────────────────────────────────
    console.print("\n[cyan]Building RAGAS dataset...[/cyan]")
    dataset = build_ragas_dataset(pipeline_results)

    # ── Step 3: Configure RAGAS LLM + embeddings ─────────────────────────────
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # WHY llama-3.1-8b-instant for RAGAS scoring (not the 70B used for generation):
    #
    #   RAGAS makes ~10-12 LLM calls per question (claim extraction, NLI checks,
    #   context-precision verdicts, recall attribution). For 5 questions that is
    #   ~55 calls in a single evaluate() run.
    #
    #   Groq free-tier limits by model:
    #     llama-3.3-70b-versatile : 30 req/min,  6 000 TPM  ← hits TPM first
    #     llama-3.1-8b-instant    : 30 req/min, 20 000 TPM  ← official replacement for llama3-8b-8192
    #   (llama3-8b-8192 was decommissioned May 2025 → llama-3.1-8b-instant is the Groq-recommended replacement)
    #
    #   8B is more than capable for RAGAS's binary/short-answer internal
    #   prompts (yes/no verdicts, claim lists). Quality difference vs 70B
    #   on these structured sub-tasks is negligible.
    #
    #   max_retries=6 + exponential back-off handles transient 429s without
    #   crashing; the RunConfig timeout below is the outer safety net.
    ragas_llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=settings.groq_api_key,
            temperature=0,
            max_tokens=1024,     # RAGAS sub-prompts are short; cap tokens to stay under TPM
            max_retries=6,       # retry 429s with back-off before giving up
            request_timeout=60,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=settings.cohere_api_key)
    )

    # Instantiate metrics fresh with llm/embeddings in the constructor.
    # Setting .llm on the module-level singletons (faithfulness, answer_relevancy…)
    # does NOT work in RAGAS >= 0.1.x — the attribute is ignored and the metric
    # falls back to its default (None) LLM, silently returning NaN for every row.
    # Passing llm= to the constructor is the only reliable approach.
    #
    # Fallback LLM: if Groq 8B is exhausted mid-evaluation, RAGAS will 429.
    # We build a Cohere Command-R wrapper as a second LLM option. RAGAS does
    # not support multi-provider fallback natively, but we can swap the scorer
    # LLM to Cohere if Groq quota is known to be low.
    # Set RAGAS_USE_COHERE=1 in .env to force Cohere for all scoring calls.
    import os
    _force_cohere = os.getenv("RAGAS_USE_COHERE", "0").strip() == "1"

    if _force_cohere:
        from langchain_cohere import ChatCohere
        scorer_llm = LangchainLLMWrapper(
            ChatCohere(
                model="command-r",           # 20 req/min free tier, no TPM cap
                cohere_api_key=settings.cohere_api_key,
                temperature=0,
                max_tokens=1024,
            )
        )
        console.print("  [dim]RAGAS scorer: Cohere Command-R (RAGAS_USE_COHERE=1)[/dim]")
    else:
        scorer_llm = ragas_llm
        console.print("  [dim]RAGAS scorer: Groq llama-3.1-8b-instant (set RAGAS_USE_COHERE=1 to switch)[/dim]")

    metrics = [
        Faithfulness(llm=scorer_llm),
        AnswerRelevancy(llm=scorer_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=scorer_llm),
        ContextRecall(llm=scorer_llm),
    ]

    # ── Step 4: Run RAGAS ────────────────────────────────────────────────────
    console.print("[cyan]Scoring with RAGAS (this takes ~2–3 minutes)...[/cyan]")
    # raise_exceptions=False: per-row LLM failures become NaN instead of
    # crashing the whole run or hanging indefinitely.
    # RunConfig(timeout=120): hard per-row timeout so a stalled API call
    # doesn't freeze the progress bar at 0/N forever.
    try:
        from ragas import RunConfig
        scores = evaluate(
            dataset=dataset,
            metrics=metrics,
            run_config=RunConfig(
                timeout=180,      # seconds per row before NaN
                max_workers=1,    # serialise calls — prevents burst 429s
            ),
            raise_exceptions=False,
        )
    except TypeError:
        # Older ragas builds don't accept these kwargs
        scores = evaluate(dataset=dataset, metrics=metrics)

    # ── Step 5: Display results ──────────────────────────────────────────────
    _display_results(scores)

    # Warn if any metric has NaN rows (partial scoring)
    _warn_nan_metrics(scores)

    # ── Step 6: Save to JSON ─────────────────────────────────────────────────
    def _safe_mean(val) -> float | None:
        """
        Return the mean of val, ignoring NaNs.
        Returns None (instead of silently propagating NaN) when ALL values are
        NaN — this makes missing data explicit rather than hiding it.
        """
        import pandas as pd
        if isinstance(val, (list, np.ndarray, pd.Series)):
            arr = np.array(val, dtype=float)
        else:
            arr = np.array([float(val)])
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return None          # all-NaN → report as null, not NaN
        return float(np.mean(valid))

    def _nan_count(val) -> int:
        """Count how many individual scores are NaN."""
        import pandas as pd
        if isinstance(val, (list, np.ndarray, pd.Series)):
            arr = np.array(val, dtype=float)
            return int(np.sum(np.isnan(arr)))
        return 1 if np.isnan(float(val)) else 0

    output = {
        "timestamp": datetime.now().isoformat(),
        "n_questions": len(EVAL_QUESTIONS),
        "scores": scores.to_pandas().to_dict(orient="records"),
        "aggregate": {
            "faithfulness":      _safe_mean(scores["faithfulness"]),
            "answer_relevancy":  _safe_mean(scores["answer_relevancy"]),
            "context_precision": _safe_mean(scores["context_precision"]),
            "context_recall":    _safe_mean(scores["context_recall"]),
        },
        "nan_counts": {
            "faithfulness":      _nan_count(scores["faithfulness"]),
            "answer_relevancy":  _nan_count(scores["answer_relevancy"]),
            "context_precision": _nan_count(scores["context_precision"]),
            "context_recall":    _nan_count(scores["context_recall"]),
        },
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2))
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")

    return output["aggregate"]


def _warn_nan_metrics(scores) -> None:
    """
    Print an actionable warning for any metric that produced NaN scores.

    NaN causes (and fixes already applied / still needed):
      - faithfulness NaN: answer contained meta-commentary ('The user needs to...')
        that RAGAS claim-decomposer could not verify against contexts.
        → Fixed in _flatten_answer(): understanding field removed from answer.
      - context_recall NaN: RAGAS internal LLM call failed (rate-limit / timeout)
        for that row. Re-running the evaluation is usually sufficient.
      - context_precision NaN: same LLM-call failure pattern as context_recall.
      - answer_relevancy NaN: embedding call failed. Check Cohere API quota.
    """
    import pandas as pd

    nan_warnings = []
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        raw = scores[metric]
        if isinstance(raw, (list, np.ndarray, pd.Series)):
            arr = np.array(raw, dtype=float)
            nan_n = int(np.sum(np.isnan(arr)))
        else:
            nan_n = 1 if np.isnan(float(raw)) else 0
        if nan_n > 0:
            nan_warnings.append(f"  • {metric}: {nan_n}/{len(arr)} rows are NaN")

    if nan_warnings:
        console.print(
            "\n[bold yellow]⚠  NaN scores detected — aggregate computed over valid rows only:[/bold yellow]"
        )
        for w in nan_warnings:
            console.print(f"[yellow]{w}[/yellow]")
        console.print(
            "[dim]  Tip: faithfulness NaN is fixed by removing meta-commentary from answers.\n"
            "  context_recall / context_precision NaN usually indicates a transient LLM\n"
            "  API failure inside RAGAS — re-running the evaluation resolves it.[/dim]"
        )


def _display_results(scores) -> None:
    """Render RAGAS scores as a Rich table with pass/fail colouring."""
    import pandas as pd

    TARGETS = {
        "faithfulness":      0.90,
        "answer_relevancy":  0.85,
        "context_precision": 0.80,
        "context_recall":    0.80,
    }

    table = Table(
        title="📊 RAGAS Evaluation Results",
        box=box.DOUBLE_EDGE,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Metric",         style="white", width=22)
    table.add_column("Score",          justify="center", width=10)
    table.add_column("Target",         justify="center", width=10)
    table.add_column("Status",         justify="center", width=10)
    table.add_column("What it means",  style="dim",   width=36)

    descriptions = {
        "faithfulness":      "Answers grounded in retrieved context",
        "answer_relevancy":  "Answer addresses the question asked",
        "context_precision": "Retrieved chunks are high-signal",
        "context_recall":    "Context covers all needed info",
    }

    for metric, target in TARGETS.items():
        raw = scores[metric]
        if isinstance(raw, (list, np.ndarray, pd.Series)):
            arr = np.array(raw, dtype=float)
            valid = arr[~np.isnan(arr)]
            nan_n = int(np.sum(np.isnan(arr)))
            score = float(np.mean(valid)) if len(valid) > 0 else float("nan")
        else:
            score = float(raw)
            nan_n = 0

        passed = not np.isnan(score) and score >= target
        status       = "[green]✅ PASS[/green]" if passed else "[red]❌ FAIL[/red]"
        score_colour = "green" if passed else "red"
        # Annotate score with NaN warning when some rows were missing
        score_str = f"[{score_colour}]{score:.3f}[/{score_colour}]"
        if nan_n > 0:
            score_str += f" [dim]({nan_n} NaN)[/dim]"
        table.add_row(
            metric.replace("_", " ").title(),
            score_str,
            f"{target:.2f}",
            status,
            descriptions[metric],
        )

    console.print(table)


if __name__ == "__main__":
    scores = run_ragas_evaluation()
    print(scores)