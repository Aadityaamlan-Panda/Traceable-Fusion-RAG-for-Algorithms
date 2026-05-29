"""
CONCEPT: The Rich library renders beautiful terminal output with:
- Panels, tables, progress bars, syntax highlighting
- Live-updating displays for streaming
- Tree structures for hierarchical data

We use this to render the RAG trace as the pipeline executes,
creating the "glass box" effect in real time.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.tree import Tree
from rich.text import Text
from rich.columns import Columns
from rich import box
from rich.live import Live
from rich.layout import Layout
import time

from .tracer import TraceStep, StepType, RAGTrace

console = Console()


# ── Step Icons ─────────────────────────────────────────
STEP_ICONS = {
    StepType.QUERY_RECEIVED:   "🔍",
    StepType.QUERY_TRANSFORM:  "🔄",
    StepType.DENSE_RETRIEVAL:  "⚡",
    StepType.BM25_RETRIEVAL:   "📝",
    StepType.RRF_FUSION:       "🔀",
    StepType.RERANKING:        "🎯",
    StepType.CONFIDENCE_CHECK: "🛡️",
    StepType.GENERATION:       "🤖",
    StepType.COMPLETE:         "✅",
    StepType.ERROR:            "❌",
}

STEP_COLORS = {
    StepType.QUERY_RECEIVED:   "bold cyan",
    StepType.DENSE_RETRIEVAL:  "blue",
    StepType.BM25_RETRIEVAL:   "blue",
    StepType.RRF_FUSION:       "magenta",
    StepType.RERANKING:        "yellow",
    StepType.CONFIDENCE_CHECK: "green",
    StepType.GENERATION:       "bright_green",
    StepType.COMPLETE:         "bold green",
    StepType.ERROR:            "red",
}


def render_query_banner(query: str):
    """Display the initial query banner."""
    console.print()
    console.print(Panel(
        f"[bold white]{query}[/bold white]",
        title="[bold cyan]🔍 AlgoRAG Query[/bold cyan]",
        border_style="cyan",
        padding=(0, 2)
    ))
    console.print()


def render_step(step: TraceStep):
    """Render a completed pipeline step to the terminal."""
    icon = STEP_ICONS.get(step.step_type, "•")
    color = STEP_COLORS.get(step.step_type, "white")
    
    status = "✓" if step.success else "✗"
    duration = f"{step.duration_ms:.0f}ms" if step.duration_ms > 0 else ""
    
    # Step header line
    header = Text()
    header.append(f"  [{step.step_number}] ", style="dim white")
    header.append(f"{icon} ", style="")
    header.append(f"{step.description}", style=color)
    header.append(f"  {status}", style="green" if step.success else "red")
    header.append(f"  {duration}", style="dim")
    
    console.print(header)
    
    # Step-specific detail rendering
    if step.step_type == StepType.DENSE_RETRIEVAL:
        _render_retrieval_detail(step, "Dense")
    
    elif step.step_type == StepType.BM25_RETRIEVAL:
        _render_retrieval_detail(step, "BM25")
    
    elif step.step_type == StepType.RRF_FUSION:
        data = step.data
        console.print(
            f"     Fused {data.get('total_candidates', 0)} unique candidates "
            f"({data.get('overlap', 0)} overlapping)",
            style="dim"
        )
    
    elif step.step_type == StepType.RERANKING:
        _render_reranking_detail(step)
    
    elif step.step_type == StepType.CONFIDENCE_CHECK:
        confidence = step.data.get("confidence", 0)
        level = step.data.get("level", "unknown")
        bar = _confidence_bar(confidence)
        color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"
        console.print(f"     Confidence: [{color}]{bar} {confidence:.0%}[/{color}]  [{level}]")
    
    elif step.step_type == StepType.GENERATION:
        data = step.data
        console.print(
            f"     Model: {data.get('model', 'unknown')} | "
            f"Tokens: {data.get('tokens', '?')} | "
            f"Chunks used: {data.get('chunks_used', '?')}",
            style="dim"
        )
    
    elif step.step_type == StepType.ERROR:
        console.print(f"     [red]{step.error}[/red]")


def _render_retrieval_detail(step: TraceStep, method: str):
    """Render retrieval results as a mini-table."""
    top_results = step.data.get("top_results", [])
    if not top_results:
        return
    
    for r in top_results[:3]:
        score_bar = _score_bar(r.get("score", 0))
        console.print(
            f"     {score_bar} {r.get('score', 0):.3f}  "
            f"[dim]{r.get('file', 'unknown')}[/dim]  "
            f"[italic]{r.get('function', '')}[/italic]"
        )


def _render_reranking_detail(step: TraceStep):
    """Render reranking results with score bars."""
    results = step.data.get("results", [])
    
    table = Table(
        show_header=False, 
        box=None, 
        padding=(0, 1),
        show_edge=False
    )
    table.add_column("rank", style="dim", width=3)
    table.add_column("bar", width=14)
    table.add_column("score", width=6)
    table.add_column("file", style="dim cyan")
    
    for r in results[:5]:
        score = r.get("score", 0)
        rank = r.get("rank", 0)
        bar = _score_bar(score, width=12)
        table.add_row(
            f"#{rank}",
            bar,
            f"{score:.3f}",
            r.get("file", "unknown")
        )
    
    console.print("    ", table)


def _score_bar(score: float, width: int = 10) -> str:
    """Create a text-based score bar."""
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[cyan]{bar}[/cyan]"


def _confidence_bar(score: float, width: int = 16) -> str:
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def render_pipeline_summary(trace: RAGTrace):
    """Render the final pipeline summary table."""
    console.print()
    
    table = Table(
        title="[bold]📊 Pipeline Trace Summary[/bold]",
        box=box.ROUNDED,
        border_style="dim cyan",
        show_header=True,
        header_style="bold"
    )
    table.add_column("Step", style="cyan")
    table.add_column("Duration", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Key Metric")
    
    for step in trace.steps:
        icon = STEP_ICONS.get(step.step_type, "•")
        status = "✅" if step.success else "❌"
        duration = f"{step.duration_ms:.0f}ms"
        
        # Key metric per step type
        metric = ""
        if step.step_type == StepType.DENSE_RETRIEVAL:
            metric = f"{step.data.get('count', 0)} retrieved"
        elif step.step_type == StepType.RERANKING:
            metric = f"top score: {step.data.get('top_score', 0):.3f}"
        elif step.step_type == StepType.CONFIDENCE_CHECK:
            metric = f"confidence: {step.data.get('confidence', 0):.0%}"
        elif step.step_type == StepType.GENERATION:
            metric = f"{step.data.get('tokens', '?')} tokens"
        
        table.add_row(f"{icon} {step.description}", duration, status, metric)
    
    # Total row
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{sum(s.duration_ms for s in trace.steps if s.duration_ms > 0):.0f}ms[/bold]",
        "",
        f"[bold]{len(trace.steps)} steps[/bold]"
    )
    
    console.print(table)
    console.print()


def render_algorithm_results(result: dict):
    """Render the final algorithm suggestions beautifully."""
    if not result.get("algorithms"):
        console.print(Panel(
            "[yellow]No suitable algorithms found in the knowledge base for this query.[/yellow]\n"
            "[dim]Try rephrasing or using more specific terms.[/dim]",
            title="[yellow]⚠️ No Results[/yellow]",
            border_style="yellow"
        ))
        return
    
    # Understanding panel
    console.print(Panel(
        f"[italic]{result.get('understanding', '')}[/italic]",
        title="[bold green]💡 Understanding[/bold green]",
        border_style="green",
        padding=(0, 1)
    ))
    
    # Algorithm panels
    for i, algo in enumerate(result.get("algorithms", []), 1):
        confidence = algo.get("confidence", 0)
        conf_bar = _confidence_bar(confidence, 12)
        conf_color = "green" if confidence > 0.7 else "yellow"
        
        content = (
            f"[bold]📁 Source:[/bold] {algo.get('source_file', 'N/A')}\n"
            f"[bold]⏱️  Time:[/bold]  {algo.get('time_complexity', 'N/A')}\n"
            f"[bold]💾 Space:[/bold] {algo.get('space_complexity', 'N/A')}\n"
            f"[bold]🎯 Reason:[/bold] {algo.get('reason', '')}\n"
            f"[bold]🛡️  Confidence:[/bold] [{conf_color}]{conf_bar} {confidence:.0%}[/{conf_color}]\n\n"
        )
        
        # Syntax-highlighted C++ code
        code = algo.get("code", "")
        if code:
            # Detect language from source file extension or metadata
            source_file = algo.get("source_file", "")
            if source_file.endswith(".java"):
                syntax_lang = "java"
            elif source_file.endswith(".py"):
                syntax_lang = "python"
            elif source_file.endswith(".cpp") or source_file.endswith(".cc"):
                syntax_lang = "cpp"
            else:
                syntax_lang = "text"
            # Use console.print with markup=True so [bold]/[green] tags render
            # correctly. Text(content) strips markup — don't use it here.
            console.print(Panel(
                content,
                title=f"[bold cyan]#{i} {algo.get('name', 'Algorithm')}[/bold cyan]",
                border_style="cyan"
            ))
            syntax = Syntax(code.strip(), syntax_lang, theme="monokai", line_numbers=True, word_wrap=True)
            console.print(Panel(syntax, border_style="dim", padding=(0, 1)))
    
    # Comparison
    if result.get("comparison"):
        console.print(Panel(
            result["comparison"],
            title="[bold yellow]⚖️  Comparison[/bold yellow]",
            border_style="yellow"
        ))
    
    # Caveats
    if result.get("caveats"):
        console.print(Panel(
            f"[dim]{result['caveats']}[/dim]",
            title="[bold]⚠️  Caveats & Edge Cases[/bold]",
            border_style="dim"
        ))