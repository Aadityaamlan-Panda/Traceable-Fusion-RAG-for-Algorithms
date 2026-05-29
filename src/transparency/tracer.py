"""
CONCEPT: The Tracer records every step of the RAG pipeline as it runs.
It's a structured event log that powers the terminal visualisation.

Design pattern: Observer / Event Emitter
- The tracer is passed to every component
- Components call tracer.record_step() as they execute
- The UI subscribes to the tracer's event stream
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class StepType(Enum):
    QUERY_RECEIVED = "query_received"
    QUERY_TRANSFORM = "query_transform"
    DENSE_RETRIEVAL = "dense_retrieval"
    BM25_RETRIEVAL = "bm25_retrieval"
    RRF_FUSION = "rrf_fusion"
    RERANKING = "reranking"
    CONFIDENCE_CHECK = "confidence_check"
    GENERATION = "generation"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class TraceStep:
    """A single step in the RAG pipeline trace."""
    step_type: StepType
    step_number: int
    description: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    data: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return -1
    
    def complete(self, data: dict = None, success: bool = True, error: str = None):
        self.end_time = time.time()
        self.success = success
        self.error = error
        if data:
            self.data.update(data)
        return self


@dataclass 
class RAGTrace:
    """Complete trace of a single RAG query execution."""
    query: str
    trace_id: str = field(default_factory=lambda: str(time.time_ns()))
    steps: list[TraceStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    final_answer: Optional[dict] = None
    
    @property
    def total_duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return -1


class PipelineTracer:
    """
    Records and streams RAG pipeline steps for terminal visualisation.
    
    Usage:
        tracer = PipelineTracer()
        trace = tracer.start_trace("find quicksort")
        
        step = tracer.begin_step(StepType.DENSE_RETRIEVAL, "Vector search")
        results = vectorstore.search(query)  # actual work
        tracer.end_step(step, data={"count": len(results)})
    """
    
    def __init__(self):
        self.current_trace: Optional[RAGTrace] = None
        self.step_counter = 0
        self._callbacks: list = []
    
    def on_step(self, callback):
        """Register a callback that fires after each step completes."""
        self._callbacks.append(callback)
        return self
    
    def start_trace(self, query: str) -> RAGTrace:
        self.step_counter = 0
        self.current_trace = RAGTrace(query=query)
        return self.current_trace
    
    def begin_step(self, step_type: StepType, description: str) -> TraceStep:
        self.step_counter += 1
        step = TraceStep(
            step_type=step_type,
            step_number=self.step_counter,
            description=description
        )
        return step
    
    def end_step(
        self, 
        step: TraceStep, 
        data: dict = None,
        success: bool = True,
        error: str = None
    ) -> TraceStep:
        step.complete(data=data, success=success, error=error)
        
        if self.current_trace:
            self.current_trace.steps.append(step)
        
        # Fire callbacks (the UI listens here)
        for callback in self._callbacks:
            try:
                callback(step)
            except Exception:
                pass
        
        return step
    
    def finish_trace(self, final_answer: dict = None) -> RAGTrace:
        if self.current_trace:
            self.current_trace.end_time = time.time()
            self.current_trace.final_answer = final_answer
        return self.current_trace
    
    def get_summary(self) -> dict:
        """Get a summary dict for display."""
        if not self.current_trace:
            return {}
        
        trace = self.current_trace
        return {
            "query": trace.query,
            "total_duration_ms": trace.total_duration_ms,
            "steps": len(trace.steps),
            "successful_steps": sum(1 for s in trace.steps if s.success),
            "step_breakdown": {
                s.step_type.value: {
                    "duration_ms": s.duration_ms,
                    "success": s.success,
                    "data": s.data
                }
                for s in trace.steps
            }
        }