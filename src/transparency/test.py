# Quick test — run this to see the visualiser in action
if __name__ == "__main__":
    from src.transparency.tracer import PipelineTracer, StepType
    from src.transparency.visualiser import *
    import time, random
    
    tracer = PipelineTracer()
    trace = tracer.start_trace("find an efficient graph traversal algorithm")
    
    render_query_banner(trace.query)
    
    # Simulate pipeline steps
    steps = [
        (StepType.QUERY_TRANSFORM, "Query Transformation (HyDE)", 
         {"original": trace.query, "hyde": "void bfs(Graph g, int src) { queue<int> q..."}),
        (StepType.DENSE_RETRIEVAL, "Dense Retrieval (Cosine)", 
         {"count": 16, "top_results": [
             {"file": "graph/bfs.cpp", "score": 0.94, "function": "bfs"},
             {"file": "graph/dfs.cpp", "score": 0.88, "function": "dfs"},
         ]}),
        (StepType.BM25_RETRIEVAL, "Sparse Retrieval (BM25)",
         {"count": 14, "top_results": [
             {"file": "graph/dijkstra.cpp", "score": 0.85, "function": "dijkstra"},
         ]}),
        (StepType.RRF_FUSION, "RRF Fusion", {"total_candidates": 28, "overlap": 4}),
        (StepType.RERANKING, "Cross-Encoder Reranking (Cohere)",
         {"results": [
             {"rank": 1, "file": "graph/bfs.cpp", "score": 0.96},
             {"rank": 2, "file": "graph/dijkstra.cpp", "score": 0.87},
             {"rank": 3, "file": "graph/dfs.cpp", "score": 0.81},
         ], "top_score": 0.96}),
        (StepType.CONFIDENCE_CHECK, "Confidence Assessment",
         {"confidence": 0.89, "level": "HIGH"}),
        (StepType.GENERATION, "Generation (Groq Llama 3.3 70B)",
         {"model": "llama-3.1-8b-instant", "tokens": 524, "chunks_used": 5}),
    ]
    
    for step_type, desc, data in steps:
        step = tracer.begin_step(step_type, desc)
        time.sleep(random.uniform(0.05, 0.3))  # simulate work
        step = tracer.end_step(step, data=data)
        render_step(step)
    
    tracer.finish_trace()
    render_pipeline_summary(trace)