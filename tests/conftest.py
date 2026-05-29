# Shared fixtures (settings, sample docs, mock LLM)
"""
CONCEPT: conftest.py provides shared fixtures to avoid duplication.

Key fixtures:
  - mock_llm: a deterministic fake LLM (no Groq API calls in unit tests)
  - sample_docs: a tiny in-memory document set
  - settings_override: uses temp directories so tests don't touch real ChromaDB
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from langchain_core.documents import Document
from src.config import Settings
from dotenv import load_dotenv
load_dotenv()

@pytest.fixture
def sample_cpp_docs() -> list[Document]:
    """Small in-memory corpus — no repo clone needed in unit tests."""
    return [
        Document(
            page_content="""// dijkstra.cpp
// Dijkstra's shortest path algorithm using a priority queue.
// Time: O((V+E) log V)  Space: O(V)
#include <vector>
#include <queue>
using namespace std;

int dijkstra(int src, vector<vector<pair<int,int>>>& adj, int n) {
    vector<int> dist(n, INT_MAX);
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    dist[src] = 0;
    pq.push({0, src});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto [w, v] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist[n-1];
}""",
            metadata={"source_file": "graph/dijkstra.cpp", "category": "graph", "lines": 22},
        ),
        Document(
            page_content="""// merge_sort.cpp
// Merge sort — stable O(n log n) divide and conquer.
#include <vector>
using namespace std;

void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> left(arr.begin()+l, arr.begin()+m+1);
    vector<int> right(arr.begin()+m+1, arr.begin()+r+1);
    int i=0, j=0, k=l;
    while (i<left.size() && j<right.size())
        arr[k++] = (left[i] <= right[j]) ? left[i++] : right[j++];
    while (i<left.size()) arr[k++] = left[i++];
    while (j<right.size()) arr[k++] = right[j++];
}

void merge_sort(vector<int>& arr, int l, int r) {
    if (l >= r) return;
    int m = l + (r-l)/2;
    merge_sort(arr, l, m);
    merge_sort(arr, m+1, r);
    merge(arr, l, m, r);
}""",
            metadata={"source_file": "sorting/merge_sort.cpp", "category": "sorting", "lines": 20},
        ),
    ]


@pytest.fixture
def mock_llm():
    """
    Deterministic fake LLM that returns a pre-set JSON response.
    Use this to unit-test generation logic without hitting Groq.
    """
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content='''{
        "answer": "Dijkstra uses a min-heap priority queue.",
        "source_files": ["graph/dijkstra.cpp"],
        "confidence_note": "High confidence — direct match found.",
        "cpp_snippet": "priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;"
    }''')
    return llm


@pytest.fixture
def temp_settings(tmp_path: Path) -> Settings:
    """Settings pointing at a temporary directory — safe for indexer tests."""
    # Fix: Settings has no 'repo_path' field; correct field is 'cpp_repo_path'
    return Settings(
        groq_api_key="test-groq-key",
        cohere_api_key="test-cohere-key",
        chroma_db_path=str(tmp_path / "chroma"),
        cpp_repo_path=str(tmp_path / "repo"),
    )