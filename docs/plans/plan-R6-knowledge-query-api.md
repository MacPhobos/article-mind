# Plan R6: Knowledge Query API (Semantic Search)

**Plan ID:** R6-knowledge-query-api
**Created:** 2026-01-19
**Dependencies:** R4 (Chunking), R5 (Embedding/Indexing with ChromaDB)
**Estimated Time:** 3-5 days

---

## 1. Overview

### Purpose

Implement a hybrid semantic search API that enables natural language queries against session knowledge. The system combines dense vector search (ChromaDB) with sparse BM25 keyword search, merging results using Reciprocal Rank Fusion (RRF) to deliver highly relevant search results with source attribution.

### Scope

- Implement dense search using ChromaDB similarity search
- Implement sparse BM25 search using `rank-bm25` library
- Combine results using Reciprocal Rank Fusion (RRF)
- Create API endpoint for searching session knowledge
- Return ranked chunks with source attribution and confidence scores
- Optional: Cross-encoder reranking for quality boost

### Research Foundation

Based on findings from:
- `docs/research/embedding-models-research-2025-2026.md`
- `docs/research/llm-orchestration-rag-best-practices-2025-2026.md`

Key research findings applied:
- **Hybrid search**: Dense + BM25 sparse for comprehensive recall
- **RRF fusion**: Fair representation from all search types, ignores scores, focuses on rank
- **Top-K retrieval**: 10-20 chunks optimal for synthesis
- **BM25 strength**: Exact matches for codes, technical terms, keywords
- **Dense strength**: Semantic similarity, paraphrased queries, conceptual matching

### Dependencies

- **R4 (Chunking):** Requires chunked documents with chunk IDs
- **R5 (Embedding/Indexing):** Requires ChromaDB with indexed embeddings

### Outputs

- Working `/api/v1/sessions/{id}/search` endpoint
- Hybrid search module combining dense and sparse retrieval
- BM25 index management for session articles
- Pydantic schemas for search requests/responses
- Comprehensive test suite for search quality

---

## 2. Hybrid Search Architecture

```
                                    ┌─────────────────────────┐
                                    │      User Query         │
                                    └───────────┬─────────────┘
                                                │
                                    ┌───────────▼─────────────┐
                                    │    Query Processing     │
                                    │  (tokenize, normalize)  │
                                    └───────────┬─────────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
        ┌───────────▼───────────┐   ┌───────────▼───────────┐   ┌───────────▼───────────┐
        │    Dense Search       │   │    Sparse Search      │   │   Optional Rerank     │
        │    (ChromaDB)         │   │    (BM25)             │   │   (Cross-Encoder)     │
        │                       │   │                       │   │                       │
        │  - Semantic matching  │   │  - Keyword matching   │   │  - Quality boost      │
        │  - Embedding lookup   │   │  - Exact phrases      │   │  - 20-35% accuracy+   │
        │  - Top-K by cosine    │   │  - Technical terms    │   │  - 200-500ms latency  │
        └───────────┬───────────┘   └───────────┬───────────┘   └───────────┬───────────┘
                    │                           │                           │
                    └───────────────┬───────────┘                           │
                                    │                                       │
                        ┌───────────▼───────────┐                           │
                        │  Reciprocal Rank      │                           │
                        │  Fusion (RRF)         │                           │
                        │                       │                           │
                        │  score = Σ 1/(k + r)  │                           │
                        └───────────┬───────────┘                           │
                                    │                                       │
                                    └───────────────┬───────────────────────┘
                                                    │
                                        ┌───────────▼───────────┐
                                        │   Final Ranking       │
                                        │   + Source Attribution│
                                        └───────────┬───────────┘
                                                    │
                                        ┌───────────▼───────────┐
                                        │   SearchResponse      │
                                        │   (Top-K Results)     │
                                        └───────────────────────┘
```

### Search Flow Details

1. **Query Processing**: Tokenize and normalize input query
2. **Parallel Retrieval**:
   - Dense: Query ChromaDB with embedded query vector
   - Sparse: Query BM25 index with tokenized terms
3. **RRF Fusion**: Combine rankings from both methods
4. **Optional Reranking**: Cross-encoder scoring for top candidates
5. **Result Assembly**: Attach source metadata and return

---

## 3. Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Dense Search** | `chromadb` | Vector similarity search |
| **Sparse Search** | `rank-bm25` | BM25 keyword search |
| **RRF Calculation** | `numpy` | Rank fusion scoring |
| **Optional Reranker** | `sentence-transformers` | Cross-encoder reranking |
| **API Framework** | FastAPI | Endpoint handling |
| **Validation** | Pydantic 2.x | Request/response schemas |
| **Testing** | pytest + pytest-asyncio | Test coverage |

### Package Versions

```toml
# pyproject.toml additions
[project.dependencies]
chromadb = ">=0.4.0"
rank-bm25 = ">=0.2.2"
numpy = ">=1.26.0"

[project.optional-dependencies]
reranking = [
    "sentence-transformers>=2.2.0",
]
```

---

## 4. Module Structure

```
article-mind-service/
└── src/article_mind_service/
    ├── search/                          # NEW: Search module
    │   ├── __init__.py                  # Module exports
    │   ├── dense_search.py              # ChromaDB similarity search
    │   ├── sparse_search.py             # BM25 keyword search
    │   ├── hybrid_search.py             # Fusion logic + orchestration
    │   ├── reranker.py                  # Optional cross-encoder
    │   └── bm25_index.py                # BM25 index management
    ├── schemas/
    │   ├── __init__.py                  # UPDATED: Export search schemas
    │   └── search.py                    # NEW: Search request/response
    ├── routers/
    │   ├── __init__.py                  # UPDATED: Export search router
    │   └── search.py                    # NEW: Search endpoints
    └── config.py                        # UPDATED: Search configuration
```

---

## 5. BM25 Index Management

### BM25 Overview

BM25 (Best Matching 25) is a probabilistic ranking function for keyword-based retrieval. It excels at:
- Exact keyword matches
- Technical terms and codes
- Entity names and identifiers
- Phrases with specific terminology

### Index Lifecycle

```python
"""BM25 index management for session articles."""

from dataclasses import dataclass, field
from rank_bm25 import BM25Okapi
import re
from typing import TypeAlias

TokenizedDoc: TypeAlias = list[str]


@dataclass
class BM25Index:
    """BM25 index for a session's articles.

    Maintains tokenized documents and BM25 scoring model.
    Rebuilt when articles are added/removed from session.
    """

    session_id: int
    chunk_ids: list[str] = field(default_factory=list)
    tokenized_docs: list[TokenizedDoc] = field(default_factory=list)
    _bm25: BM25Okapi | None = field(default=None, repr=False)

    def add_document(self, chunk_id: str, content: str) -> None:
        """Add a document chunk to the index."""
        tokens = self._tokenize(content)
        self.chunk_ids.append(chunk_id)
        self.tokenized_docs.append(tokens)
        self._invalidate_index()

    def remove_document(self, chunk_id: str) -> bool:
        """Remove a document chunk from the index."""
        try:
            idx = self.chunk_ids.index(chunk_id)
            self.chunk_ids.pop(idx)
            self.tokenized_docs.pop(idx)
            self._invalidate_index()
            return True
        except ValueError:
            return False

    def build(self) -> None:
        """Build or rebuild the BM25 index."""
        if self.tokenized_docs:
            self._bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search the index and return ranked chunk IDs with scores.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (chunk_id, bm25_score) tuples, sorted by score descending
        """
        if self._bm25 is None:
            self.build()

        if self._bm25 is None or not self.chunk_ids:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices sorted by score
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = [
            (self.chunk_ids[i], float(score))
            for i, score in indexed_scores[:top_k]
            if score > 0  # Only include non-zero scores
        ]

        return results

    def _tokenize(self, text: str) -> TokenizedDoc:
        """Tokenize text for BM25 indexing.

        Simple tokenization: lowercase, split on non-alphanumeric,
        filter short tokens.
        """
        text = text.lower()
        tokens = re.split(r'[^a-z0-9]+', text)
        return [t for t in tokens if len(t) > 1]

    def _invalidate_index(self) -> None:
        """Invalidate cached BM25 index (lazy rebuild on next search)."""
        self._bm25 = None
```

### Index Storage Strategy

**In-Memory with Lazy Rebuild:**
- BM25 indexes are lightweight (tokenized docs only)
- Store in memory per session
- Rebuild on article add/remove
- Optional: Cache to disk for persistence across restarts

```python
"""BM25 index cache manager."""

from typing import ClassVar


class BM25IndexCache:
    """Session-scoped BM25 index cache.

    Maintains one BM25 index per active session.
    Indexes are built lazily on first search.
    """

    _indexes: ClassVar[dict[int, BM25Index]] = {}

    @classmethod
    def get_or_create(cls, session_id: int) -> BM25Index:
        """Get existing index or create empty one for session."""
        if session_id not in cls._indexes:
            cls._indexes[session_id] = BM25Index(session_id=session_id)
        return cls._indexes[session_id]

    @classmethod
    def invalidate(cls, session_id: int) -> None:
        """Remove cached index for session (force rebuild)."""
        cls._indexes.pop(session_id, None)

    @classmethod
    def populate_from_chunks(
        cls,
        session_id: int,
        chunks: list[tuple[str, str]]  # [(chunk_id, content), ...]
    ) -> BM25Index:
        """Build index from list of chunks."""
        index = BM25Index(session_id=session_id)
        for chunk_id, content in chunks:
            index.add_document(chunk_id, content)
        index.build()
        cls._indexes[session_id] = index
        return index
```

---

## 6. Reciprocal Rank Fusion (RRF)

### Algorithm Overview

RRF combines rankings from multiple retrieval methods without requiring score normalization. It focuses purely on rank position, making it robust across different scoring scales.

**Formula:**
```
RRF_score(doc) = Σ 1 / (k + rank_i(doc))
```

Where:
- `k` is a constant (typically 60) that dampens the effect of high rankings
- `rank_i(doc)` is the rank of the document in retrieval method `i`

### Implementation

```python
"""Reciprocal Rank Fusion for combining search results."""

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class RankedResult:
    """A search result with its source rankings."""
    chunk_id: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],  # [(chunk_id, score), ...]
    sparse_results: list[tuple[str, float]],
    k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[RankedResult]:
    """Combine dense and sparse search results using RRF.

    Args:
        dense_results: Results from dense vector search (chunk_id, score)
        sparse_results: Results from BM25 sparse search (chunk_id, score)
        k: RRF constant (higher = less emphasis on top ranks)
        dense_weight: Weight for dense search contribution
        sparse_weight: Weight for sparse search contribution

    Returns:
        List of RankedResult objects sorted by RRF score descending
    """
    # Build lookup of chunk_id -> RankedResult
    results: dict[str, RankedResult] = {}

    # Process dense results (rank is 1-indexed)
    for rank, (chunk_id, _score) in enumerate(dense_results, start=1):
        if chunk_id not in results:
            results[chunk_id] = RankedResult(chunk_id=chunk_id)
        results[chunk_id].dense_rank = rank
        results[chunk_id].rrf_score += dense_weight * (1.0 / (k + rank))

    # Process sparse results
    for rank, (chunk_id, _score) in enumerate(sparse_results, start=1):
        if chunk_id not in results:
            results[chunk_id] = RankedResult(chunk_id=chunk_id)
        results[chunk_id].sparse_rank = rank
        results[chunk_id].rrf_score += sparse_weight * (1.0 / (k + rank))

    # Sort by RRF score descending
    ranked = sorted(results.values(), key=lambda r: r.rrf_score, reverse=True)

    return ranked


def rrf_score(ranks: list[int], k: int = 60) -> float:
    """Calculate RRF score for a single document across multiple rankings.

    Args:
        ranks: List of ranks from different retrieval methods (1-indexed)
        k: RRF constant (default 60)

    Returns:
        Combined RRF score
    """
    return sum(1.0 / (k + r) for r in ranks)
```

### RRF Parameter Tuning

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `k` | 60 | 1-100 | Higher k = more weight to lower-ranked results |
| `dense_weight` | 0.7 | 0.5-0.9 | Weight for semantic search |
| `sparse_weight` | 0.3 | 0.1-0.5 | Weight for keyword search |

**Research recommendation:** Start with k=60, dense_weight=0.7, sparse_weight=0.3, then tune based on evaluation results.

---

## 7. API Endpoint

### Endpoint Specification

```
POST /api/v1/sessions/{session_id}/search
```

**Request Headers:**
- `Content-Type: application/json`
- `Authorization: Bearer <token>` (when auth is implemented)

**Path Parameters:**
- `session_id` (int): Session ID to search within

**Request Body:**
```json
{
  "query": "How does authentication work in the system?",
  "top_k": 10,
  "include_content": true,
  "search_mode": "hybrid"
}
```

**Response Body (200 OK):**
```json
{
  "query": "How does authentication work in the system?",
  "results": [
    {
      "chunk_id": "doc_abc123:chunk_5",
      "article_id": 42,
      "content": "Authentication uses JWT tokens with refresh...",
      "score": 0.0156,
      "source_url": "https://docs.example.com/auth",
      "source_title": "Authentication Guide",
      "dense_rank": 2,
      "sparse_rank": 1
    }
  ],
  "total_chunks_searched": 1547,
  "search_mode": "hybrid",
  "timing_ms": 127
}
```

---

## 8. Pydantic Schemas

```python
"""Search request and response schemas."""

from pydantic import BaseModel, Field
from enum import Enum


class SearchMode(str, Enum):
    """Search mode selection."""
    DENSE = "dense"      # Vector search only
    SPARSE = "sparse"    # BM25 search only
    HYBRID = "hybrid"    # Combined (default)


class SearchRequest(BaseModel):
    """Search request parameters.

    See: docs/research/embedding-models-research-2025-2026.md
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language search query",
        examples=["How does authentication work?"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return (1-50)",
    )
    include_content: bool = Field(
        default=True,
        description="Include chunk content in results",
    )
    search_mode: SearchMode = Field(
        default=SearchMode.HYBRID,
        description="Search mode: dense, sparse, or hybrid",
    )


class SearchResult(BaseModel):
    """Individual search result with source attribution.

    Includes provenance information for citation support.
    """

    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier (doc_id:chunk_index)",
        examples=["doc_abc123:chunk_5"],
    )
    article_id: int = Field(
        ...,
        description="Source article ID",
    )
    content: str | None = Field(
        default=None,
        description="Chunk text content (if include_content=true)",
    )
    score: float = Field(
        ...,
        ge=0.0,
        description="Combined relevance score (RRF)",
    )
    source_url: str | None = Field(
        default=None,
        description="Original article URL",
    )
    source_title: str | None = Field(
        default=None,
        description="Article title",
    )
    dense_rank: int | None = Field(
        default=None,
        description="Rank from dense search (if applicable)",
    )
    sparse_rank: int | None = Field(
        default=None,
        description="Rank from sparse search (if applicable)",
    )


class SearchResponse(BaseModel):
    """Search response with results and metadata.

    Follows API contract specification.
    """

    query: str = Field(
        ...,
        description="Original search query",
    )
    results: list[SearchResult] = Field(
        default_factory=list,
        description="Ranked search results",
    )
    total_chunks_searched: int = Field(
        ...,
        ge=0,
        description="Total chunks in session index",
    )
    search_mode: SearchMode = Field(
        ...,
        description="Search mode used",
    )
    timing_ms: int = Field(
        ...,
        ge=0,
        description="Search execution time in milliseconds",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "How does authentication work?",
                    "results": [
                        {
                            "chunk_id": "doc_abc123:chunk_5",
                            "article_id": 42,
                            "content": "Authentication uses JWT tokens...",
                            "score": 0.0156,
                            "source_url": "https://docs.example.com/auth",
                            "source_title": "Authentication Guide",
                            "dense_rank": 2,
                            "sparse_rank": 1,
                        }
                    ],
                    "total_chunks_searched": 1547,
                    "search_mode": "hybrid",
                    "timing_ms": 127,
                }
            ]
        }
    }
```

---

## 9. Configuration

### Environment Variables

```bash
# .env additions for search configuration

# Search defaults
SEARCH_TOP_K=10
SEARCH_DENSE_WEIGHT=0.7
SEARCH_SPARSE_WEIGHT=0.3

# RRF tuning
SEARCH_RRF_K=60

# Reranking (optional)
SEARCH_RERANK_ENABLED=false
SEARCH_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
SEARCH_RERANK_TOP_K=20

# Performance
SEARCH_MAX_QUERY_LENGTH=1000
SEARCH_TIMEOUT_SECONDS=30
```

### Configuration Schema

```python
"""Search configuration extension for settings."""

from pydantic import Field
from pydantic_settings import BaseSettings


class SearchSettings(BaseSettings):
    """Search-specific configuration."""

    # Search defaults
    search_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Default number of results",
    )
    search_dense_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for dense search in RRF",
    )
    search_sparse_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for sparse search in RRF",
    )

    # RRF tuning
    search_rrf_k: int = Field(
        default=60,
        ge=1,
        le=100,
        description="RRF k constant",
    )

    # Reranking
    search_rerank_enabled: bool = Field(
        default=False,
        description="Enable cross-encoder reranking",
    )
    search_rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Reranking model name",
    )
    search_rerank_top_k: int = Field(
        default=20,
        description="Number of candidates for reranking",
    )

    # Performance
    search_max_query_length: int = Field(
        default=1000,
        description="Maximum query string length",
    )
    search_timeout_seconds: int = Field(
        default=30,
        description="Search timeout in seconds",
    )

    model_config = {
        "env_prefix": "",
        "case_sensitive": False,
    }
```

---

## 10. Implementation Steps

### Step 1: Create Search Module Structure (30 min)

Create directory and `__init__.py` files:

```bash
mkdir -p src/article_mind_service/search
touch src/article_mind_service/search/__init__.py
```

```python
# src/article_mind_service/search/__init__.py
"""Hybrid search module for knowledge retrieval."""

from .dense_search import DenseSearch
from .sparse_search import SparseSearch, BM25Index, BM25IndexCache
from .hybrid_search import HybridSearch
from .reranker import Reranker

__all__ = [
    "DenseSearch",
    "SparseSearch",
    "BM25Index",
    "BM25IndexCache",
    "HybridSearch",
    "Reranker",
]
```

### Step 2: Implement Dense Search (1-2 hours)

Create `src/article_mind_service/search/dense_search.py`:

```python
"""Dense vector search using ChromaDB."""

from dataclasses import dataclass
import chromadb
from chromadb.api.types import QueryResult

from article_mind_service.config import settings


@dataclass
class DenseSearchResult:
    """Result from dense vector search."""
    chunk_id: str
    score: float  # Similarity score (higher = more similar)
    metadata: dict


class DenseSearch:
    """ChromaDB-based dense vector search.

    Performs semantic similarity search using embeddings stored
    in ChromaDB during the indexing phase (R5).
    """

    def __init__(self, collection_name: str | None = None):
        """Initialize dense search with ChromaDB connection.

        Args:
            collection_name: ChromaDB collection name (default from settings)
        """
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_directory)
        )
        self.collection_name = collection_name or settings.chroma_collection_name

    def search(
        self,
        session_id: int,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[DenseSearchResult]:
        """Search for similar chunks using query embedding.

        Args:
            session_id: Session ID to filter results
            query_embedding: Query vector from embedding model
            top_k: Number of results to return

        Returns:
            List of DenseSearchResult sorted by similarity descending
        """
        collection = self.client.get_collection(self.collection_name)

        # Query with session filter
        results: QueryResult = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"session_id": session_id},
            include=["metadatas", "distances"],
        )

        # Convert to DenseSearchResult
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances; convert to similarity
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1.0 - distance

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                search_results.append(DenseSearchResult(
                    chunk_id=chunk_id,
                    score=similarity,
                    metadata=metadata,
                ))

        return search_results

    def get_total_chunks(self, session_id: int) -> int:
        """Get total number of chunks indexed for a session."""
        collection = self.client.get_collection(self.collection_name)

        # Count chunks with session filter
        results = collection.get(
            where={"session_id": session_id},
            include=[],  # Don't need actual data
        )

        return len(results["ids"]) if results["ids"] else 0
```

### Step 3: Implement Sparse Search / BM25 (1-2 hours)

Create `src/article_mind_service/search/sparse_search.py`:

```python
"""BM25 sparse keyword search implementation."""

from dataclasses import dataclass, field
import re
from typing import ClassVar

from rank_bm25 import BM25Okapi


TokenizedDoc = list[str]


@dataclass
class BM25Index:
    """BM25 index for a session's article chunks.

    Maintains tokenized documents and BM25 scoring model.
    Rebuilt when articles are added/removed from session.
    """

    session_id: int
    chunk_ids: list[str] = field(default_factory=list)
    chunk_contents: list[str] = field(default_factory=list)
    tokenized_docs: list[TokenizedDoc] = field(default_factory=list)
    _bm25: BM25Okapi | None = field(default=None, repr=False)

    def add_document(self, chunk_id: str, content: str) -> None:
        """Add a document chunk to the index."""
        tokens = self._tokenize(content)
        self.chunk_ids.append(chunk_id)
        self.chunk_contents.append(content)
        self.tokenized_docs.append(tokens)
        self._invalidate_index()

    def remove_document(self, chunk_id: str) -> bool:
        """Remove a document chunk from the index."""
        try:
            idx = self.chunk_ids.index(chunk_id)
            self.chunk_ids.pop(idx)
            self.chunk_contents.pop(idx)
            self.tokenized_docs.pop(idx)
            self._invalidate_index()
            return True
        except ValueError:
            return False

    def build(self) -> None:
        """Build or rebuild the BM25 index."""
        if self.tokenized_docs:
            self._bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search the index and return ranked chunk IDs with scores.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (chunk_id, bm25_score) tuples, sorted by score descending
        """
        if self._bm25 is None:
            self.build()

        if self._bm25 is None or not self.chunk_ids:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices sorted by score
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = [
            (self.chunk_ids[i], float(score))
            for i, score in indexed_scores[:top_k]
            if score > 0  # Only include non-zero scores
        ]

        return results

    def get_content(self, chunk_id: str) -> str | None:
        """Get content for a chunk by ID."""
        try:
            idx = self.chunk_ids.index(chunk_id)
            return self.chunk_contents[idx]
        except ValueError:
            return None

    def _tokenize(self, text: str) -> TokenizedDoc:
        """Tokenize text for BM25 indexing.

        Simple tokenization: lowercase, split on non-alphanumeric,
        filter short tokens.
        """
        text = text.lower()
        tokens = re.split(r'[^a-z0-9]+', text)
        return [t for t in tokens if len(t) > 1]

    def _invalidate_index(self) -> None:
        """Invalidate cached BM25 index (lazy rebuild on next search)."""
        self._bm25 = None

    def __len__(self) -> int:
        """Return number of indexed chunks."""
        return len(self.chunk_ids)


class BM25IndexCache:
    """Session-scoped BM25 index cache.

    Maintains one BM25 index per active session.
    Indexes are built lazily on first search.
    """

    _indexes: ClassVar[dict[int, BM25Index]] = {}

    @classmethod
    def get(cls, session_id: int) -> BM25Index | None:
        """Get existing index for session."""
        return cls._indexes.get(session_id)

    @classmethod
    def get_or_create(cls, session_id: int) -> BM25Index:
        """Get existing index or create empty one for session."""
        if session_id not in cls._indexes:
            cls._indexes[session_id] = BM25Index(session_id=session_id)
        return cls._indexes[session_id]

    @classmethod
    def set(cls, session_id: int, index: BM25Index) -> None:
        """Store index for session."""
        cls._indexes[session_id] = index

    @classmethod
    def invalidate(cls, session_id: int) -> None:
        """Remove cached index for session (force rebuild)."""
        cls._indexes.pop(session_id, None)

    @classmethod
    def populate_from_chunks(
        cls,
        session_id: int,
        chunks: list[tuple[str, str]],  # [(chunk_id, content), ...]
    ) -> BM25Index:
        """Build index from list of chunks."""
        index = BM25Index(session_id=session_id)
        for chunk_id, content in chunks:
            index.add_document(chunk_id, content)
        index.build()
        cls._indexes[session_id] = index
        return index


class SparseSearch:
    """BM25-based sparse keyword search.

    Manages BM25 indexes per session and provides search interface.
    """

    def __init__(self):
        """Initialize sparse search."""
        self.cache = BM25IndexCache

    def search(
        self,
        session_id: int,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search session's BM25 index.

        Args:
            session_id: Session ID to search
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of (chunk_id, score) tuples sorted by score descending
        """
        index = self.cache.get(session_id)
        if index is None:
            return []

        return index.search(query, top_k)

    def get_index(self, session_id: int) -> BM25Index | None:
        """Get BM25 index for session."""
        return self.cache.get(session_id)

    def ensure_index(
        self,
        session_id: int,
        chunks: list[tuple[str, str]],
    ) -> BM25Index:
        """Ensure BM25 index exists for session, building if necessary.

        Args:
            session_id: Session ID
            chunks: List of (chunk_id, content) tuples

        Returns:
            Built BM25Index
        """
        return self.cache.populate_from_chunks(session_id, chunks)
```

### Step 4: Implement Hybrid Search (1-2 hours)

Create `src/article_mind_service/search/hybrid_search.py`:

```python
"""Hybrid search combining dense and sparse retrieval with RRF."""

import time
from dataclasses import dataclass

from article_mind_service.config import settings
from article_mind_service.schemas.search import (
    SearchMode,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from .dense_search import DenseSearch
from .sparse_search import SparseSearch, BM25IndexCache
from .reranker import Reranker


@dataclass
class RankedResult:
    """Intermediate result with ranking metadata."""
    chunk_id: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0
    content: str | None = None
    metadata: dict | None = None


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[RankedResult]:
    """Combine dense and sparse results using RRF.

    Args:
        dense_results: Results from dense search (chunk_id, score)
        sparse_results: Results from sparse search (chunk_id, score)
        k: RRF constant
        dense_weight: Weight for dense contribution
        sparse_weight: Weight for sparse contribution

    Returns:
        List of RankedResult sorted by RRF score descending
    """
    results: dict[str, RankedResult] = {}

    # Process dense results
    for rank, (chunk_id, _score) in enumerate(dense_results, start=1):
        if chunk_id not in results:
            results[chunk_id] = RankedResult(chunk_id=chunk_id)
        results[chunk_id].dense_rank = rank
        results[chunk_id].rrf_score += dense_weight * (1.0 / (k + rank))

    # Process sparse results
    for rank, (chunk_id, _score) in enumerate(sparse_results, start=1):
        if chunk_id not in results:
            results[chunk_id] = RankedResult(chunk_id=chunk_id)
        results[chunk_id].sparse_rank = rank
        results[chunk_id].rrf_score += sparse_weight * (1.0 / (k + rank))

    # Sort by RRF score
    return sorted(results.values(), key=lambda r: r.rrf_score, reverse=True)


class HybridSearch:
    """Hybrid search orchestrator.

    Combines dense vector search and sparse BM25 search using
    Reciprocal Rank Fusion for optimal retrieval performance.
    """

    def __init__(
        self,
        dense_search: DenseSearch | None = None,
        sparse_search: SparseSearch | None = None,
        reranker: Reranker | None = None,
    ):
        """Initialize hybrid search components.

        Args:
            dense_search: Dense search instance (created if None)
            sparse_search: Sparse search instance (created if None)
            reranker: Optional reranker instance
        """
        self.dense = dense_search or DenseSearch()
        self.sparse = sparse_search or SparseSearch()
        self.reranker = reranker

        # Configuration from settings
        self.dense_weight = settings.search_dense_weight
        self.sparse_weight = settings.search_sparse_weight
        self.rrf_k = settings.search_rrf_k
        self.rerank_enabled = settings.search_rerank_enabled

    async def search(
        self,
        session_id: int,
        request: SearchRequest,
        query_embedding: list[float],
    ) -> SearchResponse:
        """Execute hybrid search.

        Args:
            session_id: Session to search
            request: Search request parameters
            query_embedding: Pre-computed query embedding

        Returns:
            SearchResponse with ranked results
        """
        start_time = time.time()

        # Determine effective top_k (get more for reranking)
        retrieve_k = request.top_k
        if self.rerank_enabled and self.reranker:
            retrieve_k = max(request.top_k * 2, settings.search_rerank_top_k)

        # Execute searches based on mode
        dense_results: list[tuple[str, float]] = []
        sparse_results: list[tuple[str, float]] = []

        if request.search_mode in (SearchMode.DENSE, SearchMode.HYBRID):
            dense_results = [
                (r.chunk_id, r.score)
                for r in self.dense.search(
                    session_id=session_id,
                    query_embedding=query_embedding,
                    top_k=retrieve_k,
                )
            ]

        if request.search_mode in (SearchMode.SPARSE, SearchMode.HYBRID):
            sparse_results = self.sparse.search(
                session_id=session_id,
                query=request.query,
                top_k=retrieve_k,
            )

        # Combine results
        if request.search_mode == SearchMode.HYBRID:
            ranked = reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                k=self.rrf_k,
                dense_weight=self.dense_weight,
                sparse_weight=self.sparse_weight,
            )
        elif request.search_mode == SearchMode.DENSE:
            ranked = [
                RankedResult(
                    chunk_id=cid,
                    dense_rank=i + 1,
                    rrf_score=score,
                )
                for i, (cid, score) in enumerate(dense_results)
            ]
        else:  # SPARSE
            ranked = [
                RankedResult(
                    chunk_id=cid,
                    sparse_rank=i + 1,
                    rrf_score=score,
                )
                for i, (cid, score) in enumerate(sparse_results)
            ]

        # Optional reranking
        if self.rerank_enabled and self.reranker and ranked:
            ranked = await self._rerank(request.query, ranked[:retrieve_k])

        # Limit to requested top_k
        ranked = ranked[:request.top_k]

        # Build response with content and metadata
        results = await self._build_results(
            session_id=session_id,
            ranked=ranked,
            include_content=request.include_content,
        )

        # Get total chunks count
        total_chunks = self.dense.get_total_chunks(session_id)

        timing_ms = int((time.time() - start_time) * 1000)

        return SearchResponse(
            query=request.query,
            results=results,
            total_chunks_searched=total_chunks,
            search_mode=request.search_mode,
            timing_ms=timing_ms,
        )

    async def _rerank(
        self,
        query: str,
        ranked: list[RankedResult],
    ) -> list[RankedResult]:
        """Rerank results using cross-encoder.

        Args:
            query: Original query
            ranked: Pre-ranked results

        Returns:
            Reranked results
        """
        if not self.reranker or not ranked:
            return ranked

        # Get content for reranking
        contents = [r.content for r in ranked if r.content]
        if not contents:
            return ranked

        # Rerank
        rerank_scores = await self.reranker.rerank(query, contents)

        # Update scores and re-sort
        for i, r in enumerate(ranked):
            if i < len(rerank_scores):
                r.rrf_score = rerank_scores[i]

        return sorted(ranked, key=lambda r: r.rrf_score, reverse=True)

    async def _build_results(
        self,
        session_id: int,
        ranked: list[RankedResult],
        include_content: bool,
    ) -> list[SearchResult]:
        """Build SearchResult objects with full metadata.

        Args:
            session_id: Session ID
            ranked: Ranked results
            include_content: Whether to include chunk content

        Returns:
            List of SearchResult objects
        """
        results = []

        # Get BM25 index for content lookup
        bm25_index = self.sparse.get_index(session_id)

        for r in ranked:
            # Get content from BM25 index (it stores content)
            content = None
            if include_content and bm25_index:
                content = bm25_index.get_content(r.chunk_id)

            # Parse article_id from chunk_id (format: doc_XXX:chunk_N)
            article_id = 0
            if ":" in r.chunk_id:
                # Extract article ID from chunk metadata
                # This would come from ChromaDB metadata in real implementation
                pass

            results.append(SearchResult(
                chunk_id=r.chunk_id,
                article_id=article_id,  # TODO: Get from metadata
                content=content,
                score=r.rrf_score,
                source_url=r.metadata.get("source_url") if r.metadata else None,
                source_title=r.metadata.get("source_title") if r.metadata else None,
                dense_rank=r.dense_rank,
                sparse_rank=r.sparse_rank,
            ))

        return results
```

### Step 5: Implement Optional Reranker (1 hour)

Create `src/article_mind_service/search/reranker.py`:

```python
"""Optional cross-encoder reranking for search quality boost."""

from article_mind_service.config import settings


class Reranker:
    """Cross-encoder reranker for improving search result quality.

    Uses sentence-transformers cross-encoder models to score
    query-document pairs for more accurate ranking.

    Research shows 20-35% accuracy improvement with 200-500ms latency.
    """

    def __init__(self, model_name: str | None = None):
        """Initialize reranker with model.

        Args:
            model_name: Cross-encoder model name (default from settings)
        """
        self.model_name = model_name or settings.search_rerank_model
        self._model = None

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def rerank(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts to rank

        Returns:
            List of relevance scores (same order as input documents)
        """
        if not documents:
            return []

        model = self._load_model()

        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]

        # Get scores from cross-encoder
        scores = model.predict(pairs)

        return [float(s) for s in scores]

    async def rerank_with_ids(
        self,
        query: str,
        documents: list[tuple[str, str]],  # [(id, content), ...]
    ) -> list[tuple[str, float]]:
        """Rerank documents and return with IDs.

        Args:
            query: Search query
            documents: List of (id, content) tuples

        Returns:
            List of (id, score) tuples sorted by score descending
        """
        if not documents:
            return []

        ids = [d[0] for d in documents]
        contents = [d[1] for d in documents]

        scores = await self.rerank(query, contents)

        # Combine and sort
        results = list(zip(ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        return results
```

### Step 6: Create Search Router (1-2 hours)

Create `src/article_mind_service/routers/search.py`:

```python
"""Search API endpoints."""

import time
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.config import settings
from article_mind_service.database import get_db
from article_mind_service.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchMode,
)
from article_mind_service.search import HybridSearch, SparseSearch, BM25IndexCache
# from article_mind_service.embeddings import get_query_embedding  # From R5

router = APIRouter(prefix="/api/v1", tags=["search"])


# Singleton search instance
_hybrid_search: HybridSearch | None = None


def get_hybrid_search() -> HybridSearch:
    """Get or create hybrid search instance."""
    global _hybrid_search
    if _hybrid_search is None:
        _hybrid_search = HybridSearch()
    return _hybrid_search


@router.post(
    "/sessions/{session_id}/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search session knowledge",
    description="""
Search the session's indexed articles using natural language queries.

Supports three search modes:
- **hybrid** (default): Combines semantic and keyword search for best results
- **dense**: Semantic similarity search only
- **sparse**: BM25 keyword search only

Returns ranked results with source attribution for citations.
    """,
)
async def search_session(
    session_id: int,
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    search: HybridSearch = Depends(get_hybrid_search),
) -> SearchResponse:
    """Search session knowledge using hybrid retrieval.

    Args:
        session_id: Session ID to search within
        request: Search parameters
        db: Database session
        search: Hybrid search instance

    Returns:
        SearchResponse with ranked results and metadata

    Raises:
        HTTPException: If session not found or search fails
    """
    # TODO: Verify session exists
    # session = await get_session(db, session_id)
    # if not session:
    #     raise HTTPException(
    #         status_code=status.HTTP_404_NOT_FOUND,
    #         detail=f"Session {session_id} not found",
    #     )

    # Check if BM25 index exists for session
    bm25_index = BM25IndexCache.get(session_id)
    if bm25_index is None or len(bm25_index) == 0:
        # No indexed content for this session
        return SearchResponse(
            query=request.query,
            results=[],
            total_chunks_searched=0,
            search_mode=request.search_mode,
            timing_ms=0,
        )

    try:
        # Get query embedding for dense search
        # TODO: Integrate with embedding service from R5
        query_embedding: list[float] = []
        if request.search_mode in (SearchMode.DENSE, SearchMode.HYBRID):
            # query_embedding = await get_query_embedding(request.query)
            # For now, return sparse-only results
            if request.search_mode == SearchMode.DENSE:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="Dense search requires embedding service (R5)",
                )
            # Fall back to sparse for hybrid until R5 is complete
            request.search_mode = SearchMode.SPARSE

        # Execute search
        response = await search.search(
            session_id=session_id,
            request=request,
            query_embedding=query_embedding,
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.get(
    "/sessions/{session_id}/search/stats",
    summary="Get search index statistics",
    description="Returns statistics about the session's search index.",
)
async def search_stats(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get search index statistics for a session.

    Args:
        session_id: Session ID
        db: Database session

    Returns:
        Dictionary with index statistics
    """
    bm25_index = BM25IndexCache.get(session_id)

    return {
        "session_id": session_id,
        "bm25_index_exists": bm25_index is not None,
        "total_chunks": len(bm25_index) if bm25_index else 0,
        "search_modes_available": [
            SearchMode.SPARSE.value,
            # Add DENSE and HYBRID when R5 embedding service is ready
        ],
    }
```

### Step 7: Update Module Exports and Routers (30 min)

Update `src/article_mind_service/schemas/__init__.py`:

```python
"""Pydantic schemas for API request/response models."""

from .health import HealthResponse
from .search import (
    SearchMode,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

__all__ = [
    "HealthResponse",
    "SearchMode",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
]
```

Update `src/article_mind_service/routers/__init__.py`:

```python
"""FastAPI routers."""

from .health import router as health_router
from .search import router as search_router

__all__ = [
    "health_router",
    "search_router",
]
```

Update `src/article_mind_service/main.py` to include search router:

```python
# Add to imports
from .routers import health_router, search_router

# Add after health router
app.include_router(search_router)
```

### Step 8: Write Tests (2-3 hours)

Create `tests/test_search.py`:

```python
"""Tests for hybrid search functionality."""

import pytest
from httpx import AsyncClient

from article_mind_service.search import (
    BM25Index,
    BM25IndexCache,
    SparseSearch,
)
from article_mind_service.search.hybrid_search import reciprocal_rank_fusion


# ============================================================================
# BM25 Index Tests
# ============================================================================

class TestBM25Index:
    """Tests for BM25 index functionality."""

    def test_create_empty_index(self):
        """Test creating an empty BM25 index."""
        index = BM25Index(session_id=1)
        assert len(index) == 0
        assert index.search("test") == []

    def test_add_document(self):
        """Test adding a document to the index."""
        index = BM25Index(session_id=1)
        index.add_document("chunk_1", "This is a test document about Python programming.")

        assert len(index) == 1
        assert "chunk_1" in index.chunk_ids

    def test_search_finds_relevant_document(self):
        """Test that search finds relevant documents."""
        index = BM25Index(session_id=1)
        index.add_document("chunk_1", "Python is a programming language.")
        index.add_document("chunk_2", "JavaScript is used for web development.")
        index.add_document("chunk_3", "Python can be used for machine learning.")
        index.build()

        results = index.search("Python programming", top_k=2)

        assert len(results) > 0
        # Python chunks should rank higher
        chunk_ids = [r[0] for r in results]
        assert "chunk_1" in chunk_ids or "chunk_3" in chunk_ids

    def test_search_exact_match(self):
        """Test that exact keyword matches score high."""
        index = BM25Index(session_id=1)
        index.add_document("chunk_1", "Authentication using JWT tokens.")
        index.add_document("chunk_2", "User login and session management.")
        index.add_document("chunk_3", "Database connection pooling.")
        index.build()

        results = index.search("JWT authentication", top_k=3)

        # JWT chunk should be first
        assert results[0][0] == "chunk_1"

    def test_remove_document(self):
        """Test removing a document from the index."""
        index = BM25Index(session_id=1)
        index.add_document("chunk_1", "First document")
        index.add_document("chunk_2", "Second document")

        assert len(index) == 2

        removed = index.remove_document("chunk_1")

        assert removed is True
        assert len(index) == 1
        assert "chunk_1" not in index.chunk_ids

    def test_tokenization(self):
        """Test tokenization handles various inputs."""
        index = BM25Index(session_id=1)

        # Test basic tokenization
        tokens = index._tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

        # Test with special characters
        tokens = index._tokenize("user@email.com has-dashes")
        assert "user" in tokens
        assert "email" in tokens
        assert "dashes" in tokens

        # Test filtering short tokens
        tokens = index._tokenize("a the is be")
        assert "a" not in tokens  # Single char filtered


class TestBM25IndexCache:
    """Tests for BM25 index caching."""

    def setup_method(self):
        """Clear cache before each test."""
        BM25IndexCache._indexes.clear()

    def test_get_or_create(self):
        """Test get_or_create returns same instance."""
        index1 = BM25IndexCache.get_or_create(session_id=1)
        index2 = BM25IndexCache.get_or_create(session_id=1)

        assert index1 is index2

    def test_different_sessions_get_different_indexes(self):
        """Test different sessions have separate indexes."""
        index1 = BM25IndexCache.get_or_create(session_id=1)
        index2 = BM25IndexCache.get_or_create(session_id=2)

        assert index1 is not index2
        assert index1.session_id == 1
        assert index2.session_id == 2

    def test_invalidate_removes_index(self):
        """Test invalidating a session's index."""
        BM25IndexCache.get_or_create(session_id=1)

        BM25IndexCache.invalidate(session_id=1)

        assert BM25IndexCache.get(session_id=1) is None

    def test_populate_from_chunks(self):
        """Test populating index from chunk list."""
        chunks = [
            ("chunk_1", "First document content"),
            ("chunk_2", "Second document content"),
        ]

        index = BM25IndexCache.populate_from_chunks(session_id=1, chunks=chunks)

        assert len(index) == 2
        assert index.get_content("chunk_1") == "First document content"


# ============================================================================
# RRF Tests
# ============================================================================

class TestReciprocalRankFusion:
    """Tests for RRF algorithm."""

    def test_empty_results(self):
        """Test RRF with empty inputs."""
        results = reciprocal_rank_fusion([], [])
        assert results == []

    def test_dense_only(self):
        """Test RRF with only dense results."""
        dense_results = [
            ("chunk_1", 0.9),
            ("chunk_2", 0.8),
        ]

        results = reciprocal_rank_fusion(dense_results, [])

        assert len(results) == 2
        assert results[0].chunk_id == "chunk_1"
        assert results[0].dense_rank == 1
        assert results[0].sparse_rank is None

    def test_sparse_only(self):
        """Test RRF with only sparse results."""
        sparse_results = [
            ("chunk_1", 5.5),
            ("chunk_2", 4.2),
        ]

        results = reciprocal_rank_fusion([], sparse_results)

        assert len(results) == 2
        assert results[0].chunk_id == "chunk_1"
        assert results[0].sparse_rank == 1
        assert results[0].dense_rank is None

    def test_fusion_combines_results(self):
        """Test RRF properly fuses dense and sparse results."""
        dense_results = [
            ("chunk_1", 0.9),
            ("chunk_2", 0.8),
            ("chunk_3", 0.7),
        ]
        sparse_results = [
            ("chunk_2", 5.5),  # chunk_2 ranks higher in sparse
            ("chunk_1", 4.2),
            ("chunk_4", 3.0),  # unique to sparse
        ]

        results = reciprocal_rank_fusion(dense_results, sparse_results)

        # Should have 4 unique chunks
        assert len(results) == 4

        # chunk_1 and chunk_2 appear in both, should rank high
        top_ids = [r.chunk_id for r in results[:2]]
        assert "chunk_1" in top_ids or "chunk_2" in top_ids

    def test_weights_affect_ranking(self):
        """Test that weights affect final ranking."""
        dense_results = [("chunk_1", 0.9)]
        sparse_results = [("chunk_2", 5.5)]

        # High dense weight
        results_dense_heavy = reciprocal_rank_fusion(
            dense_results, sparse_results,
            dense_weight=0.9, sparse_weight=0.1
        )

        # High sparse weight
        results_sparse_heavy = reciprocal_rank_fusion(
            dense_results, sparse_results,
            dense_weight=0.1, sparse_weight=0.9
        )

        # Dense-heavy should favor chunk_1
        assert results_dense_heavy[0].chunk_id == "chunk_1"
        # Sparse-heavy should favor chunk_2
        assert results_sparse_heavy[0].chunk_id == "chunk_2"


# ============================================================================
# API Tests
# ============================================================================

@pytest.mark.asyncio
class TestSearchAPI:
    """Tests for search API endpoint."""

    async def test_search_empty_session(self, async_client: AsyncClient):
        """Test search on session with no indexed content."""
        response = await async_client.post(
            "/api/v1/sessions/999/search",
            json={"query": "test query"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["total_chunks_searched"] == 0

    async def test_search_request_validation(self, async_client: AsyncClient):
        """Test search request validation."""
        # Empty query
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={"query": ""},
        )
        assert response.status_code == 422

        # Query too long
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={"query": "x" * 2000},
        )
        assert response.status_code == 422

        # Invalid top_k
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={"query": "test", "top_k": 100},
        )
        assert response.status_code == 422

    async def test_search_sparse_mode(self, async_client: AsyncClient):
        """Test sparse-only search mode."""
        # First populate the index
        BM25IndexCache.populate_from_chunks(
            session_id=1,
            chunks=[
                ("chunk_1", "Python programming language"),
                ("chunk_2", "JavaScript web development"),
            ],
        )

        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={
                "query": "Python",
                "search_mode": "sparse",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["search_mode"] == "sparse"
        assert len(data["results"]) > 0

    async def test_search_response_structure(self, async_client: AsyncClient):
        """Test search response has correct structure."""
        BM25IndexCache.populate_from_chunks(
            session_id=1,
            chunks=[("chunk_1", "Test content")],
        )

        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={"query": "test"},
        )

        data = response.json()

        # Required fields
        assert "query" in data
        assert "results" in data
        assert "total_chunks_searched" in data
        assert "search_mode" in data
        assert "timing_ms" in data

        # Result structure
        if data["results"]:
            result = data["results"][0]
            assert "chunk_id" in result
            assert "score" in result

    async def test_search_stats_endpoint(self, async_client: AsyncClient):
        """Test search stats endpoint."""
        response = await async_client.get("/api/v1/sessions/1/search/stats")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "bm25_index_exists" in data
        assert "total_chunks" in data


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
class TestSearchIntegration:
    """Integration tests for end-to-end search flow."""

    def setup_method(self):
        """Set up test data."""
        BM25IndexCache._indexes.clear()

    async def test_full_search_flow(self, async_client: AsyncClient):
        """Test complete search flow from indexing to results."""
        # Simulate indexed content (would come from R4/R5 in real flow)
        chunks = [
            ("doc_1:chunk_1", "Authentication in web applications uses tokens like JWT."),
            ("doc_1:chunk_2", "JWTs contain encoded claims about the user."),
            ("doc_2:chunk_1", "Database connections should use connection pooling."),
            ("doc_2:chunk_2", "PostgreSQL supports advanced features like JSONB."),
        ]

        BM25IndexCache.populate_from_chunks(session_id=1, chunks=chunks)

        # Search for authentication
        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={
                "query": "How does JWT authentication work?",
                "top_k": 3,
                "search_mode": "sparse",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should find JWT-related chunks
        results = data["results"]
        assert len(results) > 0

        # Top results should be about JWT/auth
        top_chunk_ids = [r["chunk_id"] for r in results[:2]]
        assert any("doc_1" in cid for cid in top_chunk_ids)

    async def test_search_relevance_ordering(self, async_client: AsyncClient):
        """Test that results are ordered by relevance."""
        chunks = [
            ("chunk_1", "Python is a programming language."),
            ("chunk_2", "Java is also a programming language."),
            ("chunk_3", "Python Python Python - many mentions."),
        ]

        BM25IndexCache.populate_from_chunks(session_id=1, chunks=chunks)

        response = await async_client.post(
            "/api/v1/sessions/1/search",
            json={"query": "Python", "search_mode": "sparse"},
        )

        results = response.json()["results"]

        # Chunk with most Python mentions should score highest
        assert results[0]["chunk_id"] == "chunk_3"
```

---

## 11. Testing Strategy

### Test Categories

| Category | Coverage Target | Tools |
|----------|-----------------|-------|
| Unit Tests | BM25Index, RRF, tokenization | pytest |
| Integration Tests | Search API endpoint | pytest + httpx |
| Quality Tests | Relevance ranking | Golden set evaluation |
| Performance Tests | Latency benchmarks | pytest-benchmark |

### Quality Evaluation

Create a golden set for search quality evaluation:

```yaml
# tests/golden_search_queries.yaml
queries:
  - query: "How does JWT authentication work?"
    expected_topics: ["jwt", "authentication", "token"]
    expected_sources: ["auth-guide.md"]
    min_recall: 0.8

  - query: "database connection pooling"
    expected_topics: ["database", "connection", "pool"]
    expected_sources: ["db-config.md"]
    min_recall: 0.7
```

### Performance Benchmarks

Target latencies:
- BM25 search (1000 chunks): <50ms
- Dense search (1000 chunks): <100ms
- Hybrid search (1000 chunks): <200ms
- With reranking: <500ms total

---

## 12. Acceptance Criteria

### Functional Requirements

- [ ] BM25 index builds correctly from chunked documents
- [ ] BM25 search returns relevant results for keyword queries
- [ ] Dense search returns semantically similar results (after R5)
- [ ] RRF correctly combines rankings from both methods
- [ ] API endpoint validates request parameters
- [ ] API endpoint returns proper response structure
- [ ] Results include source attribution (chunk_id, source_url, source_title)
- [ ] Search respects session boundaries (no cross-session leakage)
- [ ] Empty sessions return empty results gracefully

### Non-Functional Requirements

- [ ] Search latency <200ms for 1000 chunks (without reranking)
- [ ] BM25 index memory footprint <100MB for 10,000 chunks
- [ ] Test coverage >80% for search module
- [ ] All Pydantic schemas validated with OpenAPI generation
- [ ] Search endpoints documented in OpenAPI spec

### Integration Requirements

- [ ] Works with R4 chunked output format
- [ ] Ready to integrate with R5 embedding service
- [ ] BM25 index cache integrates with article lifecycle events

---

## 13. Common Pitfalls

### Issue 1: BM25 Returns No Results

**Symptom:** BM25 search always returns empty list

**Solution:**
- Verify index was built (`index.build()` called)
- Check tokenization produces non-empty token lists
- Ensure query contains words >1 character
- Verify documents were added before building

### Issue 2: RRF Scores All Zero

**Symptom:** All RRF scores are 0.0

**Solution:**
- Verify at least one search method returns results
- Check weights are not both 0.0
- Ensure k parameter is positive

### Issue 3: Slow Search Performance

**Symptom:** Search takes >1 second

**Solution:**
- Check chunk count (>10K chunks may need optimization)
- Consider async processing for dense search
- Enable result caching for repeated queries
- Profile with `cProfile` to find bottlenecks

### Issue 4: Memory Growth with Sessions

**Symptom:** Memory increases as more sessions are searched

**Solution:**
- Implement LRU eviction for BM25IndexCache
- Set max cache size based on memory constraints
- Consider persisting indexes to disk for inactive sessions

---

## 14. Next Steps

After completing this plan:

1. **Integration with R5 (Embedding/Indexing):**
   - Connect dense search to ChromaDB populated by R5
   - Add embedding generation for queries
   - Enable full hybrid search mode

2. **Quality Optimization:**
   - Evaluate search quality on golden set
   - Tune RRF parameters based on results
   - Consider enabling reranking for quality-critical use cases

3. **Performance Optimization:**
   - Add result caching for frequent queries
   - Implement async query processing
   - Consider batch query support

4. **R7 (Q&A Synthesis):**
   - Use search results as context for LLM synthesis
   - Implement citation generation from search results
   - Add grounding checks against retrieved chunks

---

## 15. Success Criteria

- [ ] `/api/v1/sessions/{id}/search` endpoint operational
- [ ] Sparse (BM25) search returns relevant results
- [ ] Response includes source attribution for citations
- [ ] All tests pass (20+ tests for search module)
- [ ] Search latency <200ms for typical session (100-1000 chunks)
- [ ] OpenAPI spec includes search endpoint with proper schemas
- [ ] Integration ready for R5 embedding service
- [ ] BM25 index management handles article add/remove lifecycle

---

**Plan Status:** Ready for implementation
**Last Updated:** 2026-01-19
