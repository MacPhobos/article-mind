# Plan R8: Evidence and Traceability Improvements

**Plan ID:** R8-evidence-traceability
**Created:** 2026-01-20
**Dependencies:** R6 (Knowledge Query API), R7 (Chat Interface), R4 (Content Extraction)
**Estimated Time:** 3-4 days

---

## 1. Overview

### Purpose

Fix critical gaps in evidence and traceability throughout the article scraping and RAG pipeline to establish user trust and enable effective debugging. This plan addresses the complete disconnect between rich internal data and minimal user-facing visibility, making it impossible for users to verify answer quality or for developers to debug retrieval issues.

### Executive Summary

**Current State:**
- RAG pipeline is **completely non-functional** - search integration is stubbed out
- Article content is extracted but **never shown to users**
- Chat citations show only 200-char excerpts, not full chunks
- **Zero structured logging** in RAG pipeline for debugging
- No retrieval metadata in responses (scores, rankings, search method)

**Impact:**
- Users cannot verify what content was extracted from articles
- Users cannot assess citation relevance or answer grounding
- Developers cannot debug RAG issues (no logs, no traceability)
- Search API exists but is unused by chat pipeline

**This Plan Will:**
1. **Fix P0 Critical**: Connect RAG pipeline to HybridSearch (makes RAG functional)
2. **Fix P1 High Impact**: Enhanced citations, structured logging, content display
3. **Add P2 Medium**: Retrieval metadata persistence, search analytics

### Scope

**In Scope:**
- Fix RAG→Search integration (CRITICAL - chat is broken without this)
- Return full chunk content in citations (not 200-char excerpts)
- Add structured logging to RAG/search pipelines
- Display extracted article content and metadata in UI
- Add retrieval metadata to ChatResponse schema
- Persist RAG context to database for audit trails

**Out of Scope:**
- Advanced analytics dashboards (future enhancement)
- Real-time monitoring infrastructure (future enhancement)
- LLM response quality scoring (future enhancement)

### Research Foundation

Based on findings from:
- `/export/workspace/article-mind/docs/research/evidence-traceability-gap-analysis-2026-01-20.md`

Key findings applied:
- **Gap #1**: RAG pipeline returns empty chunks (search not connected)
- **Gap #2**: Limited citation evidence (200-char excerpts insufficient)
- **Gap #3**: No structured logging (cannot debug RAG issues)
- **Gap #4**: No article content verification (users can't audit extraction)

### Dependencies

- **R6 (Knowledge Query API):** Provides `/api/v1/sessions/{id}/search` endpoint
- **R7 (Chat Interface):** Chat endpoints and RAG pipeline to enhance
- **R4 (Content Extraction):** Article content to display

### Outputs

- **Working RAG integration** with HybridSearch
- Enhanced ChatSource schema with full content and metadata
- Structured logging throughout RAG/search pipelines
- Article content display in UI
- Retrieval metadata in database and API responses
- Comprehensive traceability for debugging

---

## 2. Architecture Overview

### Current vs. Target Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│ CURRENT STATE (BROKEN)                                                     │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  User Question                                                              │
│      │                                                                      │
│      ▼                                                                      │
│  RAGPipeline._retrieve_chunks()                                             │
│      │                                                                      │
│      └──> return []  ⚠️ ALWAYS EMPTY! Search not connected                │
│                                                                            │
│  LLM gets NO CONTEXT → Hallucinates without article knowledge              │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ TARGET STATE (THIS PLAN)                                                   │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  User Question                                                              │
│      │                                                                      │
│      ▼                                                                      │
│  RAGPipeline._retrieve_chunks()                                             │
│      │                                                                      │
│      ├──> Call HybridSearch.search()                                        │
│      │    ├─> DenseSearch (ChromaDB vector similarity)                      │
│      │    ├─> SparseSearch (BM25 keyword matching)                          │
│      │    └─> RRF Fusion (combine rankings)                                 │
│      │                                                                      │
│      ├──> ✅ Return chunks with:                                           │
│      │    ├─> Full content (not 200-char excerpt)                           │
│      │    ├─> Relevance scores (RRF, dense_rank, sparse_rank)               │
│      │    ├─> Source metadata (title, URL, article_id)                      │
│      │    └─> Search method (hybrid/dense/sparse)                           │
│      │                                                                      │
│      ▼                                                                      │
│  LLM Context (formatted with ALL chunk content)                             │
│      │                                                                      │
│      ▼                                                                      │
│  📝 Structured Logging at every step:                                      │
│      - Query received                                                       │
│      - Chunks retrieved (IDs, scores)                                       │
│      - Context formatted (length, preview)                                  │
│      - LLM response (tokens, model, provider)                               │
│      - Citations extracted                                                  │
│      │                                                                      │
│      ▼                                                                      │
│  💾 Persist to Database:                                                   │
│      - ChatMessage.retrieval_metadata (JSON)                                │
│      - ChatMessage.context_chunks (JSON)                                    │
│      │                                                                      │
│      ▼                                                                      │
│  📤 Return to User:                                                        │
│      - ChatResponse with full chunk citations                               │
│      - RetrievalMetadata (chunks retrieved/cited, search mode, timing)      │
│      - Sources with full content and scores                                 │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

### Enhanced Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. ARTICLE CONTENT VERIFICATION                                         │
└─────────────────────────────────────────────────────────────────────────┘

User → GET /api/v1/sessions/{id}/articles/{article_id}/content
         ↓
      ArticleContentResponse {
        id, title, content_text,
        extraction_status,
        ✨ NEW: word_count, reading_time, extraction_method,
        ✨ NEW: content_preview (first 500 chars),
        ✨ NEW: extraction_warnings []
      }
         ↓
      UI displays:
        ✅ Full extracted content (scrollable)
        ✅ Extraction quality metrics
        ✅ Warnings if any

┌─────────────────────────────────────────────────────────────────────────┐
│ 2. ENHANCED CHAT FLOW                                                   │
└─────────────────────────────────────────────────────────────────────────┘

User → POST /api/v1/sessions/{id}/chat
         ↓
      📝 LOG: "rag.query.start" {session_id, question}
         ↓
      RAGPipeline._retrieve_chunks()
         ↓
      ✨ FIXED: Call HybridSearch.search()
         ↓
      📝 LOG: "rag.query.chunks_retrieved" {chunks_count, chunk_ids, scores}
         ↓
      format_context_with_metadata(chunks)
         ↓
      📝 LOG: "rag.query.context_formatted" {context_length, preview}
         ↓
      LLM.generate(context_chunks)
         ↓
      📝 LOG: "rag.query.llm_response" {tokens_input, tokens_output, model}
         ↓
      extract_cited_sources()
         ↓
      📝 LOG: "rag.query.citations_extracted" {cited_count, total_retrieved}
         ↓
      💾 PERSIST: ChatMessage {
           content, sources,
           ✨ NEW: retrieval_metadata {
             search_mode, chunks_retrieved, chunks_cited,
             search_timing_ms, total_chunks_in_session
           },
           ✨ NEW: context_chunks [
             {chunk_id, content, score, dense_rank, sparse_rank, cited}
           ]
         }
         ↓
      ChatResponse {
        content, sources,
        ✨ ENHANCED: sources [{
          excerpt: FULL_CHUNK_CONTENT (not 200 chars!),
          ✨ NEW: relevance_score, search_method,
          ✨ NEW: dense_rank, sparse_rank
        }],
        ✨ NEW: retrieval_metadata {
          chunks_retrieved, chunks_cited,
          search_mode, timing_ms
        }
      }
```

---

## 3. Priority Breakdown

### P0 - Critical (Must Fix for RAG to Work)

**Issue:** RAG pipeline always returns empty context
**Impact:** Chat cannot answer from article knowledge (100% broken)
**Effort:** Medium (2-3 hours)

**Task:** Connect `RAGPipeline._retrieve_chunks()` to `HybridSearch`
- File: `article-mind-service/src/article_mind_service/chat/rag_pipeline.py`
- Current: Lines 934-994 return `[]` (stub)
- Fix: Call internal HybridSearch API with proper embedding generation

### P1 - High Impact (User Trust & Developer Experience)

**Issue 1:** Citations show only 200-char excerpts
**Impact:** Users cannot verify answer grounding
**Effort:** Low (1 hour)

**Task:** Return full chunk content in ChatSource
- File: `article-mind-service/src/article_mind_service/schemas/chat.py`
- Change: `ChatSource.excerpt` from 200 chars to full chunk content
- Rename field: `excerpt` → `content` for clarity

**Issue 2:** No search metadata in citations
**Impact:** Users don't know why sources were selected
**Effort:** Low (1 hour)

**Task:** Add search metadata to ChatSource
- Fields to add: `relevance_score`, `search_method`, `dense_rank`, `sparse_rank`
- File: `article-mind-service/schemas/chat.py`

**Issue 3:** No structured logging in RAG pipeline
**Impact:** Cannot debug retrieval or answer quality issues
**Effort:** Medium (2-3 hours)

**Task:** Add structured logging with `structlog`
- Install: `structlog>=24.1.0`
- Log at: query start, retrieval, context formatting, LLM response, citations
- Files: `chat/rag_pipeline.py`, `routers/chat.py`, `routers/search.py`

**Issue 4:** Article content never shown to users
**Impact:** Cannot verify extraction quality
**Effort:** Low (1-2 hours)

**Task:** Display article content and metadata in UI
- Verify `/articles/{id}/content` endpoint is called
- Add content preview to ArticleResponse
- Show word count, reading time, extraction warnings
- Files: `schemas/article.py`, frontend components

### P2 - Medium (Audit Trail & Analytics)

**Issue 1:** No retrieval metadata persisted
**Impact:** Cannot analyze answer quality over time
**Effort:** Medium (2-3 hours)

**Task:** Add retrieval metadata to ChatMessage model
- Fields: `retrieval_metadata` (JSONB), `context_chunks` (JSONB)
- Migration to add columns
- Store search mode, chunk counts, timing, full chunks
- File: `models/chat.py`

**Issue 2:** No retrieval context in API response
**Impact:** Cannot understand search performance
**Effort:** Low (1 hour)

**Task:** Add RetrievalMetadata to ChatResponse
- Schema: `chunks_retrieved`, `chunks_cited`, `search_mode`, `timing_ms`
- File: `schemas/chat.py`

---

## 4. Database Schema Changes

### ChatMessage Model Enhancements

```python
# src/article_mind_service/models/chat.py

class ChatMessage(Base):
    """Chat message with enhanced RAG traceability."""

    __tablename__ = "chat_messages"

    # Existing fields...
    id: Mapped[int]
    session_id: Mapped[int]
    role: Mapped[str]
    content: Mapped[str]
    sources: Mapped[dict | None]
    llm_provider: Mapped[str | None]
    llm_model: Mapped[str | None]
    tokens_used: Mapped[int | None]
    created_at: Mapped[datetime]

    # ✨ NEW: RAG metadata for audit and analysis
    retrieval_metadata: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Search and retrieval metadata for traceability"
    )
    context_chunks: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Full chunks used for context (audit trail)"
    )
```

### Retrieval Metadata Schema

```json
{
  "retrieval_metadata": {
    "search_mode": "hybrid",
    "chunks_retrieved": 5,
    "chunks_cited": 2,
    "search_timing_ms": 127,
    "total_chunks_in_session": 547,
    "dense_weight": 0.7,
    "sparse_weight": 0.3,
    "rrf_k": 60
  }
}
```

### Context Chunks Schema

```json
{
  "context_chunks": [
    {
      "chunk_id": "doc_123:chunk_5",
      "article_id": 123,
      "content": "Full chunk text content...",
      "score": 0.0156,
      "dense_rank": 2,
      "sparse_rank": 1,
      "cited": true
    },
    {
      "chunk_id": "doc_124:chunk_3",
      "article_id": 124,
      "content": "Another full chunk...",
      "score": 0.0142,
      "dense_rank": 3,
      "sparse_rank": 2,
      "cited": false
    }
  ]
}
```

### Migration

```python
"""add retrieval metadata to chat messages

Revision ID: xxx
Revises: previous_revision
Create Date: 2026-01-20
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'xxx'
down_revision = 'previous_revision'

def upgrade() -> None:
    op.add_column(
        'chat_messages',
        sa.Column(
            'retrieval_metadata',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment='Search and retrieval metadata for traceability'
        )
    )
    op.add_column(
        'chat_messages',
        sa.Column(
            'context_chunks',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment='Full chunks used for context (audit trail)'
        )
    )

def downgrade() -> None:
    op.drop_column('chat_messages', 'context_chunks')
    op.drop_column('chat_messages', 'retrieval_metadata')
```

---

## 5. API Schema Enhancements

### Enhanced ChatSource Schema

```python
# src/article_mind_service/schemas/chat.py

class ChatSource(BaseModel):
    """Enhanced source citation with full traceability."""

    citation_index: int = Field(..., description="Citation number [1], [2], etc.")
    article_id: int = Field(..., description="Source article ID")
    chunk_id: str | None = Field(default=None, description="Chunk identifier")
    title: str | None = Field(default=None, description="Article title")
    url: str | None = Field(default=None, description="Article URL")

    # ✨ CHANGED: Full content instead of 200-char excerpt
    content: str = Field(
        ...,
        description="Full chunk content (not truncated)",
        examples=["Authentication uses JWT tokens for stateless auth. The token contains..."]
    )

    # ✨ NEW: Search metadata
    relevance_score: float | None = Field(
        default=None,
        description="Combined relevance score (RRF)",
        ge=0.0,
        examples=[0.0156]
    )
    search_method: str | None = Field(
        default=None,
        description="Search method used: semantic, keyword, or hybrid",
        examples=["hybrid"]
    )
    dense_rank: int | None = Field(
        default=None,
        description="Rank from semantic vector search",
        ge=1,
        examples=[2]
    )
    sparse_rank: int | None = Field(
        default=None,
        description="Rank from BM25 keyword search",
        ge=1,
        examples=[1]
    )
```

### New RetrievalMetadata Schema

```python
# src/article_mind_service/schemas/chat.py

class RetrievalMetadata(BaseModel):
    """Metadata about the retrieval process for transparency."""

    chunks_retrieved: int = Field(
        ...,
        description="Total chunks retrieved from search",
        ge=0,
        examples=[5]
    )
    chunks_cited: int = Field(
        ...,
        description="Number of chunks actually cited in response",
        ge=0,
        examples=[2]
    )
    search_mode: str = Field(
        ...,
        description="Search mode used: hybrid, dense, or sparse",
        examples=["hybrid"]
    )
    total_chunks_in_session: int = Field(
        ...,
        description="Total indexed chunks in session",
        ge=0,
        examples=[547]
    )
    search_timing_ms: int = Field(
        ...,
        description="Search execution time in milliseconds",
        ge=0,
        examples=[127]
    )
```

### Enhanced ChatResponse Schema

```python
# src/article_mind_service/schemas/chat.py

class ChatResponse(BaseModel):
    """Enhanced chat response with full traceability."""

    message_id: int = Field(..., description="Assistant message ID")
    content: str = Field(..., description="Generated response with citations")

    # ✨ ENHANCED: Sources with full content and metadata
    sources: list[ChatSource] = Field(
        default_factory=list,
        description="Cited sources with full content and rankings"
    )

    llm_provider: str | None = Field(default=None, description="LLM provider")
    llm_model: str | None = Field(default=None, description="Model identifier")
    tokens_used: int | None = Field(default=None, description="Total tokens consumed")
    created_at: datetime = Field(..., description="Response timestamp")

    # ✨ NEW: Retrieval context metadata
    retrieval_metadata: RetrievalMetadata | None = Field(
        default=None,
        description="Search and retrieval metadata"
    )
```

### Enhanced ArticleResponse Schema

```python
# src/article_mind_service/schemas/article.py

class ArticleResponse(BaseModel):
    """Enhanced article response with extraction quality indicators."""

    # Existing fields...
    id: int
    session_id: int
    type: str
    original_url: str | None
    title: str | None
    extraction_status: str
    created_at: datetime

    # ✨ NEW: Content quality indicators
    has_content: bool = Field(
        default=False,
        description="Whether content_text is available"
    )
    word_count: int | None = Field(
        default=None,
        description="Word count of extracted content",
        examples=[2547]
    )
    reading_time_minutes: int | None = Field(
        default=None,
        description="Estimated reading time in minutes",
        examples=[12]
    )
    extraction_method: str | None = Field(
        default=None,
        description="Extraction method used",
        examples=["trafilatura", "playwright"]
    )
    extraction_warnings: list[str] | None = Field(
        default=None,
        description="Warnings or issues during extraction",
        examples=[["Partial content - JavaScript required"]]
    )
    content_preview: str | None = Field(
        default=None,
        description="First 500 characters of content",
        max_length=500
    )
```

---

## 6. P0 Implementation: Fix RAG Integration

### Current Broken Code

```python
# article-mind-service/src/article_mind_service/chat/rag_pipeline.py:934-994

async def _retrieve_chunks(
    self,
    session_id: int,
    query: str,
    limit: int,
    search_client: Any | None = None,
) -> list[dict]:
    """Retrieve relevant chunks from R6 search API.

    TODO: Integrate with R6 search API when available
    For now, return empty to demonstrate flow
    """

    # ⚠️ PROBLEM: Always returns empty!
    return []
```

### Fixed Implementation

```python
# article-mind-service/src/article_mind_service/chat/rag_pipeline.py

async def _retrieve_chunks(
    self,
    session_id: int,
    query: str,
    limit: int,
    search_client: Any | None = None,
) -> list[dict]:
    """Retrieve relevant chunks from R6 search API.

    Calls internal HybridSearch with full metadata.

    Args:
        session_id: Session to search within
        query: User's question
        limit: Max chunks to retrieve
        search_client: Optional client for testing (mocking)

    Returns:
        List of chunk dicts with content, scores, and metadata
    """
    # Use injected client for testing
    if search_client:
        results = await search_client.search(
            session_id=session_id,
            query=query,
            limit=limit,
        )
        return [
            {
                "content": r.get("content", ""),
                "article_id": r.get("article_id"),
                "chunk_id": r.get("chunk_id"),
                "title": r.get("source_title"),
                "url": r.get("source_url"),
                "score": r.get("score", 0.0),
                "dense_rank": r.get("dense_rank"),
                "sparse_rank": r.get("sparse_rank"),
            }
            for r in results.get("results", [])
        ]

    # ✅ FIXED: Call internal HybridSearch API
    from article_mind_service.search import HybridSearch
    from article_mind_service.schemas.search import SearchRequest, SearchMode
    from article_mind_service.embeddings import get_embedding_provider

    # Generate query embedding for dense search
    embedding_provider = get_embedding_provider()
    embeddings = await embedding_provider.embed([query])
    query_embedding = embeddings[0]

    # Build search request
    search_request = SearchRequest(
        query=query,
        top_k=limit,
        include_content=True,
        search_mode=SearchMode.HYBRID
    )

    # Execute hybrid search
    search = HybridSearch()
    response = await search.search(
        session_id=session_id,
        request=search_request,
        query_embedding=query_embedding
    )

    # Convert SearchResult to chunk dict
    chunks = []
    for result in response.results:
        chunks.append({
            "content": result.content or "",
            "article_id": result.article_id,
            "chunk_id": result.chunk_id,
            "title": result.source_title,
            "url": result.source_url,
            "score": result.score,
            "dense_rank": result.dense_rank,
            "sparse_rank": result.sparse_rank,
        })

    return chunks
```

### Integration Test

```python
# tests/integration/test_rag_integration.py
"""Integration test for RAG pipeline with HybridSearch."""

import pytest
from httpx import AsyncClient

from article_mind_service.search import BM25IndexCache


@pytest.mark.asyncio
async def test_rag_retrieves_chunks_from_search(
    async_client: AsyncClient,
    test_session,
    db
):
    """Test that RAG pipeline retrieves chunks from search."""

    # Setup: Populate search index
    BM25IndexCache.populate_from_chunks(
        session_id=test_session.id,
        chunks=[
            ("chunk_1", "JWT authentication tokens are used for stateless auth."),
            ("chunk_2", "OAuth2 provides delegated authorization."),
        ]
    )

    # Send chat message about authentication
    response = await async_client.post(
        f"/api/v1/sessions/{test_session.id}/chat",
        json={"message": "How does JWT authentication work?"}
    )

    assert response.status_code == 200
    data = response.json()

    # Verify chunks were retrieved
    assert len(data["sources"]) > 0, "Expected citations from retrieved chunks"

    # Verify content is returned (not empty)
    assert data["content"] != "", "Expected non-empty response"
    assert "JWT" in data["content"] or "authentication" in data["content"]
```

---

## 7. P1 Implementation: Structured Logging

### Install structlog

```toml
# pyproject.toml
[project]
dependencies = [
    # Existing...
    "structlog>=24.1.0",
]
```

### Configure structlog

```python
# src/article_mind_service/logging_config.py
"""Structured logging configuration."""

import logging
import sys
import structlog


def configure_logging(log_level: str = "INFO"):
    """Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
```

### Add Logging to RAG Pipeline

```python
# src/article_mind_service/chat/rag_pipeline.py

import structlog

logger = structlog.get_logger(__name__)


async def query(
    self,
    session_id: int,
    question: str,
    db: AsyncSession,
    search_client: Any | None = None,
) -> RAGResponse:
    """Execute RAG pipeline with comprehensive logging."""

    # ✨ LOG: Query start
    logger.info(
        "rag.query.start",
        session_id=session_id,
        question=question,
        max_context_chunks=self.max_context_chunks
    )

    # Step 1: Retrieve chunks
    chunks = await self._retrieve_chunks(
        session_id=session_id,
        query=question,
        limit=self.max_context_chunks,
        search_client=search_client,
    )

    # ✨ LOG: Chunks retrieved
    logger.info(
        "rag.query.chunks_retrieved",
        session_id=session_id,
        chunks_count=len(chunks),
        chunk_ids=[c.get("chunk_id") for c in chunks],
        scores=[c.get("score") for c in chunks]
    )

    # Step 2: Format context
    context_str, source_metadata = format_context_with_metadata(chunks)
    has_context = len(chunks) > 0

    # ✨ LOG: Context formatted
    logger.debug(
        "rag.query.context_formatted",
        session_id=session_id,
        context_length=len(context_str),
        context_preview=context_str[:500],
        sources_count=len(source_metadata)
    )

    # Step 3: Generate answer
    system_prompt = build_system_prompt(has_context=has_context)

    llm_response = await self.llm_provider.generate(
        system_prompt=system_prompt,
        user_message=question,
        context_chunks=[context_str] if has_context else [],
        max_tokens=settings.llm_max_tokens,
        temperature=0.3,
    )

    # ✨ LOG: LLM response
    logger.info(
        "rag.query.llm_response",
        session_id=session_id,
        provider=llm_response.provider,
        model=llm_response.model,
        tokens_input=llm_response.tokens_input,
        tokens_output=llm_response.tokens_output,
        response_length=len(llm_response.content)
    )

    # Step 4: Extract citations
    cited_sources = self._extract_cited_sources(
        content=llm_response.content,
        source_metadata=source_metadata,
    )

    # ✨ LOG: Citations extracted
    logger.info(
        "rag.query.citations_extracted",
        session_id=session_id,
        cited_count=len(cited_sources),
        total_retrieved=len(chunks),
        citation_rate=len(cited_sources) / len(chunks) if chunks else 0
    )

    return RAGResponse(
        content=llm_response.content,
        sources=cited_sources,
        llm_provider=llm_response.provider,
        llm_model=llm_response.model,
        tokens_used=llm_response.total_tokens,
        chunks_retrieved=len(chunks),
    )
```

### Add Logging to Search Router

```python
# src/article_mind_service/routers/search.py

import structlog

logger = structlog.get_logger(__name__)


@router.post("/sessions/{session_id}/search", response_model=SearchResponse)
async def search_session(
    session_id: int,
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    search: HybridSearch = Depends(get_hybrid_search),
) -> SearchResponse:
    """Search session knowledge with structured logging."""

    # ✨ LOG: Search start
    logger.info(
        "search.request.start",
        session_id=session_id,
        query=request.query,
        top_k=request.top_k,
        search_mode=request.search_mode.value
    )

    # Execute search
    response = await search.search(
        session_id=session_id,
        request=request,
        query_embedding=query_embedding,
    )

    # ✨ LOG: Search results
    logger.info(
        "search.request.completed",
        session_id=session_id,
        results_count=len(response.results),
        total_chunks_searched=response.total_chunks_searched,
        search_mode=response.search_mode.value,
        timing_ms=response.timing_ms
    )

    return response
```

---

## 8. P1 Implementation: Enhanced Citations

### Update ChatSource Schema

```python
# src/article_mind_service/schemas/chat.py

class ChatSource(BaseModel):
    """Enhanced source citation with full content and metadata."""

    citation_index: int = Field(..., description="Citation number [1], [2], etc.")
    article_id: int = Field(..., description="Source article ID")
    chunk_id: str | None = Field(default=None, description="Chunk identifier")
    title: str | None = Field(default=None, description="Article title")
    url: str | None = Field(default=None, description="Article URL")

    # ✨ CHANGED: Full content instead of excerpt
    content: str = Field(
        ...,
        description="Full chunk content (not truncated)"
    )

    # ✨ NEW: Search metadata
    relevance_score: float | None = Field(default=None, description="RRF score", ge=0.0)
    search_method: str | None = Field(default=None, description="hybrid/dense/sparse")
    dense_rank: int | None = Field(default=None, description="Semantic search rank", ge=1)
    sparse_rank: int | None = Field(default=None, description="Keyword search rank", ge=1)
```

### Update Prompt Formatting

```python
# src/article_mind_service/chat/prompts.py

def format_context_with_metadata(
    chunks: list[dict],
) -> tuple[str, list[dict]]:
    """Format context chunks with enhanced metadata.

    Returns full chunk content and enriched source metadata.
    """
    if not chunks:
        return "No relevant context found.", []

    formatted_lines = []
    source_metadata = []

    for i, chunk in enumerate(chunks, start=1):
        content = chunk.get("content", "")
        title = chunk.get("title", "Unknown Source")
        url = chunk.get("url", "")

        # Format context line
        source_ref = f"(Source: {title}"
        if url:
            source_ref += f", URL: {url}"
        source_ref += ")"

        formatted_lines.append(f"[{i}] {content}\n{source_ref}")

        # ✨ ENHANCED: Track full metadata including search scores
        source_metadata.append({
            "citation_index": i,
            "article_id": chunk.get("article_id"),
            "chunk_id": chunk.get("chunk_id"),
            "title": title,
            "url": url,
            "content": content,  # ✨ CHANGED: Full content, not excerpt
            "relevance_score": chunk.get("score"),
            "search_method": "hybrid",  # From search mode
            "dense_rank": chunk.get("dense_rank"),
            "sparse_rank": chunk.get("sparse_rank"),
        })

    return "\n\n".join(formatted_lines), source_metadata
```

---

## 9. P1 Implementation: Article Content Display

### Enhance ArticleResponse

```python
# src/article_mind_service/schemas/article.py

class ArticleResponse(BaseModel):
    """Enhanced article response with quality indicators."""

    # Existing fields...
    id: int
    session_id: int
    type: str
    original_url: str | None
    title: str | None
    extraction_status: str
    created_at: datetime

    # ✨ NEW: Content quality indicators
    has_content: bool = Field(default=False, description="Content available")
    word_count: int | None = Field(default=None, description="Word count")
    reading_time_minutes: int | None = Field(default=None, description="Reading time")
    extraction_method: str | None = Field(default=None, description="Extraction method")
    extraction_warnings: list[str] | None = Field(default=None, description="Warnings")
    content_preview: str | None = Field(default=None, description="First 500 chars", max_length=500)

    model_config = {"from_attributes": True}
```

### Update article_to_response Helper

```python
# src/article_mind_service/routers/articles.py

def article_to_response(article: Article) -> ArticleResponse:
    """Convert Article model to ArticleResponse with enhancements."""

    # Calculate content preview
    content_preview = None
    if article.content_text:
        content_preview = article.content_text[:500]

    # Extract warnings from metadata
    warnings = None
    if article.extraction_metadata:
        warnings = article.extraction_metadata.get("warnings", [])

    return ArticleResponse(
        id=article.id,
        session_id=article.session_id,
        type=article.type,
        original_url=article.original_url,
        title=article.title,
        extraction_status=article.extraction_status,
        created_at=article.created_at,
        # ✨ NEW fields
        has_content=bool(article.content_text),
        word_count=article.word_count,
        reading_time_minutes=article.reading_time_minutes,
        extraction_method=article.extraction_method,
        extraction_warnings=warnings,
        content_preview=content_preview,
    )
```

### Frontend: Article Content Display Component

```svelte
<!-- src/lib/components/articles/ArticleContentView.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import type { components } from '$lib/api/generated';
  import { getArticleContent } from '$lib/api/articles';

  type ArticleContent = components['schemas']['ArticleContentResponse'];

  interface Props {
    sessionId: number;
    articleId: number;
  }
  let { sessionId, articleId }: Props = $props();

  let content = $state<ArticleContent | null>(null);
  let isLoading = $state(false);
  let error = $state<string | null>(null);

  onMount(async () => {
    await loadContent();
  });

  async function loadContent() {
    isLoading = true;
    error = null;
    try {
      content = await getArticleContent(sessionId, articleId);
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load content';
    } finally {
      isLoading = false;
    }
  }
</script>

<div class="article-content-view">
  {#if isLoading}
    <div class="loading">Loading content...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else if content}
    <div class="content-header">
      <h2>{content.title || 'Untitled'}</h2>
      <div class="metadata">
        <span>Status: {content.extraction_status}</span>
        {#if content.word_count}
          <span>Words: {content.word_count.toLocaleString()}</span>
        {/if}
        {#if content.reading_time_minutes}
          <span>Reading time: ~{content.reading_time_minutes} min</span>
        {/if}
      </div>
    </div>

    <div class="content-body">
      {content.content_text}
    </div>
  {/if}
</div>

<style>
  .article-content-view {
    padding: 1.5rem;
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }

  .content-header {
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e5e7eb;
  }

  .content-header h2 {
    margin: 0 0 0.5rem 0;
    font-size: 1.5rem;
  }

  .metadata {
    display: flex;
    gap: 1rem;
    font-size: 0.875rem;
    color: #6b7280;
  }

  .content-body {
    line-height: 1.7;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .loading, .error {
    padding: 2rem;
    text-align: center;
  }

  .error {
    color: #dc2626;
  }
</style>
```

---

## 10. P2 Implementation: Retrieval Metadata Persistence

### Enhanced RAGResponse

```python
# src/article_mind_service/chat/rag_pipeline.py

@dataclass
class RAGResponse:
    """Enhanced RAG response with retrieval metadata."""

    content: str
    sources: list[dict]
    llm_provider: str
    llm_model: str
    tokens_used: int
    chunks_retrieved: int

    # ✨ NEW: Full retrieval metadata
    retrieval_metadata: dict
    context_chunks: list[dict]
```

### Update RAGPipeline.query()

```python
# src/article_mind_service/chat/rag_pipeline.py

async def query(
    self,
    session_id: int,
    question: str,
    db: AsyncSession,
    search_client: Any | None = None,
) -> RAGResponse:
    """Execute RAG pipeline with full metadata tracking."""

    start_time = time.time()

    # Retrieve chunks
    chunks = await self._retrieve_chunks(...)

    # Format context
    context_str, source_metadata = format_context_with_metadata(chunks)

    # Generate answer
    llm_response = await self.llm_provider.generate(...)

    # Extract citations
    cited_sources = self._extract_cited_sources(...)

    search_timing_ms = int((time.time() - start_time) * 1000)

    # ✨ NEW: Build retrieval metadata
    retrieval_metadata = {
        "search_mode": "hybrid",
        "chunks_retrieved": len(chunks),
        "chunks_cited": len(cited_sources),
        "search_timing_ms": search_timing_ms,
        "total_chunks_in_session": 0,  # TODO: Get from search response
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
        "rrf_k": 60,
    }

    # ✨ NEW: Build context chunks audit trail
    context_chunks = []
    for chunk in chunks:
        context_chunks.append({
            "chunk_id": chunk.get("chunk_id"),
            "article_id": chunk.get("article_id"),
            "content": chunk.get("content"),
            "score": chunk.get("score"),
            "dense_rank": chunk.get("dense_rank"),
            "sparse_rank": chunk.get("sparse_rank"),
            "cited": any(
                s["chunk_id"] == chunk.get("chunk_id")
                for s in cited_sources
            ),
        })

    return RAGResponse(
        content=llm_response.content,
        sources=cited_sources,
        llm_provider=llm_response.provider,
        llm_model=llm_response.model,
        tokens_used=llm_response.total_tokens,
        chunks_retrieved=len(chunks),
        retrieval_metadata=retrieval_metadata,
        context_chunks=context_chunks,
    )
```

### Update Chat Router to Persist Metadata

```python
# src/article_mind_service/routers/chat.py

@router.post("", response_model=ChatResponse)
async def send_chat_message(
    session_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """Send chat message with full metadata persistence."""

    # Save user message
    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=request.message,
    )
    db.add(user_message)
    await db.flush()

    # Execute RAG pipeline
    pipeline = RAGPipeline()
    rag_response = await pipeline.query(
        session_id=session_id,
        question=request.message,
        db=db,
    )

    # ✨ ENHANCED: Save with retrieval metadata
    assistant_message = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=rag_response.content,
        sources=rag_response.sources,
        llm_provider=rag_response.llm_provider,
        llm_model=rag_response.llm_model,
        tokens_used=rag_response.tokens_used,
        # ✨ NEW: Persist retrieval metadata
        retrieval_metadata=rag_response.retrieval_metadata,
        context_chunks=rag_response.context_chunks,
    )
    db.add(assistant_message)
    await db.commit()

    # ✨ ENHANCED: Return with metadata
    return ChatResponse(
        message_id=assistant_message.id,
        content=assistant_message.content,
        sources=[ChatSource(**s) for s in rag_response.sources],
        llm_provider=rag_response.llm_provider,
        llm_model=rag_response.llm_model,
        tokens_used=rag_response.tokens_used,
        created_at=assistant_message.created_at,
        # ✨ NEW: Include retrieval metadata
        retrieval_metadata=RetrievalMetadata(
            **rag_response.retrieval_metadata
        ),
    )
```

---

## 11. Implementation Steps

### Phase 1: P0 Critical Fix (Day 1)

| Step | Task | Time | Files |
|------|------|------|-------|
| 1.1 | Fix RAG→Search integration in `_retrieve_chunks()` | 2h | `chat/rag_pipeline.py` |
| 1.2 | Write integration test for RAG retrieval | 1h | `tests/integration/test_rag_integration.py` |
| 1.3 | Verify chunks flow to LLM context | 30m | Manual testing |
| 1.4 | Test end-to-end chat with actual retrieval | 30m | Manual testing |

**Total:** 4 hours

### Phase 2: P1 Structured Logging (Day 1-2)

| Step | Task | Time | Files |
|------|------|------|-------|
| 2.1 | Add `structlog>=24.1.0` to dependencies | 15m | `pyproject.toml` |
| 2.2 | Create `logging_config.py` with structlog setup | 30m | `logging_config.py` |
| 2.3 | Add logging to RAG pipeline | 1h | `chat/rag_pipeline.py` |
| 2.4 | Add logging to search router | 30m | `routers/search.py` |
| 2.5 | Add logging to chat router | 30m | `routers/chat.py` |
| 2.6 | Test logs in development | 30m | Manual testing |

**Total:** 3.5 hours

### Phase 3: P1 Enhanced Citations (Day 2)

| Step | Task | Time | Files |
|------|------|------|-------|
| 3.1 | Update ChatSource schema with metadata fields | 30m | `schemas/chat.py` |
| 3.2 | Update `format_context_with_metadata()` | 30m | `chat/prompts.py` |
| 3.3 | Update frontend to display full content | 1h | Frontend components |
| 3.4 | Test citation display in UI | 30m | Manual testing |

**Total:** 2.5 hours

### Phase 4: P1 Article Content Display (Day 2)

| Step | Task | Time | Files |
|------|------|------|-------|
| 4.1 | Enhance ArticleResponse schema | 30m | `schemas/article.py` |
| 4.2 | Update `article_to_response()` helper | 30m | `routers/articles.py` |
| 4.3 | Create ArticleContentView component | 1h | Frontend component |
| 4.4 | Integrate into article detail page | 30m | Frontend routing |
| 4.5 | Test content display | 30m | Manual testing |

**Total:** 3 hours

### Phase 5: P2 Retrieval Metadata (Day 3)

| Step | Task | Time | Files |
|------|------|------|-------|
| 5.1 | Create migration for retrieval metadata | 30m | Alembic migration |
| 5.2 | Run migration | 15m | Database |
| 5.3 | Add RetrievalMetadata schema | 30m | `schemas/chat.py` |
| 5.4 | Update RAGResponse with metadata | 1h | `chat/rag_pipeline.py` |
| 5.5 | Update chat router to persist metadata | 1h | `routers/chat.py` |
| 5.6 | Test metadata persistence | 30m | Integration tests |

**Total:** 3.5 hours

### Phase 6: Testing & Polish (Day 3-4)

| Step | Task | Time | Files |
|------|------|------|-------|
| 6.1 | Write unit tests for enhanced schemas | 1h | `tests/unit/` |
| 6.2 | Write integration tests for full flow | 2h | `tests/integration/` |
| 6.3 | End-to-end testing with real articles | 1h | Manual testing |
| 6.4 | Documentation update | 1h | README, API docs |
| 6.5 | Error handling improvements | 1h | Multiple files |

**Total:** 6 hours

---

## 12. Testing Strategy

### Unit Tests

#### Test Enhanced ChatSource

```python
# tests/unit/schemas/test_chat_schemas.py
"""Test enhanced chat schemas."""

import pytest
from pydantic import ValidationError

from article_mind_service.schemas.chat import ChatSource, RetrievalMetadata


def test_chat_source_with_full_content():
    """Test ChatSource accepts full content."""
    source = ChatSource(
        citation_index=1,
        article_id=123,
        chunk_id="doc_123:chunk_5",
        title="Test Article",
        url="https://example.com",
        content="This is the full chunk content, not just 200 chars. " * 10,
        relevance_score=0.0156,
        search_method="hybrid",
        dense_rank=2,
        sparse_rank=1,
    )

    assert len(source.content) > 200
    assert source.relevance_score == 0.0156
    assert source.search_method == "hybrid"


def test_retrieval_metadata_validation():
    """Test RetrievalMetadata schema validation."""
    metadata = RetrievalMetadata(
        chunks_retrieved=5,
        chunks_cited=2,
        search_mode="hybrid",
        total_chunks_in_session=547,
        search_timing_ms=127,
    )

    assert metadata.chunks_retrieved == 5
    assert metadata.chunks_cited == 2

    # Test validation
    with pytest.raises(ValidationError):
        RetrievalMetadata(
            chunks_retrieved=-1,  # Invalid: negative
            chunks_cited=2,
            search_mode="hybrid",
            total_chunks_in_session=547,
            search_timing_ms=127,
        )
```

### Integration Tests

#### Test RAG Retrieval

```python
# tests/integration/test_rag_retrieval.py
"""Integration tests for RAG retrieval."""

import pytest
from httpx import AsyncClient

from article_mind_service.search import BM25IndexCache


@pytest.mark.asyncio
async def test_rag_retrieves_and_cites_chunks(
    async_client: AsyncClient,
    test_session,
    db
):
    """Test full RAG flow with retrieval and citations."""

    # Setup: Populate search index
    BM25IndexCache.populate_from_chunks(
        session_id=test_session.id,
        chunks=[
            ("chunk_1", "JWT authentication uses tokens for stateless auth."),
            ("chunk_2", "Tokens contain encoded user claims and signatures."),
            ("chunk_3", "OAuth2 provides delegated authorization framework."),
        ]
    )

    # Send question
    response = await async_client.post(
        f"/api/v1/sessions/{test_session.id}/chat",
        json={"message": "How does JWT authentication work?"}
    )

    assert response.status_code == 200
    data = response.json()

    # Verify sources have full content
    assert len(data["sources"]) > 0
    for source in data["sources"]:
        assert len(source["content"]) > 200, "Expected full chunk content"
        assert "relevance_score" in source
        assert "search_method" in source

    # Verify retrieval metadata
    assert "retrieval_metadata" in data
    metadata = data["retrieval_metadata"]
    assert metadata["chunks_retrieved"] > 0
    assert metadata["search_mode"] == "hybrid"
    assert metadata["search_timing_ms"] > 0


@pytest.mark.asyncio
async def test_chat_message_persists_retrieval_metadata(
    async_client: AsyncClient,
    test_session,
    db
):
    """Test retrieval metadata is persisted to database."""

    # Setup index
    BM25IndexCache.populate_from_chunks(
        session_id=test_session.id,
        chunks=[("chunk_1", "Test content about authentication.")]
    )

    # Send chat message
    response = await async_client.post(
        f"/api/v1/sessions/{test_session.id}/chat",
        json={"message": "Tell me about authentication"}
    )

    message_id = response.json()["message_id"]

    # Query database directly
    from article_mind_service.models.chat import ChatMessage
    from sqlalchemy import select

    result = await db.execute(
        select(ChatMessage).where(ChatMessage.id == message_id)
    )
    message = result.scalar_one()

    # Verify metadata was persisted
    assert message.retrieval_metadata is not None
    assert message.context_chunks is not None
    assert message.retrieval_metadata["chunks_retrieved"] > 0
    assert len(message.context_chunks) > 0
```

### Logging Tests

```python
# tests/unit/chat/test_logging.py
"""Test structured logging in RAG pipeline."""

import pytest
from unittest.mock import AsyncMock, patch
import structlog.testing

from article_mind_service.chat.rag_pipeline import RAGPipeline


@pytest.mark.asyncio
async def test_rag_pipeline_logs_query_flow(mock_llm_provider, mock_search_client):
    """Test that RAG pipeline logs all steps."""

    with structlog.testing.capture_logs() as cap_logs:
        pipeline = RAGPipeline()
        pipeline._llm_provider = mock_llm_provider

        await pipeline.query(
            session_id=1,
            question="Test question?",
            db=AsyncMock(),
            search_client=mock_search_client,
        )

        # Verify all log events
        events = [log["event"] for log in cap_logs]

        assert "rag.query.start" in events
        assert "rag.query.chunks_retrieved" in events
        assert "rag.query.context_formatted" in events
        assert "rag.query.llm_response" in events
        assert "rag.query.citations_extracted" in events
```

---

## 13. Acceptance Criteria

### Functional Requirements

#### P0 - Critical
- [ ] RAG pipeline retrieves chunks from HybridSearch (not empty)
- [ ] Chat responses include content from indexed articles
- [ ] LLM context contains retrieved chunk content
- [ ] Integration tests verify chunk retrieval

#### P1 - High Impact
- [ ] ChatSource returns full chunk content (>200 chars)
- [ ] Citations include relevance scores and rankings
- [ ] Structured logging at all RAG pipeline steps
- [ ] Logs are JSON-formatted for parsing
- [ ] Article content displayed in UI
- [ ] Extraction metadata shown (word count, reading time)

#### P2 - Medium
- [ ] retrieval_metadata persisted to chat_messages table
- [ ] context_chunks persisted for audit trail
- [ ] ChatResponse includes RetrievalMetadata
- [ ] Migration runs successfully

### Non-Functional Requirements

- [ ] P0 fix deployed within 1 day (RAG is critical)
- [ ] Logs do not contain sensitive user data
- [ ] Database migration is reversible
- [ ] Test coverage >80% for new code
- [ ] No performance degradation from logging

### API Contract Compliance

- [ ] ChatSource schema updated in OpenAPI spec
- [ ] RetrievalMetadata schema documented
- [ ] ArticleResponse enhancements in spec
- [ ] All new fields have examples
- [ ] Breaking changes versioned appropriately

---

## 14. Risks and Mitigations

### Risk 1: Search Integration Breaks Existing Flow

**Probability:** Medium
**Impact:** High

**Mitigation:**
- Add feature flag for search integration
- Comprehensive integration tests before deployment
- Gradual rollout with monitoring

### Risk 2: Full Content in Citations Increases Response Size

**Probability:** High
**Impact:** Low

**Mitigation:**
- Monitor API response sizes
- Consider pagination for chat history
- Frontend handles large responses gracefully

### Risk 3: Structured Logging Overhead

**Probability:** Low
**Impact:** Low

**Mitigation:**
- Use async logging where possible
- Configure log levels appropriately
- Profile performance impact

### Risk 4: Database Migration Failure

**Probability:** Low
**Impact:** Medium

**Mitigation:**
- Test migration on staging first
- Backup database before migration
- Have rollback plan ready
- Make columns nullable for safety

---

## 15. Success Metrics

### P0 Critical Success Metrics

- **RAG Functionality**: >0 chunks retrieved per query (was 0)
- **Citation Rate**: >70% of queries include citations
- **Answer Quality**: Manual spot-checks show grounding in articles

### P1 High Impact Metrics

- **Content Visibility**: Average citation content length >500 chars (was 200)
- **Logging Coverage**: 100% of RAG steps logged
- **Article Views**: Users click "View Content" on >30% of articles
- **Metadata Completeness**: 100% of responses include search metadata

### P2 Medium Metrics

- **Audit Trail**: 100% of chat messages have retrieval_metadata
- **Analytics Queries**: Ability to query chunk citation frequency
- **Debugging Speed**: Time to diagnose RAG issues reduced by 80%

### Developer Experience Metrics

- **Debug Time**: <5 minutes to trace RAG issue (was hours)
- **Log Queries**: Structured logs enable analytics queries
- **Incident Response**: Can identify bad retrievals in <10 minutes

---

## 16. Future Enhancements

### Phase 2 Enhancements (Future Plans)

**Advanced Analytics:**
- Article citation frequency reports
- Source quality scoring based on citation rate
- Search performance dashboards
- Token cost tracking per session

**Enhanced Logging:**
- Distributed tracing with correlation IDs
- Performance profiling hooks
- LLM response quality metrics
- User feedback integration

**Retrieval Improvements:**
- Confidence scoring for answers
- Alternative chunk suggestions
- Citation quality indicators
- Source diversity metrics

**UI Enhancements:**
- Citation highlight on hover
- Interactive chunk exploration
- Search explanation tooltips
- Quality indicator badges

---

## 17. Dependencies to Add

### Backend (pyproject.toml)

```toml
[project]
dependencies = [
    # Existing...
    "structlog>=24.1.0",
]
```

### Configuration Updates

```env
# .env additions

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json  # json or text
```

---

## 18. Rollout Plan

### Stage 1: P0 Critical Fix (Immediate)

**Duration:** 1 day
**Scope:** RAG→Search integration only
**Risk:** Low (restores basic functionality)

**Steps:**
1. Deploy P0 fix to staging
2. Run integration tests
3. Manual verification with real queries
4. Deploy to production
5. Monitor error rates and response times

### Stage 2: P1 Logging & Citations (Week 1)

**Duration:** 2 days
**Scope:** Structured logging, enhanced citations
**Risk:** Low (additive changes)

**Steps:**
1. Deploy logging configuration
2. Verify log aggregation works
3. Deploy citation enhancements
4. Monitor log volume and performance
5. Train team on log queries

### Stage 3: P1 Content Display (Week 1)

**Duration:** 1 day
**Scope:** Article content UI
**Risk:** Low (UI only)

**Steps:**
1. Deploy frontend changes
2. Verify content loads correctly
3. Monitor API response times
4. Gather user feedback

### Stage 4: P2 Metadata Persistence (Week 2)

**Duration:** 1 day
**Scope:** Database changes, audit trail
**Risk:** Medium (schema migration)

**Steps:**
1. Run migration on staging
2. Verify data persists correctly
3. Test rollback procedure
4. Deploy to production during low-traffic
5. Monitor database performance

---

**Plan Status:** Ready for implementation
**Last Updated:** 2026-01-20
**Estimated Total Time:** 3-4 days
**Priority:** CRITICAL (P0 must be fixed immediately)
