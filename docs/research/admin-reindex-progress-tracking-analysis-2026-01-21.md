# Admin Reindex Feature with Progress Tracking - Codebase Analysis

**Date:** 2026-01-21
**Researcher:** Claude (Research Agent)
**Context:** Implementation planning for admin reindexing endpoint with progress tracking

## Executive Summary

This analysis examines the article-mind-service codebase to understand existing patterns for implementing an admin-level reindexing feature with progress tracking. The service currently has single-session reindexing using FastAPI BackgroundTasks. This research identifies the architecture, patterns, and recommended approach for extending this to all-sessions reindexing with real-time progress updates.

**Key Findings:**
- Existing reindex implementation uses FastAPI BackgroundTasks (fire-and-forget)
- No task queue infrastructure (Celery/ARQ/RQ) currently in use
- No SSE or WebSocket infrastructure for streaming progress
- Database models track embedding status per article
- Embedding pipeline processes articles sequentially with batch chunking

**Recommended Approach:** SSE (Server-Sent Events) with in-memory progress tracking

---

## 1. Current Reindex Implementation

### 1.1 Single-Session Reindex Endpoint

**Location:** `src/article_mind_service/routers/sessions.py`

**Endpoint:** `POST /api/v1/sessions/{session_id}/reindex`

**How It Works:**

1. **Article Selection Criteria:**
   ```python
   # Only reindex articles with:
   # - extraction_status = "completed" (has content)
   # - embedding_status = "pending" OR "failed"
   # - deleted_at IS NULL (not soft-deleted)
   ```

2. **Background Task Pattern:**
   ```python
   # Uses FastAPI BackgroundTasks
   for article in articles:
       background_tasks.add_task(
           _reindex_article,
           article_id=article.id,
           session_id=str(session.id),
           text=article.content_text,
           source_url=article.original_url or article.canonical_url or "",
           pipeline=pipeline,
           db=db,
       )
   ```

3. **Response Schema:**
   ```python
   class ReindexResponse(BaseModel):
       session_id: int
       articles_queued: int  # Count of articles queued
       article_ids: list[int]  # IDs of queued articles
   ```

4. **Design Trade-offs:**
   - ✅ **Fast response:** Returns immediately with queued count
   - ✅ **Fire-and-forget:** No blocking on embedding completion
   - ❌ **No progress tracking:** Client can't monitor progress
   - ❌ **No failure visibility:** Can't tell if embedding fails after queuing

### 1.2 Reindex Helper Function

**Location:** `src/article_mind_service/routers/sessions.py:470`

```python
async def _reindex_article(
    article_id: int,
    session_id: str,
    text: str | None,
    source_url: str,
    pipeline: "EmbeddingPipeline",
    db: AsyncSession,
) -> None:
    """Background task to reindex a single article."""

    # Error handling: Catches all exceptions
    # Logs errors but doesn't propagate
    # Updates article.embedding_status to "failed" on error
```

**Key Characteristics:**
- Self-contained error handling
- No progress reporting mechanism
- Logs to standard logging (no structured events)
- Updates database directly (no intermediate state)

---

## 2. Embedding Pipeline Architecture

### 2.1 Pipeline Flow

**Location:** `src/article_mind_service/embeddings/pipeline.py`

**Process Flow:**

```
1. Update article.embedding_status = "processing"
   ↓
2. Chunk text using TextChunker (batch_size=100)
   ↓
3. For each batch:
   - Generate embeddings (OpenAI or Ollama)
   - Store in ChromaDB
   - Add to BM25 index
   ↓
4. Update article.embedding_status = "completed"
   Set article.chunk_count = total_chunks
   ↓
5. Commit database changes
```

**Batch Processing:**
- Default batch size: 100 chunks
- Sequential batch processing (not parallel)
- Progress updates ONLY at database level (status field)

### 2.2 Progress Information Available

**During Processing:**
- `embedding_status`: "pending" → "processing" → "completed"/"failed"
- `chunk_count`: Total chunks created (set on completion)

**NOT Available:**
- Current chunk being processed
- Percentage complete
- ETA or time remaining
- Real-time events

### 2.3 Performance Characteristics

**From Pipeline Documentation:**
- 1K words: ~1 second
- 5K words: ~2-5 seconds
- 10K words: ~5-10 seconds
- Bottleneck: Embedding generation (not storage)

**Implication for Progress Tracking:**
- Articles complete quickly (seconds)
- For bulk reindex, session-level progress is more meaningful
- Don't need chunk-level granularity for progress

---

## 3. Database Models

### 3.1 Article Model

**Location:** `src/article_mind_service/models/article.py`

**Relevant Fields:**

```python
class Article(Base):
    id: Mapped[int]
    session_id: Mapped[int]

    # Extraction tracking
    extraction_status: Mapped[str]  # "pending", "processing", "completed", "failed"
    content_text: Mapped[str | None]

    # Embedding tracking
    embedding_status: Mapped[str]  # "pending", "processing", "completed", "failed"
    chunk_count: Mapped[int | None]

    # Soft delete
    deleted_at: Mapped[datetime | None]
```

**Indexes:**
- `ix_articles_session_status`: (session_id, extraction_status)
- `ix_articles_session_deleted`: (session_id, deleted_at)

**Query Pattern for Reindex:**
```sql
SELECT * FROM articles
WHERE extraction_status = 'completed'
  AND embedding_status IN ('pending', 'failed')
  AND deleted_at IS NULL
ORDER BY session_id, id
```

### 3.2 Session Model

**Location:** `src/article_mind_service/models/session.py`

**Relevant Fields:**

```python
class ResearchSession(Base):
    id: Mapped[int]
    name: Mapped[str]
    status: Mapped[str]  # "draft", "active", "completed", "archived"
    deleted_at: Mapped[datetime | None]

    # Relationship
    articles: Mapped[list["Article"]] = relationship(
        "Article",
        back_populates="session",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
```

**No Built-in Progress Tracking:**
- No fields for tracking background operations
- No job/task tracking table

---

## 4. Background Task Patterns

### 4.1 Current Implementation: FastAPI BackgroundTasks

**Used in:**
- `routers/sessions.py:reindex_session()` - Embedding reindex
- `routers/articles.py` - Content extraction (likely)

**Pattern:**
```python
from fastapi import BackgroundTasks

@router.post("/endpoint")
async def endpoint(background_tasks: BackgroundTasks):
    # Queue task
    background_tasks.add_task(
        my_task_function,
        arg1=value1,
        arg2=value2,
    )

    # Return immediately
    return {"status": "queued"}
```

**Characteristics:**
- ✅ Simple, no external dependencies
- ✅ Built into FastAPI
- ❌ No task persistence
- ❌ No progress tracking
- ❌ No task cancellation
- ❌ Lost on server restart

### 4.2 No Task Queue Infrastructure

**Checked for:**
- Celery: Not found
- ARQ: Not found
- RQ: Not found
- Custom queue: Not found

**Dependencies (from pyproject.toml):**
- No task queue libraries
- No Redis/RabbitMQ clients
- Only async primitives: httpx, aiofiles

---

## 5. Streaming/Progress Tracking Patterns

### 5.1 Current State: No Streaming Infrastructure

**Checked for:**
- SSE (Server-Sent Events): ❌ Not found
- WebSocket: ❌ Not found
- Long polling: ❌ Not found
- `sse-starlette` dependency: ❌ Not in pyproject.toml

**Implication:**
Need to add streaming infrastructure from scratch.

### 5.2 SSE vs WebSocket vs Polling

**Comparison for Reindex Progress:**

| Feature | SSE | WebSocket | Polling |
|---------|-----|-----------|---------|
| **Complexity** | Low | High | Low |
| **Bi-directional** | No | Yes | No |
| **Browser Support** | Excellent | Excellent | Excellent |
| **Connection Overhead** | Low | Medium | High |
| **Best For** | One-way progress updates | Two-way interaction | Simple status checks |
| **Library Required** | `sse-starlette` | Built-in FastAPI | None |
| **Cancellation** | HTTP disconnect | WebSocket close | Manual flag |

**Recommendation for Reindex Progress: SSE**

**Rationale:**
1. **One-way communication:** Server → Client progress updates only
2. **Simple implementation:** Add `sse-starlette` dependency
3. **Automatic reconnection:** Browser handles reconnect on disconnect
4. **HTTP-based:** Works with existing infrastructure (reverse proxies, load balancers)
5. **No persistent state:** Progress tracking is ephemeral (in-memory)

---

## 6. Recommended Implementation Approach

### 6.1 Architecture Overview

```
Admin Reindex Endpoint
    ↓
[Start Background Task] → [Progress Tracker (in-memory)]
    ↓                           ↓
Return task_id             [SSE Endpoint] → Client receives progress
    ↓
[Reindex Worker]
    ↓
For each session:
    For each article:
        - Reindex article
        - Update progress tracker
        - Database status updates
```

### 6.2 Components to Build

#### Component 1: Progress Tracker (In-Memory)

**Purpose:** Track progress for multiple concurrent reindex operations

```python
# Pseudo-code structure
class ReindexProgress:
    task_id: str
    started_at: datetime
    total_articles: int
    processed_articles: int
    failed_articles: int
    current_session_id: int | None
    status: Literal["running", "completed", "failed"]
    errors: list[dict]

class ProgressTracker:
    _tasks: dict[str, ReindexProgress] = {}

    def create_task(self, task_id: str, total_articles: int)
    def update_progress(self, task_id: str, ...)
    def get_progress(self, task_id: str) -> ReindexProgress
    async def stream_progress(self, task_id: str) -> AsyncGenerator
```

#### Component 2: SSE Endpoint

**Endpoint:** `GET /api/v1/admin/reindex/{task_id}/progress`

**Response Format:**
```
event: progress
data: {"processed": 10, "total": 100, "percent": 10}

event: progress
data: {"processed": 20, "total": 100, "percent": 20}

event: complete
data: {"processed": 100, "total": 100, "errors": []}
```

**Implementation:**
```python
from sse_starlette.sse import EventSourceResponse

@router.get("/admin/reindex/{task_id}/progress")
async def stream_reindex_progress(task_id: str):
    async def event_generator():
        async for progress in progress_tracker.stream_progress(task_id):
            yield {
                "event": "progress",
                "data": progress.model_dump_json()
            }

    return EventSourceResponse(event_generator())
```

#### Component 3: Admin Reindex Endpoint

**Endpoint:** `POST /api/v1/admin/reindex`

**Request Schema:**
```python
class AdminReindexRequest(BaseModel):
    session_ids: list[int] | None = None  # None = all sessions
    force: bool = False  # Force reindex even if completed
```

**Response Schema:**
```python
class AdminReindexResponse(BaseModel):
    task_id: str  # UUID for progress tracking
    total_sessions: int
    total_articles: int
    progress_url: str  # SSE endpoint URL
```

**Flow:**
1. Validate request (check session_ids exist)
2. Count articles to reindex
3. Generate task_id (UUID)
4. Create progress tracker entry
5. Queue background task
6. Return task_id + progress_url immediately

#### Component 4: Reindex Worker

**Function:** `_reindex_all_articles(task_id, session_ids, progress_tracker, db)`

**Pseudo-code:**
```python
async def _reindex_all_articles(
    task_id: str,
    session_ids: list[int] | None,
    progress_tracker: ProgressTracker,
    db: AsyncSession,
):
    # Query articles to reindex
    articles = await db.execute(
        select(Article)
        .where(
            Article.extraction_status == "completed",
            Article.embedding_status.in_(["pending", "failed"]),
            Article.deleted_at.is_(None),
            # If session_ids provided, filter
        )
    )

    total = len(articles)
    progress_tracker.create_task(task_id, total)

    for idx, article in enumerate(articles):
        try:
            # Reindex article (reuse existing _reindex_article logic)
            await _reindex_article(article, pipeline, db)

            # Update progress
            progress_tracker.update_progress(
                task_id,
                processed=idx + 1,
                current_session_id=article.session_id,
            )
        except Exception as e:
            progress_tracker.record_error(task_id, article.id, str(e))

    progress_tracker.mark_complete(task_id)
```

### 6.3 Database Schema Changes

**Option A: No Schema Changes (Recommended)**
- Use in-memory progress tracking only
- Progress lost on server restart (acceptable for admin operation)
- Simpler implementation

**Option B: Add Job Tracking Table**
- Track reindex operations persistently
- Enables restart/resume of interrupted jobs
- More complex implementation

**Recommendation:** Start with Option A (in-memory), upgrade to Option B if needed.

---

## 7. File Modifications Required

### 7.1 New Files to Create

1. **`src/article_mind_service/progress_tracker.py`**
   - `ReindexProgress` Pydantic model
   - `ProgressTracker` singleton class
   - In-memory task tracking with asyncio.Queue for SSE

2. **`src/article_mind_service/routers/admin.py`**
   - `POST /api/v1/admin/reindex` - Start reindex
   - `GET /api/v1/admin/reindex/{task_id}/progress` - SSE progress stream
   - `GET /api/v1/admin/reindex/{task_id}` - Get final status

3. **`src/article_mind_service/schemas/admin.py`**
   - `AdminReindexRequest` - Request schema
   - `AdminReindexResponse` - Response with task_id
   - `ReindexProgressEvent` - SSE event schema

### 7.2 Files to Modify

1. **`pyproject.toml`**
   - Add dependency: `sse-starlette>=2.1.0`

2. **`src/article_mind_service/main.py`**
   - Register admin router: `app.include_router(admin.router)`

3. **`src/article_mind_service/routers/sessions.py`**
   - Extract `_reindex_article()` helper to shared module
   - Reuse in both session and admin reindex

### 7.3 Abstraction Opportunities

**Shared Reindex Logic:**
Move `_reindex_article()` to `src/article_mind_service/tasks/reindex.py`:

```python
# tasks/reindex.py
async def reindex_article(
    article: Article,
    pipeline: EmbeddingPipeline,
    db: AsyncSession,
    progress_callback: Callable | None = None,
) -> int:
    """Reindex single article with optional progress callback."""
    # Existing logic from sessions.py
    # Call progress_callback(article_id, status) if provided
```

**Benefits:**
- Reuse in session reindex endpoint
- Reuse in admin reindex endpoint
- Single source of truth for reindex logic
- Easier testing

---

## 8. Implementation Recommendations

### 8.1 Phase 1: SSE Infrastructure (Minimal Viable)

**Goal:** Add basic progress tracking with SSE

**Tasks:**
1. Add `sse-starlette` dependency
2. Create `ProgressTracker` with in-memory storage
3. Create SSE endpoint for streaming progress
4. Test with single-session reindex

**Validation:**
- Can stream progress events to client
- Progress updates in real-time
- Client can reconnect and resume stream

### 8.2 Phase 2: Admin Reindex Endpoint

**Goal:** Implement all-sessions reindex

**Tasks:**
1. Create admin router and schemas
2. Implement admin reindex endpoint
3. Integrate with progress tracker
4. Test with multiple sessions

**Validation:**
- Can reindex all sessions
- Progress tracked accurately
- Errors handled gracefully

### 8.3 Phase 3: Refinements

**Optional Enhancements:**
- Cancellation support (set flag, check in worker)
- Persistent job tracking (database table)
- Retry failed articles automatically
- Parallel session processing (asyncio.gather)
- Rate limiting for embedding API

---

## 9. Alternative Approaches Considered

### 9.1 Polling-Based Progress

**Approach:** Client polls `GET /admin/reindex/{task_id}` every N seconds

**Pros:**
- ✅ No SSE dependency
- ✅ Simpler server implementation
- ✅ Works with all HTTP clients

**Cons:**
- ❌ Inefficient (many wasted requests)
- ❌ Higher latency (up to poll interval)
- ❌ More server load

**Verdict:** Not recommended due to inefficiency

### 9.2 WebSocket-Based Progress

**Approach:** Full-duplex WebSocket connection for progress

**Pros:**
- ✅ Bi-directional communication
- ✅ Can send cancel commands
- ✅ Built into FastAPI

**Cons:**
- ❌ Overkill for one-way progress
- ❌ More complex implementation
- ❌ Connection management complexity

**Verdict:** Over-engineered for this use case

### 9.3 Task Queue (Celery/ARQ)

**Approach:** Add full task queue infrastructure

**Pros:**
- ✅ Persistent task tracking
- ✅ Retry mechanisms
- ✅ Distributed processing
- ✅ Built-in progress tracking

**Cons:**
- ❌ Requires Redis/RabbitMQ
- ❌ Significant infrastructure complexity
- ❌ Overkill for admin operation

**Verdict:** Too heavy for current needs

---

## 10. Testing Strategy

### 10.1 Unit Tests

**Files to Test:**
- `progress_tracker.py`: Progress state management
- `tasks/reindex.py`: Reindex logic isolation
- `schemas/admin.py`: Schema validation

### 10.2 Integration Tests

**Scenarios:**
1. **Single session reindex:** Verify existing behavior unchanged
2. **All sessions reindex:** Verify all articles processed
3. **Partial session reindex:** Filter by session_ids
4. **Progress streaming:** Verify SSE events emitted
5. **Error handling:** Verify failed articles tracked
6. **Concurrent reindex:** Verify multiple task_ids work

### 10.3 Manual Testing

**Test Cases:**
1. Start reindex → Open SSE stream in browser
2. Verify progress updates in real-time
3. Close stream → Reconnect → Verify resume
4. Cancel server → Restart → Verify graceful degradation

---

## 11. Performance Considerations

### 11.1 Bottlenecks

**Embedding Generation:**
- Current bottleneck (per pipeline docs)
- Sequential processing of articles
- API rate limits (OpenAI/Ollama)

**Database Queries:**
- Single query to fetch all articles (acceptable for <10K articles)
- Multiple status updates (one per article)

**SSE Connections:**
- One connection per client per task
- Lightweight (text-based events)

### 11.2 Scaling Recommendations

**For Large Deployments:**
1. **Parallel processing:** Use `asyncio.gather()` for concurrent article reindex
2. **Rate limiting:** Respect embedding provider rate limits
3. **Progress aggregation:** Update every N articles instead of every article
4. **Connection pooling:** Ensure adequate database connection pool

---

## 12. API Contract Example

### 12.1 Request/Response

**Start Reindex:**
```http
POST /api/v1/admin/reindex
Content-Type: application/json

{
  "session_ids": [1, 2, 3],  // Optional, null = all
  "force": false
}

Response 202 Accepted:
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_sessions": 3,
  "total_articles": 47,
  "progress_url": "/api/v1/admin/reindex/550e8400-e29b-41d4-a716-446655440000/progress"
}
```

**Stream Progress (SSE):**
```http
GET /api/v1/admin/reindex/550e8400-e29b-41d4-a716-446655440000/progress
Accept: text/event-stream

Response:
event: progress
data: {"task_id":"550e...","processed":10,"total":47,"percent":21,"current_session_id":1}

event: progress
data: {"task_id":"550e...","processed":20,"total":47,"percent":42,"current_session_id":2}

event: complete
data: {"task_id":"550e...","processed":47,"total":47,"failed":0,"errors":[]}
```

**Get Final Status:**
```http
GET /api/v1/admin/reindex/550e8400-e29b-41d4-a716-446655440000

Response 200:
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "total_articles": 47,
  "processed_articles": 47,
  "failed_articles": 0,
  "started_at": "2026-01-21T10:00:00Z",
  "completed_at": "2026-01-21T10:05:32Z",
  "errors": []
}
```

---

## 13. Key Takeaways

### 13.1 What Exists

✅ **Single-session reindex endpoint** - Working, tested pattern
✅ **Embedding pipeline** - Robust, handles errors gracefully
✅ **Background task pattern** - FastAPI BackgroundTasks in use
✅ **Database status tracking** - embedding_status field per article

### 13.2 What's Missing

❌ **Progress tracking infrastructure** - No in-memory or persistent progress
❌ **Streaming endpoints** - No SSE or WebSocket support
❌ **Admin endpoints** - No admin-specific routes
❌ **Multi-session reindex** - Only single session supported

### 13.3 Recommended Path Forward

1. **Add SSE infrastructure** - sse-starlette + ProgressTracker
2. **Create admin router** - New endpoint for all-sessions reindex
3. **Reuse existing logic** - Extract and share reindex logic
4. **In-memory progress** - Start simple, upgrade to persistent if needed
5. **Test thoroughly** - Integration tests for concurrent reindex

### 13.4 Estimated Effort

- **SSE Infrastructure:** 4-6 hours
- **Admin Reindex Endpoint:** 3-4 hours
- **Integration and Testing:** 3-4 hours
- **Documentation:** 2 hours
- **Total:** 12-16 hours

---

## 14. References

### 14.1 Codebase Files Analyzed

- `src/article_mind_service/routers/sessions.py` - Existing reindex endpoint
- `src/article_mind_service/embeddings/pipeline.py` - Embedding orchestration
- `src/article_mind_service/models/article.py` - Article database model
- `src/article_mind_service/models/session.py` - Session database model
- `src/article_mind_service/schemas/session.py` - Session API schemas
- `src/article_mind_service/tasks/extraction.py` - Background task pattern
- `pyproject.toml` - Current dependencies

### 14.2 External Resources

- **FastAPI BackgroundTasks:** https://fastapi.tiangolo.com/tutorial/background-tasks/
- **sse-starlette:** https://github.com/sysid/sse-starlette
- **Server-Sent Events Spec:** https://html.spec.whatwg.org/multipage/server-sent-events.html
- **FastAPI Streaming:** https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse

---

## Appendix A: Sample Implementation Code

### A.1 Progress Tracker

```python
# src/article_mind_service/progress_tracker.py
import asyncio
from datetime import datetime
from typing import AsyncGenerator
from uuid import uuid4

from pydantic import BaseModel


class ReindexProgress(BaseModel):
    """Progress state for a reindex operation."""

    task_id: str
    status: str  # "running", "completed", "failed"
    total_articles: int
    processed_articles: int = 0
    failed_articles: int = 0
    current_session_id: int | None = None
    started_at: datetime
    completed_at: datetime | None = None
    errors: list[dict] = []


class ProgressTracker:
    """In-memory progress tracker for reindex operations."""

    def __init__(self):
        self._tasks: dict[str, ReindexProgress] = {}
        self._queues: dict[str, asyncio.Queue] = {}

    def create_task(self, total_articles: int) -> str:
        """Create new task and return task_id."""
        task_id = str(uuid4())
        self._tasks[task_id] = ReindexProgress(
            task_id=task_id,
            status="running",
            total_articles=total_articles,
            started_at=datetime.utcnow(),
        )
        self._queues[task_id] = asyncio.Queue()
        return task_id

    async def update_progress(
        self,
        task_id: str,
        processed: int | None = None,
        current_session_id: int | None = None,
    ):
        """Update progress and notify listeners."""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        if processed is not None:
            task.processed_articles = processed
        if current_session_id is not None:
            task.current_session_id = current_session_id

        # Notify SSE listeners
        await self._queues[task_id].put(task.model_copy())

    async def stream_progress(self, task_id: str) -> AsyncGenerator[ReindexProgress, None]:
        """Stream progress updates via async generator."""
        if task_id not in self._queues:
            return

        queue = self._queues[task_id]

        # Send current state immediately
        if task_id in self._tasks:
            yield self._tasks[task_id]

        # Stream updates
        while True:
            progress = await queue.get()
            yield progress

            # Stop streaming when complete
            if progress.status in ("completed", "failed"):
                break


# Singleton instance
progress_tracker = ProgressTracker()
```

### A.2 SSE Endpoint

```python
# src/article_mind_service/routers/admin.py
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from article_mind_service.progress_tracker import progress_tracker
from article_mind_service.schemas.admin import (
    AdminReindexRequest,
    AdminReindexResponse,
)

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
)


@router.get("/reindex/{task_id}/progress")
async def stream_reindex_progress(task_id: str):
    """Stream reindex progress via SSE."""

    async def event_generator():
        async for progress in progress_tracker.stream_progress(task_id):
            event_name = "progress" if progress.status == "running" else "complete"
            yield {
                "event": event_name,
                "data": progress.model_dump_json(),
            }

    return EventSourceResponse(event_generator())


@router.post("/reindex", response_model=AdminReindexResponse)
async def start_admin_reindex(
    request: AdminReindexRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Start reindexing all or selected sessions."""

    # Query articles to reindex
    query = select(Article).where(
        Article.extraction_status == "completed",
        Article.embedding_status.in_(["pending", "failed"]),
        Article.deleted_at.is_(None),
    )

    if request.session_ids:
        query = query.where(Article.session_id.in_(request.session_ids))

    result = await db.execute(query)
    articles = result.scalars().all()

    # Create progress tracker
    task_id = progress_tracker.create_task(total_articles=len(articles))

    # Queue background task
    background_tasks.add_task(
        _reindex_all_articles,
        task_id=task_id,
        articles=articles,
        db=db,
    )

    return AdminReindexResponse(
        task_id=task_id,
        total_articles=len(articles),
        progress_url=f"/api/v1/admin/reindex/{task_id}/progress",
    )
```

---

**End of Research Document**
