# ChromaDB Collection Missing - Root Cause Analysis

**Date:** 2026-01-20
**Session:** 179
**Error:** `chromadb.errors.NotFoundError: Collection [article_chunks] does not exist`
**Status:** ✅ Root cause identified, solution provided

---

## Executive Summary

**Problem:** RAG pipeline fails when searching session 179 because ChromaDB collection `article_chunks` doesn't exist.

**Root Cause:** Articles are extracted but never embedded/indexed. The embedding pipeline exists but is **not triggered** after content extraction completes.

**Impact:**
- Chat/Q&A feature completely broken (cannot retrieve context)
- Search API non-functional
- Articles in session 179 have `embedding_status: pending` despite extraction being completed

**Solution:** Trigger embedding pipeline after extraction completes OR implement manual indexing endpoint.

---

## Architecture Overview

### Two-Phase Processing Pipeline

The system has **two separate asynchronous pipelines**:

#### Phase 1: Content Extraction (✅ Working)
```
POST /api/v1/sessions/{session_id}/articles/url
  ↓
Create Article (extraction_status: pending)
  ↓
Background Task: extract_article_content()
  ↓
Extract content via trafilatura/playwright
  ↓
Update Article (extraction_status: completed, content_text populated)
```

**File:** `src/article_mind_service/tasks/extraction.py`

#### Phase 2: Embedding/Indexing (❌ Not Triggered)
```
SUPPOSED TO HAPPEN (but doesn't):
  ↓
Trigger EmbeddingPipeline.process_article()
  ↓
Chunk text (512 chars, 50 overlap)
  ↓
Generate embeddings (OpenAI or Ollama)
  ↓
Store in ChromaDB collection: session_{session_id}
  ↓
Update Article (embedding_status: completed, chunk_count set)
```

**File:** `src/article_mind_service/embeddings/pipeline.py`

---

## Current State Investigation

### Database Verification (Session 179)

```sql
SELECT id, session_id, type, title, extraction_status, embedding_status, chunk_count
FROM articles
WHERE session_id = 179;
```

**Result:**
```
 id | session_id | type | title | extraction_status | embedding_status | chunk_count
----+------------+------+-------+-------------------+------------------+-------------
 39 |        179 | url  |       | completed         | pending          |
```

**Analysis:**
- ✅ Article 39 exists in session 179
- ✅ Content extraction completed successfully
- ❌ `embedding_status: pending` (never triggered)
- ❌ `chunk_count: NULL` (no chunks created)

### ChromaDB Collection Naming

**Two Different Collection Strategies Found:**

1. **EmbeddingPipeline** (used during indexing):
   - Collection name: `session_{session_id}`
   - Example: `session_179`
   - **Purpose:** Session-isolated embeddings, easy cleanup

2. **DenseSearch** (used during querying):
   - Collection name: `article_chunks` (from config)
   - **Purpose:** Unified collection for all articles
   - **Config:** `settings.chroma_collection_name = "article_chunks"`

**🚨 CRITICAL ISSUE:** Collection naming **mismatch** between indexing and search!

```python
# Indexing uses:
collection_name = f"session_{session_id}"  # embeddings/chromadb_store.py:97

# Searching uses:
collection_name = settings.chroma_collection_name  # "article_chunks"  # search/dense_search.py:69
```

---

## Root Cause Analysis

### Issue #1: Embedding Pipeline Never Triggered

**File:** `src/article_mind_service/tasks/extraction.py`

The `extract_article_content()` function:
- ✅ Extracts content successfully
- ✅ Updates `extraction_status: completed`
- ❌ **Never calls** `EmbeddingPipeline.process_article()`

**Evidence:**
```python
# Line 72-92 in extraction.py
article.extraction_status = "completed"
article.content_text = result.content
# ... other metadata updates ...

# NO CALL TO EMBEDDING PIPELINE FOUND
```

**Expected Behavior:**
```python
# After line 92, should trigger:
from article_mind_service.embeddings import get_embedding_pipeline

pipeline = get_embedding_pipeline()
await pipeline.process_article(
    article_id=article.id,
    session_id=str(article.session_id),
    text=result.content,
    source_url=url,
    db=db
)
```

### Issue #2: Collection Naming Inconsistency

**EmbeddingPipeline creates:** `session_{session_id}`
**DenseSearch expects:** `article_chunks`

**Files:**
- `embeddings/chromadb_store.py:97` → `collection_name = f"session_{session_id}"`
- `search/dense_search.py:69` → `collection_name = settings.chroma_collection_name` (defaults to `article_chunks`)

**Impact:** Even if embeddings were created, search would fail because it's looking in the wrong collection.

### Issue #3: No Indexing Trigger Mechanism

**Search paths for indexing trigger:**

1. ❌ **Not in extraction task** (`tasks/extraction.py`)
2. ❌ **Not in article router** (`routers/articles.py`)
3. ❌ **Not in application startup** (`main.py`)
4. ❌ **No background worker** (no Celery, no ARQ, no scheduler)
5. ❌ **No manual indexing endpoint** (no `POST /api/v1/sessions/{id}/reindex`)

**Conclusion:** The system has **no mechanism** to trigger embedding pipeline.

---

## Technical Deep Dive

### EmbeddingPipeline Implementation

**File:** `src/article_mind_service/embeddings/pipeline.py`

**Design:**
- Chunks text into 512-character segments with 50-character overlap
- Generates embeddings using configured provider (OpenAI or Ollama)
- Stores in ChromaDB with metadata: `article_id`, `chunk_index`, `source_url`
- Updates database: `embedding_status`, `chunk_count`

**Key Method:**
```python
async def process_article(
    article_id: int,
    session_id: str,
    text: str,
    source_url: str,
    db: AsyncSession,
) -> int:
    """Process a single article through the embedding pipeline."""
```

**Collection Creation:**
```python
# Line 120-123
collection = self.store.get_or_create_collection(
    session_id=session_id,
    dimensions=self.provider.dimensions,
)
```

**Chunk ID Format:**
```python
# Line 138
ids = [f"article_{article_id}_chunk_{c['chunk_index']}" for c in batch]
```

### ChromaDB Store Implementation

**File:** `src/article_mind_service/embeddings/chromadb_store.py`

**Collection Strategy:**
- One collection per session: `session_{session_id}`
- Benefits: Session isolation, easy cleanup, multi-tenant safe
- Metadata: `article_id`, `chunk_index`, `source_url`

**Storage:**
- Persistent: `./data/chromadb` (DuckDB + Parquet backend)
- HNSW index for O(log n) vector search
- Suitable for <200K vectors

### DenseSearch Implementation

**File:** `src/article_mind_service/search/dense_search.py`

**Query Method:**
```python
def search(
    session_id: int,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[DenseSearchResult]:
    collection = self.client.get_collection(self.collection_name)  # ❌ "article_chunks"
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"session_id": session_id},  # ❌ Filter by session_id metadata
    )
```

**🚨 PROBLEM:** Tries to get collection `article_chunks` which doesn't exist.

---

## Failure Scenario Walkthrough

### User Action: Add Article to Session 179

```bash
POST /api/v1/sessions/179/articles/url
{
  "url": "https://example.com/article"
}
```

**Backend Flow:**

1. ✅ **Article Created** (id: 39, extraction_status: pending)
2. ✅ **Background Extraction Started** (`extract_article_content()`)
3. ✅ **Content Extracted** (trafilatura successful)
4. ✅ **Article Updated** (extraction_status: completed, content_text populated)
5. ❌ **Embedding Pipeline NOT Triggered** (no call to `process_article()`)

**Database State:**
```
article_id: 39
extraction_status: completed
embedding_status: pending  ← STUCK HERE
chunk_count: NULL
```

### User Action: Ask Question in Session 179

```bash
POST /api/v1/sessions/179/chat
{
  "message": "What is the main topic?"
}
```

**Backend Flow:**

1. ✅ **RAG Pipeline Started** (`RAGPipeline.query()`)
2. ✅ **Embedding Generated** (query embedding via OpenAI/Ollama)
3. ✅ **HybridSearch Initialized** (calls DenseSearch internally)
4. ❌ **DenseSearch Fails:**
   ```python
   collection = self.client.get_collection("article_chunks")
   # chromadb.errors.NotFoundError: Collection [article_chunks] does not exist
   ```
5. ❌ **Chat Request Fails** (500 error, no response)

**Error Log:**
```
chromadb.errors.NotFoundError: Collection [article_chunks] does not exist
```

---

## Solutions

### Option 1: Auto-Trigger After Extraction (Recommended)

**Modify:** `src/article_mind_service/tasks/extraction.py`

**Add after line 92:**
```python
# After successful extraction, trigger embedding pipeline
if article.extraction_status == "completed" and article.content_text:
    try:
        from article_mind_service.embeddings import get_embedding_pipeline

        logger.info(f"Starting embedding pipeline for article {article_id}")

        pipeline = get_embedding_pipeline()
        chunk_count = await pipeline.process_article(
            article_id=article.id,
            session_id=str(article.session_id),
            text=article.content_text,
            source_url=url,
            db=db
        )

        logger.info(
            f"Embedding completed for article {article_id}: "
            f"{chunk_count} chunks indexed"
        )
    except Exception as e:
        logger.error(f"Embedding failed for article {article_id}: {e}")
        # Don't fail extraction if embedding fails
```

**Pros:**
- ✅ Automatic, no manual intervention
- ✅ Embeddings created immediately after extraction
- ✅ Users can search/chat as soon as extraction completes

**Cons:**
- ❌ Longer processing time (extraction + embedding)
- ❌ Tighter coupling between extraction and embedding

### Option 2: Manual Indexing Endpoint

**Create:** `POST /api/v1/sessions/{session_id}/reindex`

**Implementation:**
```python
@router.post("/{session_id}/reindex")
async def reindex_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Manually trigger embedding for all completed articles in session."""

    # Get all completed articles with pending embeddings
    result = await db.execute(
        select(Article).where(
            Article.session_id == session_id,
            Article.extraction_status == "completed",
            Article.embedding_status == "pending",
            Article.deleted_at.is_(None),
        )
    )
    articles = result.scalars().all()

    pipeline = get_embedding_pipeline()
    indexed_count = 0

    for article in articles:
        if not article.content_text:
            continue

        await pipeline.process_article(
            article_id=article.id,
            session_id=str(article.session_id),
            text=article.content_text,
            source_url=article.original_url or "",
            db=db
        )
        indexed_count += 1

    return {
        "session_id": session_id,
        "articles_indexed": indexed_count,
        "total_articles": len(articles)
    }
```

**Pros:**
- ✅ Decouples extraction from embedding
- ✅ Can reindex after configuration changes
- ✅ Allows batch reindexing of multiple articles

**Cons:**
- ❌ Requires manual trigger
- ❌ Users must wait for manual reindex before chat works

### Option 3: Fix Collection Naming Consistency

**Choose ONE strategy:**

#### Strategy A: Use Session-Isolated Collections (Recommended)

**Change:** `src/article_mind_service/search/dense_search.py`

```python
# Line 62-69
def __init__(self, collection_name: str | None = None) -> None:
    self.client = chromadb.PersistentClient(path=str(settings.chroma_persist_directory))
    # Remove collection_name parameter, use session-based naming in search()

# Line 71-76
def search(
    session_id: int,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[DenseSearchResult]:
    # Use session-based collection name
    collection_name = f"session_{session_id}"
    try:
        collection = self.client.get_collection(collection_name)
    except ValueError:
        return []  # Collection doesn't exist yet
```

**Pros:**
- ✅ Matches indexing strategy
- ✅ Session isolation maintained
- ✅ Easy cleanup (delete collection when session deleted)

**Cons:**
- ❌ More collections to manage
- ❌ Cannot search across sessions easily

#### Strategy B: Use Unified Collection

**Change:** `src/article_mind_service/embeddings/chromadb_store.py`

```python
# Line 75-106
def get_or_create_collection(
    self,
    session_id: str,
    dimensions: int,
) -> chromadb.Collection:
    # Use unified collection name
    collection_name = "article_chunks"

    return self.client.get_or_create_collection(
        name=collection_name,
        metadata={
            "dimensions": dimensions,
        },
    )
```

**Update chunk metadata to include session_id:**
```python
# Line 138-146 in pipeline.py
metadatas = [
    {
        "article_id": c["article_id"],
        "chunk_index": c["chunk_index"],
        "source_url": c["source_url"],
        "session_id": session_id,  # ADD THIS
    }
    for c in batch
]
```

**Pros:**
- ✅ Single collection to manage
- ✅ Cross-session search possible
- ✅ Matches current search implementation

**Cons:**
- ❌ Harder to cleanup (cannot delete session collection)
- ❌ Must filter by session_id metadata (slower)

---

## Immediate Fix for Session 179

### Step 1: Verify Article Content

```bash
cd /export/workspace/article-mind/article-mind-service

PGPASSWORD=article_mind psql -U article_mind -d article_mind -h localhost -c "
SELECT id, session_id, LENGTH(content_text) as content_length, extraction_status, embedding_status
FROM articles
WHERE session_id = 179 AND id = 39;
"
```

**Expected:** `content_length > 0` and `extraction_status = completed`

### Step 2: Manual Indexing Script

**Create:** `scripts/index_article.py`

```python
"""Manual indexing script for articles."""
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from article_mind_service.config import settings
from article_mind_service.embeddings import get_embedding_pipeline
from article_mind_service.models import Article

async def index_article(article_id: int):
    """Index a single article."""
    # Create async engine
    engine = create_async_engine(
        settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
        echo=False,
    )

    # Create session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        # Get article
        article = await session.get(Article, article_id)
        if not article:
            print(f"❌ Article {article_id} not found")
            return

        if not article.content_text:
            print(f"❌ Article {article_id} has no content")
            return

        print(f"📄 Article {article_id}:")
        print(f"  Session: {article.session_id}")
        print(f"  Content length: {len(article.content_text)} chars")
        print(f"  Extraction status: {article.extraction_status}")
        print(f"  Embedding status: {article.embedding_status}")

        # Run embedding pipeline
        pipeline = get_embedding_pipeline()

        print(f"\n🔄 Starting embedding pipeline...")
        chunk_count = await pipeline.process_article(
            article_id=article.id,
            session_id=str(article.session_id),
            text=article.content_text,
            source_url=article.original_url or "",
            db=session,
        )

        print(f"✅ Indexed {chunk_count} chunks")
        print(f"✅ Collection: session_{article.session_id}")

    await engine.dispose()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python scripts/index_article.py <article_id>")
        sys.exit(1)

    article_id = int(sys.argv[1])
    asyncio.run(index_article(article_id))
```

### Step 3: Run Indexing

```bash
cd /export/workspace/article-mind/article-mind-service

# Index article 39
uv run python scripts/index_article.py 39
```

**Expected Output:**
```
📄 Article 39:
  Session: 179
  Content length: 5234 chars
  Extraction status: completed
  Embedding status: pending

🔄 Starting embedding pipeline...
✅ Indexed 12 chunks
✅ Collection: session_179
```

### Step 4: Verify ChromaDB Collection

**Check collection exists:**

```python
# In Python shell:
import chromadb

client = chromadb.PersistentClient(path="./data/chromadb")
collections = client.list_collections()
print([c.name for c in collections])
# Expected: ['session_179']

collection = client.get_collection("session_179")
print(f"Chunks in session_179: {collection.count()}")
# Expected: 12 (or number of chunks created)
```

### Step 5: Fix DenseSearch Collection Name

**Temporary Fix (Quick):**

```bash
# Set environment variable
export CHROMA_COLLECTION_NAME="session_179"

# Or in .env file:
echo "CHROMA_COLLECTION_NAME=session_179" >> .env
```

**Permanent Fix (Better):**

Edit `src/article_mind_service/search/dense_search.py`:

```python
# Line 71-100
def search(
    session_id: int,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[DenseSearchResult]:
    # Use session-based collection naming
    collection_name = f"session_{session_id}"

    try:
        collection = self.client.get_collection(collection_name)
    except ValueError:
        # Collection doesn't exist yet
        logger.warning(f"Collection {collection_name} not found")
        return []

    # ... rest of search logic
```

### Step 6: Test Chat/Search

```bash
# Test search endpoint
curl -X POST http://localhost:13010/api/v1/sessions/179/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "main topic",
    "top_k": 5,
    "search_mode": "hybrid"
  }'

# Test chat endpoint
curl -X POST http://localhost:13010/api/v1/sessions/179/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the main topic of the article?"
  }'
```

**Expected:** 200 OK with search results / chat response

---

## Recommended Implementation Plan

### Phase 1: Immediate Fix (Session 179)

1. ✅ Run manual indexing script for article 39
2. ✅ Fix DenseSearch collection naming to use `session_{session_id}`
3. ✅ Test search/chat functionality

### Phase 2: Permanent Solution

1. **Add auto-triggering** (Option 1):
   - Modify `tasks/extraction.py` to trigger embedding after extraction
   - Add error handling (don't fail extraction if embedding fails)
   - Add logging for embedding status

2. **Fix collection naming** (Option 3, Strategy A):
   - Update DenseSearch to use `session_{session_id}` naming
   - Keep session-isolated collections for better isolation
   - Update tests to reflect new naming

3. **Add manual reindex endpoint** (Option 2):
   - Create `POST /api/v1/sessions/{id}/reindex` endpoint
   - Allow reindexing all articles or specific articles
   - Return status of indexing operation

4. **Add monitoring**:
   - Log embedding pipeline status
   - Track `embedding_status` distribution in metrics
   - Alert on stuck embeddings (pending > 5 minutes)

### Phase 3: Testing

1. **Integration Tests:**
   - Test full flow: add article → extract → embed → search
   - Test collection creation and cleanup
   - Test error scenarios (extraction success, embedding fail)

2. **Manual Testing:**
   - Add article to new session
   - Verify automatic embedding
   - Test chat/search immediately after extraction
   - Verify no manual intervention needed

---

## Configuration Reference

### Environment Variables

```bash
# Embedding Provider
EMBEDDING_PROVIDER=openai  # or "ollama"
OPENAI_API_KEY=sk-...

# ChromaDB
CHROMADB_PATH=./data/chromadb
CHROMA_COLLECTION_NAME=article_chunks  # ❌ WILL CHANGE TO session-based naming

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

### File Locations

| Component | File Path |
|-----------|-----------|
| Extraction Task | `src/article_mind_service/tasks/extraction.py` |
| Embedding Pipeline | `src/article_mind_service/embeddings/pipeline.py` |
| ChromaDB Store | `src/article_mind_service/embeddings/chromadb_store.py` |
| Dense Search | `src/article_mind_service/search/dense_search.py` |
| RAG Pipeline | `src/article_mind_service/chat/rag_pipeline.py` |
| Article Model | `src/article_mind_service/models/article.py` |
| Config | `src/article_mind_service/config.py` |

---

## Testing Checklist

After implementing fixes:

- [ ] Article extraction completes successfully
- [ ] `embedding_status` transitions to `processing` then `completed`
- [ ] `chunk_count` is populated (not NULL)
- [ ] ChromaDB collection `session_{session_id}` exists
- [ ] Collection contains expected number of chunks
- [ ] Search endpoint returns results
- [ ] Chat endpoint generates answers with citations
- [ ] Error handling works (extraction success, embedding fail)
- [ ] Reindex endpoint works for stuck articles
- [ ] Collection cleanup works when session deleted

---

## Appendix: Code References

### ChromaDBStore Collection Creation

**File:** `src/article_mind_service/embeddings/chromadb_store.py`

```python
def get_or_create_collection(
    self,
    session_id: str,
    dimensions: int,
) -> chromadb.Collection:
    collection_name = f"session_{session_id}"  # Line 97

    return self.client.get_or_create_collection(
        name=collection_name,
        metadata={
            "dimensions": dimensions,
            "session_id": session_id,
        },
    )
```

### DenseSearch Collection Query

**File:** `src/article_mind_service/search/dense_search.py`

```python
def __init__(self, collection_name: str | None = None) -> None:
    self.client = chromadb.PersistentClient(path=str(settings.chroma_persist_directory))
    self.collection_name = collection_name or settings.chroma_collection_name  # Line 69

def search(
    session_id: int,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[DenseSearchResult]:
    try:
        collection = self.client.get_collection(self.collection_name)  # Line 97
        # ❌ Uses "article_chunks" but should use f"session_{session_id}"
```

### EmbeddingPipeline Process Article

**File:** `src/article_mind_service/embeddings/pipeline.py`

```python
async def process_article(
    self,
    article_id: int,
    session_id: str,
    text: str,
    source_url: str,
    db: AsyncSession,
) -> int:
    # Step 2: Get or create collection
    collection = self.store.get_or_create_collection(  # Line 120
        session_id=session_id,
        dimensions=self.provider.dimensions,
    )

    # Step 3: Process in batches
    for batch_start in range(0, total_chunks, self.BATCH_SIZE):
        # ... chunking and embedding logic

        # Store in ChromaDB
        self.store.add_embeddings(  # Line 149
            collection=collection,
            embeddings=embeddings,
            texts=batch_texts,
            metadatas=metadatas,
            ids=ids,
        )
```

---

## Questions for Clarification

1. **Collection Strategy:** Should we use session-isolated collections (`session_{id}`) or unified collection (`article_chunks`)?
   - **Recommendation:** Session-isolated for better isolation and cleanup

2. **Triggering Strategy:** Auto-trigger after extraction or manual endpoint?
   - **Recommendation:** Auto-trigger for better UX + manual endpoint for recovery

3. **Error Handling:** What should happen if embedding fails but extraction succeeds?
   - **Recommendation:** Keep extraction success, mark embedding as failed, allow retry

4. **Backward Compatibility:** Should we support both collection naming strategies during migration?
   - **Recommendation:** No, clean cutover to session-isolated strategy

---

**End of Analysis**
