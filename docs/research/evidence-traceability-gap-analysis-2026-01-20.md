# Evidence and Traceability Gap Analysis: Article Mind Chat/Query Flow

**Research Date:** 2026-01-20
**Researcher:** Research Agent
**Focus:** Current state of evidence, traceability, and logging in article scraping and RAG pipeline

---

## Executive Summary

This investigation analyzed the article-mind-service codebase to understand how evidence and traceability are currently implemented in the article scraping and chat query flow. The analysis reveals a **significant gap between internal data availability and user-facing evidence**, with limited logging for debugging and no mechanism for users to verify what content was used to answer their questions.

### Key Findings

1. **Article content is extracted and stored but never shown to users** - No endpoint exposes scraped content for verification
2. **RAG pipeline has source citations but lacks content traceability** - Users see which articles were cited but not the actual text chunks used
3. **Minimal logging in critical paths** - Only basic print statements in chat flow, no structured logging of retrieval context
4. **Search metadata is rich internally but not exposed** - Hybrid search rankings, scores, and chunks are not returned to users
5. **No audit trail for RAG decisions** - Cannot trace why specific chunks were retrieved or how LLM used them

---

## 1. Article Scraping & Content Flow

### How Articles are Submitted

**URL Submission Flow** (`routers/articles.py:171-209`):

```python
@router.post("/url", response_model=ArticleResponse, status_code=201)
async def add_url_article(session_id, data: AddUrlRequest, background_tasks, db):
    # 1. Verify session exists
    await get_session_or_404(session_id, db)

    # 2. Create article record with status="pending"
    article = Article(
        session_id=session_id,
        type="url",
        original_url=str(data.url),
        title=data.title,
        extraction_status="pending"
    )

    # 3. Trigger background extraction
    background_tasks.add_task(extract_article_content, article.id, db)

    # 4. Return article metadata (NO CONTENT YET)
    return article_to_response(article)
```

**Key Observation:** Users receive an immediate response with `extraction_status="pending"`, but no mechanism to monitor extraction progress or retrieve results later.

### Content Extraction Pipeline

**Extraction Task** (`tasks/extraction.py:21-112`):

```python
async def extract_article_content(article_id, db, config=None):
    # Updates status to "processing"
    article.extraction_status = "processing"

    # Runs extraction using Trafilatura/Playwright
    pipeline = ExtractionPipeline(pipeline_config)
    result = await pipeline.extract(url)

    # Stores EXTENSIVE metadata in database:
    article.content_text = result.content        # Full extracted text
    article.content_hash = _compute_hash(result.content)
    article.title = result.title
    article.author = result.author
    article.published_date = result.published_date
    article.language = result.language
    article.word_count = result.word_count
    article.reading_time_minutes = _estimate_reading_time(result.word_count)
    article.extraction_metadata = result.metadata  # Keywords, tags, etc.
    article.extraction_method = result.extraction_method
    article.extracted_at = datetime.utcnow()
```

**Database Schema** (`models/article.py:24-265`):

The `Article` model stores comprehensive content and metadata:

- `content_text: Text` - Full extracted article text
- `content_hash: String(64)` - SHA-256 hash for deduplication
- `title, author, published_date, language`
- `word_count, reading_time_minutes`
- `extraction_metadata: JSON` - Keywords, tags, warnings
- `extraction_method: String` - Which tool was used (trafilatura, playwright, etc.)
- `extraction_status: Enum` - pending/processing/completed/failed
- `extraction_error: Text` - Error message if failed

### Where Content is Stored

1. **Database:** Full `content_text` stored in PostgreSQL `articles.content_text` (TEXT column)
2. **ChromaDB:** Content chunked and embedded for vector search
3. **BM25 Index:** Content tokenized for sparse keyword search
4. **File System:** Original uploaded files in `data/uploads/{session_id}/{article_id}/`

### ⚠️ **GAP #1: No User-Facing Content Retrieval**

**Current Endpoint** (`routers/articles.py:441-489`):

```python
@router.get("/{article_id}/content", response_model=ArticleContentResponse)
async def get_article_content(session_id, article_id, db):
    article = await get_article_or_404(session_id, article_id, db)

    # Requires extraction to be completed
    if article.extraction_status != "completed":
        raise HTTPException(400, detail=f"Content not available. Status: {article.extraction_status}")

    # Returns FULL content text
    return ArticleContentResponse(
        id=article.id,
        title=article.title,
        content_text=article.content_text,  # Entire article
        extraction_status=article.extraction_status
    )
```

**Issues:**

1. **Frontend may not be calling this endpoint** - Need to verify if UI displays article content
2. **No preview or summary** - Returns entire text, which could be massive
3. **No metadata shown** - Users don't see extraction method, word count, reading time, etc.
4. **No extraction warnings** - If trafilatura had issues, users aren't informed

**Evidence Gap:** Users cannot easily verify:
- "What content was actually extracted from this article?"
- "Is the extraction complete and accurate?"
- "Were there any warnings during extraction?"

---

## 2. Chat Query & RAG Pipeline

### Chat Query Flow

**Chat Endpoint** (`routers/chat.py:24-115`):

```python
@router.post("", response_model=ChatResponse)
async def send_chat_message(session_id, request: ChatRequest, db):
    # 1. Save user message to database
    user_message = ChatMessage(session_id=session_id, role="user", content=request.message)
    db.add(user_message)

    # 2. Execute RAG pipeline
    pipeline = RAGPipeline()
    rag_response = await pipeline.query(session_id=session_id, question=request.message, db=db)

    # 3. Save assistant response with sources
    assistant_message = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=rag_response.content,
        sources=rag_response.sources,  # JSON array of citations
        llm_provider=rag_response.llm_provider,
        llm_model=rag_response.llm_model,
        tokens_used=rag_response.tokens_used
    )
    db.add(assistant_message)

    # 4. Return response to user
    return ChatResponse(
        message_id=assistant_message.id,
        content=assistant_message.content,
        sources=[ChatSource(**s) for s in rag_response.sources],
        llm_provider=rag_response.llm_provider,
        llm_model=rag_response.llm_model,
        tokens_used=rag_response.tokens_used
    )
```

### RAG Pipeline Internals

**Pipeline Orchestration** (`chat/rag_pipeline.py:78-132`):

```python
async def query(self, session_id, question, db, search_client=None):
    # Step 1: Retrieve relevant chunks from search
    chunks = await self._retrieve_chunks(
        session_id=session_id,
        query=question,
        limit=self.max_context_chunks  # Default: 5 chunks
    )

    # Step 2: Format context with metadata
    context_str, source_metadata = format_context_with_metadata(chunks)

    # Step 3: Generate answer via LLM
    system_prompt = build_system_prompt(has_context=len(chunks) > 0)
    llm_response = await self.llm_provider.generate(
        system_prompt=system_prompt,
        user_message=question,
        context_chunks=[context_str],  # Formatted context
        max_tokens=settings.llm_max_tokens,
        temperature=0.3
    )

    # Step 4: Extract cited sources only
    cited_sources = self._extract_cited_sources(
        content=llm_response.content,
        source_metadata=source_metadata
    )

    return RAGResponse(
        content=llm_response.content,
        sources=cited_sources,  # Only sources that were cited [1], [2], etc.
        chunks_retrieved=len(chunks)
    )
```

### Context Formatting

**Prompt Engineering** (`chat/prompts.py:57-106`):

```python
def format_context_with_metadata(chunks: list[dict]) -> tuple[str, list[dict]]:
    """Format chunks with numbered citations [1], [2], etc."""

    formatted_lines = []
    source_metadata = []

    for i, chunk in enumerate(chunks, start=1):
        content = chunk.get("content", "")
        title = chunk.get("title", "Unknown Source")
        url = chunk.get("url", "")

        # Format: [1] Content text...\n(Source: Title, URL: https://...)
        formatted_lines.append(f"[{i}] {content}\n(Source: {title}, URL: {url})")

        # Track metadata for response
        source_metadata.append({
            "citation_index": i,
            "article_id": chunk.get("article_id"),
            "chunk_id": chunk.get("chunk_id"),
            "title": title,
            "url": url,
            "excerpt": content[:200] + "..."  # First 200 chars
        })

    return "\n\n".join(formatted_lines), source_metadata
```

**System Prompt** (`chat/prompts.py:6-31`):

```python
RESEARCH_ASSISTANT_PROMPT = """You are a research assistant helping users understand
their saved articles and content. Your task is to answer questions based ONLY on the
provided context.

## Instructions

1. **Answer from context only**: Base your answers solely on the numbered context
   passages provided. Do not use external knowledge.

2. **Cite your sources**: Use inline citations in the format [1], [2], etc. to
   reference the context passages that support your statements.

3. **Be honest about limitations**: If the context doesn't contain enough information
   to fully answer the question, clearly state what information is missing.
"""
```

### Article Knowledge Retrieval

**⚠️ CRITICAL GAP: Search Integration Not Connected**

**Current State** (`chat/rag_pipeline.py:134-193`):

```python
async def _retrieve_chunks(self, session_id, query, limit, search_client=None):
    """Retrieve relevant chunks from R6 search API.

    TODO: Integrate with R6 search API when available
    For now, return empty to demonstrate flow
    """

    # Expected R6 API call:
    # GET /api/v1/sessions/{session_id}/search?q={query}&limit={limit}

    if search_client:
        # Use injected client for testing
        results = await search_client.search(session_id=session_id, query=query, limit=limit)
        return [...]

    # Placeholder until R6 is implemented
    return []  # ⚠️ ALWAYS RETURNS EMPTY!
```

**This means:**
- **RAG pipeline is NEVER retrieving actual article content**
- All chat responses are generated with `has_context=False`
- LLM is using `NO_CONTEXT_PROMPT` and answering without article knowledge
- Search API exists (`routers/search.py`) but is NOT integrated with chat

### Search API (Exists but Unused)

**Search Endpoint** (`routers/search.py:44-145`):

```python
@router.post("/sessions/{session_id}/search", response_model=SearchResponse)
async def search_session(session_id, request: SearchRequest, db, search: HybridSearch):
    """Search session knowledge using hybrid retrieval (dense + sparse)."""

    # Execute hybrid search with RRF fusion
    response = await search.search(
        session_id=session_id,
        request=request,
        query_embedding=query_embedding
    )

    return SearchResponse(
        query=request.query,
        results=results,  # List of SearchResult with chunks
        total_chunks_searched=total_chunks,
        search_mode=request.search_mode,
        timing_ms=timing_ms
    )
```

**SearchResult Schema** (`schemas/search.py:72-118`):

```python
class SearchResult(BaseModel):
    chunk_id: str          # "doc_abc123:chunk_5"
    article_id: int
    content: str | None    # Actual chunk text (if include_content=True)
    score: float           # RRF combined score
    source_url: str | None
    source_title: str | None
    dense_rank: int | None   # Rank from vector search
    sparse_rank: int | None  # Rank from BM25 search
```

### What's Returned to Users vs. What's Available

**Currently Returned** (`schemas/chat.py:86-135`):

```python
class ChatResponse(BaseModel):
    message_id: int
    content: str                    # LLM response with citations [1][2]
    sources: list[ChatSource]       # Cited sources only
    llm_provider: str               # "openai" or "anthropic"
    llm_model: str                  # "gpt-4o-mini"
    tokens_used: int                # Total tokens
    created_at: datetime

class ChatSource(BaseModel):
    citation_index: int        # [1], [2], etc.
    article_id: int
    chunk_id: str | None
    title: str | None
    url: str | None
    excerpt: str | None        # First 200 chars of chunk
```

**What Users See:**
- ✅ LLM-generated answer with inline citations [1], [2]
- ✅ List of cited sources with article title and URL
- ✅ Brief excerpt (200 chars) from each source
- ✅ Model used and token count

**What Users DON'T See:**
- ❌ **Full chunk content** that was used (only 200-char excerpt)
- ❌ **How chunks were retrieved** (search mode, ranking, scores)
- ❌ **Why these chunks were selected** (relevance scores, search method)
- ❌ **What other chunks were retrieved but not cited**
- ❌ **Search metadata** (dense rank, sparse rank, RRF score)
- ❌ **Context window** sent to LLM (formatted prompt with all chunks)

### ⚠️ **GAP #2: Limited Source Evidence**

**Current Citation Format:**

```json
{
  "sources": [
    {
      "citation_index": 1,
      "article_id": 123,
      "chunk_id": "doc_123:chunk_5",
      "title": "Understanding JWT Authentication",
      "url": "https://example.com/jwt",
      "excerpt": "JWT tokens are used for stateless authentication..."
    }
  ]
}
```

**Missing Information:**
1. **Full chunk text** - Users see only 200 chars, not the full passage cited
2. **Retrieval confidence** - No score or ranking to indicate how relevant the chunk was
3. **Alternative chunks** - No visibility into what else was considered
4. **Search method** - Users don't know if chunk was found via semantic or keyword search

---

## 3. Current Logging State

### Console Logging

**Chat Flow** (`routers/chat.py:86`):

```python
except Exception as e:
    print(f"RAG pipeline error: {e}")  # ⚠️ Simple print, no logger
    await db.rollback()
    raise HTTPException(500, detail=f"Failed to generate response: {str(e)}")
```

**Extraction Flow** (`tasks/extraction.py:18-112`):

```python
logger = logging.getLogger(__name__)

logger.info(f"Starting extraction for article {article_id}: {url}")
logger.info(f"Extraction completed: {result.word_count} words, method={result.extraction_method}")
logger.warning(f"Network error extracting article {article_id}: {e}")
logger.exception(f"Unexpected error extracting article {article_id}")
```

**Application Startup** (`main.py:34-48`):

```python
print(f"Starting {settings.app_name} v{settings.app_version}")
print("✅ Database connection verified")
print(f"⚠️  Database connection failed: {e}")
```

### What's Currently Logged

**Extraction Pipeline:**
- ✅ Extraction start (article ID, URL)
- ✅ Extraction completion (word count, method)
- ✅ Errors and warnings
- ❌ **No content preview or metadata**
- ❌ **No extraction duration metrics**

**Chat/RAG Pipeline:**
- ❌ **No logging of user questions**
- ❌ **No logging of retrieved chunks**
- ❌ **No logging of LLM prompts or responses**
- ❌ **No logging of source citations**
- ❌ **Single print statement for errors only**

**Search Pipeline:**
- ❌ **No logging of search queries**
- ❌ **No logging of search results or rankings**
- ❌ **No performance metrics (timing)**

### ⚠️ **GAP #3: Minimal RAG Traceability**

**For debugging RAG issues, developers need:**
- User's question
- Retrieved chunks (IDs, content, scores)
- Formatted context sent to LLM
- LLM response before citation extraction
- Which sources were cited vs. retrieved
- Search mode and performance metrics

**Current logging provides:** NONE of the above

### Configuration

**Log Levels** (`config.py:23-24`):

```python
log_level: str = "INFO"
sqlalchemy_log_level: str = "WARNING"  # SQLAlchemy-specific (INFO shows all SQL)
```

**No structured logging:**
- No request IDs for tracing
- No correlation between extraction → embedding → search → chat
- No JSON-formatted logs for parsing
- No separate log files (everything to console)

---

## 4. Gap Analysis & Recommendations

### Critical Evidence Gaps

#### 1. Article Content Verification Gap

**Problem:** Users cannot verify extracted content is accurate

**Impact:**
- No way to audit extraction quality
- Cannot detect truncation or corruption
- Cannot compare original vs. extracted text

**Recommendation:**
- **UI Enhancement:** Display extracted content with metadata in article detail view
- **API Enhancement:** Add preview/summary to ArticleResponse
- **Metrics:** Show extraction quality indicators (word count, reading time, warnings)

**Example Enhancement:**

```python
class ArticleResponse(BaseModel):
    # Existing fields...
    has_content: bool
    word_count: int | None
    reading_time_minutes: int | None
    extraction_method: str | None
    extraction_warnings: list[str] | None  # NEW
    content_preview: str | None  # NEW: First 500 chars
```

#### 2. RAG Context Transparency Gap

**Problem:** Users cannot see what content was used to answer their questions

**Impact:**
- Cannot verify answer grounding
- Cannot assess citation relevance
- Cannot understand why certain sources were used

**Recommendation:**
- **Return full chunk content** in ChatSource (not just 200-char excerpt)
- **Add search metadata** (relevance scores, search mode, rankings)
- **Show unchunked chunks** that were retrieved but not cited

**Example Enhancement:**

```python
class ChatSource(BaseModel):
    # Existing fields...
    excerpt: str  # Change from 200 chars to full chunk content
    relevance_score: float  # NEW: Search score
    search_method: str  # NEW: "semantic", "keyword", or "hybrid"
    dense_rank: int | None  # NEW: Position in semantic search
    sparse_rank: int | None  # NEW: Position in keyword search

class ChatResponse(BaseModel):
    # Existing fields...
    retrieval_metadata: RetrievalMetadata  # NEW: Search context

class RetrievalMetadata(BaseModel):
    chunks_retrieved: int
    chunks_cited: int
    search_mode: str  # "hybrid", "dense", "sparse"
    total_chunks_in_session: int
    search_timing_ms: int
```

#### 3. Search Integration Gap

**CRITICAL:** RAG pipeline is not connected to search API

**Problem:** Chat always generates answers with `has_context=False`

**Impact:**
- **RAG is completely non-functional**
- All answers are hallucinations (no article knowledge used)
- Search API exists but is unused

**Recommendation:**
- **Immediate Fix:** Connect `RAGPipeline._retrieve_chunks()` to search API
- **Implementation:** Call `/api/v1/sessions/{session_id}/search` internally
- **Validation:** Add integration tests verifying chunks are retrieved

**Code Fix:**

```python
async def _retrieve_chunks(self, session_id, query, limit, search_client=None):
    # Call internal search API
    from article_mind_service.search import HybridSearch
    from article_mind_service.schemas.search import SearchRequest, SearchMode

    search = HybridSearch()

    # Generate query embedding
    provider = get_embedding_provider()
    embeddings = await provider.embed([query])
    query_embedding = embeddings[0]

    # Execute search
    request = SearchRequest(
        query=query,
        top_k=limit,
        include_content=True,
        search_mode=SearchMode.HYBRID
    )

    response = await search.search(
        session_id=session_id,
        request=request,
        query_embedding=query_embedding
    )

    # Convert SearchResult to chunk dict
    return [
        {
            "content": r.content,
            "article_id": r.article_id,
            "chunk_id": r.chunk_id,
            "title": r.source_title,
            "url": r.source_url,
            "score": r.score,
            "dense_rank": r.dense_rank,
            "sparse_rank": r.sparse_rank
        }
        for r in response.results
    ]
```

#### 4. Logging and Observability Gap

**Problem:** Cannot debug RAG issues without structured logging

**Impact:**
- No visibility into what content is being retrieved
- Cannot trace why certain answers were generated
- Cannot diagnose poor answer quality

**Recommendation:** Add structured logging to RAG pipeline

**Example Logging:**

```python
import structlog

logger = structlog.get_logger(__name__)

async def query(self, session_id, question, db, search_client=None):
    logger.info("rag.query.start", session_id=session_id, question=question)

    # Retrieve chunks
    chunks = await self._retrieve_chunks(...)
    logger.info("rag.query.chunks_retrieved",
                session_id=session_id,
                chunks_count=len(chunks),
                chunk_ids=[c["chunk_id"] for c in chunks])

    # Format context
    context_str, source_metadata = format_context_with_metadata(chunks)
    logger.debug("rag.query.context_formatted",
                 session_id=session_id,
                 context_length=len(context_str),
                 context_preview=context_str[:500])

    # Generate answer
    llm_response = await self.llm_provider.generate(...)
    logger.info("rag.query.llm_response",
                session_id=session_id,
                provider=llm_response.provider,
                model=llm_response.model,
                tokens_input=llm_response.tokens_input,
                tokens_output=llm_response.tokens_output)

    # Extract citations
    cited_sources = self._extract_cited_sources(...)
    logger.info("rag.query.citations_extracted",
                session_id=session_id,
                cited_count=len(cited_sources),
                total_retrieved=len(chunks))

    return RAGResponse(...)
```

#### 5. Audit Trail Gap

**Problem:** No permanent record of RAG decisions for later analysis

**Impact:**
- Cannot analyze answer quality over time
- Cannot understand which articles are most useful
- Cannot detect retrieval issues

**Recommendation:** Persist RAG metadata to database

**Schema Addition:**

```python
class ChatMessage(Base):
    # Existing fields...

    # NEW: RAG metadata
    retrieval_metadata: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Search and retrieval metadata"
    )
    context_chunks: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Full chunks used for context (for audit)"
    )
```

**Example Data:**

```json
{
  "retrieval_metadata": {
    "search_mode": "hybrid",
    "chunks_retrieved": 5,
    "chunks_cited": 2,
    "search_timing_ms": 127,
    "total_chunks_in_session": 547,
    "dense_weight": 0.7,
    "sparse_weight": 0.3
  },
  "context_chunks": [
    {
      "chunk_id": "doc_123:chunk_5",
      "article_id": 123,
      "content": "Full chunk text...",
      "score": 0.0156,
      "dense_rank": 2,
      "sparse_rank": 1,
      "cited": true
    },
    // ... more chunks
  ]
}
```

### Implementation Priority Matrix

| Gap | Impact | Effort | Priority | Recommended Action |
|-----|--------|--------|----------|-------------------|
| **Search Integration** | CRITICAL | Medium | P0 | Connect RAG to search API immediately |
| **Article Content Display** | High | Low | P1 | Add UI to show extracted content |
| **Full Chunk Content in Citations** | High | Low | P1 | Return full text instead of 200-char excerpt |
| **Search Metadata in Response** | Medium | Low | P2 | Add scores, ranks, search mode to ChatSource |
| **Structured Logging** | High | Medium | P1 | Add structlog to RAG pipeline |
| **Audit Trail Persistence** | Medium | Medium | P2 | Store retrieval metadata in chat_messages |
| **Unchunked Chunks Display** | Low | Medium | P3 | Show all retrieved chunks, not just cited |
| **Extraction Quality Metrics** | Medium | Low | P2 | Display word count, warnings in UI |

---

## 5. Specific File Locations

### Article Scraping

- **Article Model:** `src/article_mind_service/models/article.py`
- **Article Router:** `src/article_mind_service/routers/articles.py`
- **Extraction Task:** `src/article_mind_service/tasks/extraction.py`
- **Extraction Pipeline:** `src/article_mind_service/extraction/pipeline.py`
- **Article Schemas:** `src/article_mind_service/schemas/article.py`

### Chat & RAG

- **Chat Router:** `src/article_mind_service/routers/chat.py`
- **RAG Pipeline:** `src/article_mind_service/chat/rag_pipeline.py`
- **Prompt Templates:** `src/article_mind_service/chat/prompts.py`
- **Chat Schemas:** `src/article_mind_service/schemas/chat.py`
- **Chat Model:** `src/article_mind_service/models/chat.py`

### Search

- **Search Router:** `src/article_mind_service/routers/search.py`
- **Hybrid Search:** `src/article_mind_service/search/hybrid_search.py`
- **Dense Search:** `src/article_mind_service/search/dense_search.py`
- **Sparse Search:** `src/article_mind_service/search/sparse_search.py`
- **Search Schemas:** `src/article_mind_service/schemas/search.py`

### LLM Providers

- **OpenAI Provider:** `src/article_mind_service/chat/providers/openai.py`
- **Anthropic Provider:** `src/article_mind_service/chat/providers/anthropic.py`
- **Provider Base:** `src/article_mind_service/chat/llm_providers.py`

### Configuration

- **Settings:** `src/article_mind_service/config.py`
- **Database:** `src/article_mind_service/database.py`
- **Main App:** `src/article_mind_service/main.py`

---

## 6. Current Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. ARTICLE SCRAPING FLOW                                            │
└─────────────────────────────────────────────────────────────────────┘

User → POST /api/v1/sessions/{id}/articles/url
         ↓
      Article Created (status=pending)
         ↓
      Background Task: extract_article_content()
         ↓
      ExtractionPipeline.extract(url)
         ↓
      Article Updated:
        - content_text (FULL TEXT) ✅ Stored
        - extraction_metadata ✅ Stored
        - word_count, reading_time ✅ Stored
        - extraction_method ✅ Stored
         ↓
      GET /api/v1/sessions/{id}/articles/{article_id}/content
         ↓
      ❌ UI MAY NOT CALL THIS - Users don't see content!

┌─────────────────────────────────────────────────────────────────────┐
│ 2. CHAT QUERY FLOW (CURRENT - BROKEN)                               │
└─────────────────────────────────────────────────────────────────────┘

User → POST /api/v1/sessions/{id}/chat
         ↓
      RAGPipeline.query(question)
         ↓
      _retrieve_chunks(session_id, query, limit=5)
         ↓
      ⚠️ RETURNS EMPTY LIST (search not connected!)
         ↓
      format_context_with_metadata([])  → "No relevant context found."
         ↓
      build_system_prompt(has_context=False)  → NO_CONTEXT_PROMPT
         ↓
      LLM.generate(system_prompt, question, context=[])
         ↓
      ❌ LLM responds WITHOUT article knowledge (hallucination mode)
         ↓
      _extract_cited_sources(content, [])  → []
         ↓
      ChatResponse(content=..., sources=[])
         ↓
      User sees answer with NO SOURCES

┌─────────────────────────────────────────────────────────────────────┐
│ 3. CHAT QUERY FLOW (INTENDED - ONCE FIXED)                         │
└─────────────────────────────────────────────────────────────────────┘

User → POST /api/v1/sessions/{id}/chat
         ↓
      RAGPipeline.query(question)
         ↓
      _retrieve_chunks(session_id, query, limit=5)
         ↓
      ✅ CALL HybridSearch.search(session_id, query, top_k=5)
         ↓
      HybridSearch executes:
        - DenseSearch (semantic vectors via ChromaDB)
        - SparseSearch (BM25 keyword matching)
        - RRF Fusion (combines rankings)
         ↓
      Returns SearchResult[] with:
        - chunk_id, article_id
        - content (FULL CHUNK TEXT) ✅ Available
        - score, dense_rank, sparse_rank ✅ Available
        - source_title, source_url ✅ Available
         ↓
      format_context_with_metadata(chunks)
         ↓
      Context formatted as:
        [1] Chunk content...\n(Source: Title, URL: https://...)
        [2] Chunk content...\n(Source: Title, URL: https://...)
         ↓
      LLM.generate(RESEARCH_ASSISTANT_PROMPT, question, context_chunks)
         ↓
      LLM response with inline citations: "Based on... [1][2]"
         ↓
      _extract_cited_sources(content, source_metadata)
         ↓
      Parses [1], [2] and returns ONLY cited sources
         ↓
      ChatResponse(
        content="...[1][2]",
        sources=[
          {citation_index: 1, article_id: 123, excerpt: "first 200 chars..."},
          {citation_index: 2, article_id: 124, excerpt: "first 200 chars..."}
        ]
      )
         ↓
      ❌ User sees 200-char excerpts, NOT full chunks
      ❌ User doesn't see search scores or rankings
      ❌ User doesn't see unchunked chunks (3 retrieved, 2 cited)

┌─────────────────────────────────────────────────────────────────────┐
│ 4. SEARCH API FLOW (EXISTS BUT UNUSED)                             │
└─────────────────────────────────────────────────────────────────────┘

User → POST /api/v1/sessions/{id}/search
         ↓
      HybridSearch.search(session_id, query, top_k=10)
         ↓
      SearchResponse(
        results=[
          {
            chunk_id, article_id,
            content: "FULL CHUNK TEXT" ✅,
            score: 0.0156,
            dense_rank: 2,
            sparse_rank: 1,
            source_url, source_title
          }
        ],
        total_chunks_searched: 547,
        search_mode: "hybrid",
        timing_ms: 127
      )
         ↓
      ❌ NEVER CALLED BY RAG PIPELINE
      ❌ PROBABLY NOT CALLED BY UI EITHER
```

---

## 7. Evidence Requirements by Stakeholder

### End Users

**Need to verify:**
- "What content was extracted from this article?" → ❌ Not shown
- "Is the extraction accurate?" → ❌ No quality indicators
- "What text was used to answer my question?" → ⚠️ Only 200-char excerpts
- "Why was this source cited?" → ❌ No relevance scores

**Current Visibility:**
- ✅ Article title and URL
- ✅ Extraction status (pending/completed/failed)
- ⚠️ Brief excerpt from cited sources (200 chars)
- ❌ Full extracted content
- ❌ Full chunks used for answers
- ❌ Search relevance or quality metrics

### Developers/Debuggers

**Need to trace:**
- "Which chunks were retrieved for this query?" → ❌ Not logged
- "Why did search return these chunks?" → ❌ No scores in logs
- "What context was sent to the LLM?" → ❌ Not logged
- "How did the LLM use the context?" → ❌ Not logged
- "Why is answer quality poor?" → ❌ No traceability

**Current Logging:**
- ✅ Extraction start/completion (article ID, URL, word count)
- ✅ Extraction errors
- ❌ RAG query details
- ❌ Retrieved chunks
- ❌ LLM prompts/responses
- ❌ Search performance
- ❌ Citation extraction

### Product/Analytics

**Need to understand:**
- "Which articles are most useful?" → ⚠️ Can infer from citation frequency
- "What are common query patterns?" → ⚠️ Stored in chat_messages
- "How often does RAG fail to find context?" → ❌ Not tracked
- "What's the average retrieval quality?" → ❌ Not tracked

**Current Data:**
- ✅ Chat messages (user questions, assistant responses)
- ✅ Citation counts (via sources JSON)
- ✅ Token usage per query
- ❌ Retrieval metadata (chunks retrieved vs cited)
- ❌ Search performance metrics
- ❌ Answer quality indicators

---

## 8. Recommended Next Steps

### Immediate (P0 - Critical)

1. **Fix RAG Integration**
   - Connect `RAGPipeline._retrieve_chunks()` to `HybridSearch`
   - Add integration test verifying chunks are retrieved
   - Validate context is sent to LLM
   - **File:** `src/article_mind_service/chat/rag_pipeline.py:134-193`

2. **Verify Article Content Display**
   - Check if frontend calls `/api/v1/sessions/{id}/articles/{article_id}/content`
   - If not, add UI component to display extracted content
   - **Files:** Frontend codebase, `src/article_mind_service/routers/articles.py:441-489`

### High Priority (P1)

3. **Enhance Citation Evidence**
   - Return full chunk content in `ChatSource.excerpt` (not 200 chars)
   - Add search metadata to citations (score, ranks, search mode)
   - **File:** `src/article_mind_service/schemas/chat.py:9-43`

4. **Add Structured Logging**
   - Install `structlog`
   - Log RAG pipeline execution (chunks, context, LLM response)
   - Log search queries and results
   - **Files:** `src/article_mind_service/chat/rag_pipeline.py`, `src/article_mind_service/routers/search.py`

5. **Display Extraction Metadata**
   - Show word count, reading time, warnings in article list/detail
   - Add extraction quality indicators
   - **File:** `src/article_mind_service/schemas/article.py`

### Medium Priority (P2)

6. **Persist RAG Metadata**
   - Add `retrieval_metadata` and `context_chunks` to `ChatMessage`
   - Store full retrieval context for audit
   - **File:** `src/article_mind_service/models/chat.py`

7. **Add Search Metadata to Chat Response**
   - Include `RetrievalMetadata` in `ChatResponse`
   - Show chunks retrieved vs cited, search timing
   - **File:** `src/article_mind_service/schemas/chat.py`

8. **Implement Search Analytics**
   - Track retrieval quality metrics
   - Monitor answer quality over time
   - Identify underutilized articles

### Low Priority (P3)

9. **Show Unchunked Chunks**
   - Display all retrieved chunks in UI, not just cited
   - Allow users to understand what else was considered
   - **File:** Frontend implementation

10. **Add Content Preview API**
    - Add `/api/v1/sessions/{id}/articles/{article_id}/preview` endpoint
    - Return first 500 chars + metadata
    - **File:** `src/article_mind_service/routers/articles.py`

---

## 9. Conclusion

The article-mind-service has a **robust data foundation** with comprehensive content extraction and storage, but suffers from **critical evidence and traceability gaps** in the user experience and debugging capabilities.

**Most Critical Issues:**

1. **RAG pipeline is non-functional** - Not connected to search, always returns no context
2. **Users cannot verify article content** - No UI to display extracted text
3. **Limited citation evidence** - Only 200-char excerpts, no full chunks or relevance scores
4. **Minimal logging** - Cannot debug RAG issues or trace answer quality

**Strengths:**

1. ✅ Article content is extracted and stored with rich metadata
2. ✅ Search infrastructure is implemented and functional
3. ✅ Source citations are tracked and returned to users
4. ✅ Data models support comprehensive audit trails

**The path forward is clear:** Fix the RAG integration (P0), enhance evidence in responses (P1), and add structured logging (P1). These three changes will dramatically improve both user trust and developer debuggability.

---

**Research Document Version:** 1.0
**Last Updated:** 2026-01-20
**Next Review:** After P0/P1 fixes are implemented
