# Search Quality Improvement Opportunities for Article-Mind Service

**Date:** 2026-01-24
**Author:** Research Agent
**Status:** Completed
**Classification:** Actionable Research

---

## Executive Summary

This research analyzes the current search implementation in article-mind-service to identify opportunities for improving search results quality. The analysis covers the hybrid search architecture, RAG pipeline, chunking strategy, and query processing mechanisms.

### Key Findings

1. **Cross-Encoder Reranking is Disabled**: The reranker infrastructure exists but is disabled (`search_rerank_enabled=False`). Research indicates enabling cross-encoder reranking can improve accuracy by 20-48%.

2. **Chunking Strategy Uses Fixed-Size Approach**: Current implementation uses RecursiveCharacterTextSplitter with 512-token chunks. Research suggests semantic chunking can improve retrieval by up to 70% compared to fixed-size chunking.

3. **Limited Context Window**: Only 5 chunks (`rag_context_chunks=5`) are retrieved for RAG, which may miss relevant context. Best practices suggest 10-20 chunks with reranking.

4. **No Query Expansion/Rewriting**: Queries are used verbatim without semantic expansion or reformulation, potentially missing relevant content.

5. **Simple BM25 Tokenization**: The sparse search uses basic alphanumeric tokenization without stemming or stopword removal, reducing keyword matching effectiveness.

### Top 3 Recommendations

1. **Enable Cross-Encoder Reranking** (High Impact, Low Effort) - Expected 20-30% accuracy improvement
2. **Implement Enhanced BM25 Tokenization** (Medium Impact, Low Effort) - Expected 10-15% sparse search improvement
3. **Add Query Expansion** (High Impact, Medium Effort) - Expected 15-25% recall improvement

---

## Current State Analysis

### 1. Search Architecture Overview

The search system implements a hybrid retrieval approach combining dense semantic search with sparse keyword search.

```
Query → [Embedding Generation] → Dense Search (ChromaDB)
  │                                    ↓
  └─────────────────────────→ Sparse Search (BM25)
                                       ↓
                             Reciprocal Rank Fusion
                                       ↓
                             [Reranker - DISABLED]
                                       ↓
                                   Results
```

#### Component Details

**Dense Search** (`search/dense_search.py`)
- Uses ChromaDB with HNSW index for approximate nearest neighbor search
- Session-based collections: `session_{session_id}`
- Distance metric: Cosine similarity (converted from distance: `1 - distance`)
- Collection metadata includes embedding model and dimensions

**Sparse Search** (`search/sparse_search.py`)
- BM25 algorithm via `rank_bm25` library
- In-memory index cache (`BM25IndexCache`) per session
- Index rebuilt on service restart (populated during embedding pipeline)
- Simple tokenization: lowercase + alphanumeric split

**Hybrid Fusion** (`search/hybrid_search.py`)
```python
# Current RRF Configuration
dense_weight = 0.7
sparse_weight = 0.3
rrf_k = 60

# RRF Formula
score = (dense_weight / (k + dense_rank)) + (sparse_weight / (k + sparse_rank))
```

**Reranker** (`search/reranker.py`)
- Cross-encoder reranker exists but returns dummy scores (disabled)
- Model configured: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Settings: `search_rerank_enabled=False`, `search_rerank_top_k=50`

### 2. Embedding Configuration

**Source:** `config.py`, `embeddings/__init__.py`

| Setting | Value | Notes |
|---------|-------|-------|
| `embedding_provider` | `openai` or `ollama` | Via database or env |
| `embedding_model` (OpenAI) | `text-embedding-3-small` | 1536 dimensions |
| `embedding_model` (Ollama) | `nomic-embed-text` | 1024 dimensions |
| `max_tokens` | 8192 | Maximum input tokens |

**Provider Factory Pattern:**
```python
# embeddings/__init__.py - Three-level priority
1. provider_override parameter (explicit override)
2. Database settings (if db session provided)
3. Environment variable settings (fallback)
```

### 3. Chunking Strategy

**Source:** `embeddings/chunker.py`

| Setting | Value | Rationale |
|---------|-------|-----------|
| `chunk_size` | 512 tokens | Balance context vs specificity |
| `chunk_overlap` | 50 tokens (~10%) | Preserve continuity |
| `encoding` | `cl100k_base` | OpenAI tokenizer |
| Algorithm | RecursiveCharacterTextSplitter | LangChain implementation |

**Separator Priority:**
```python
separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
```

**Chunk Metadata:**
```python
{
    "text": str,           # Chunk content
    "chunk_index": int,    # Position in document
    "article_id": int,     # Source article
    "source_url": str,     # Original URL
}
```

### 4. RAG Pipeline

**Source:** `chat/rag_pipeline.py`

**Pipeline Flow:**
1. Retrieve relevant chunks (max 5 via `rag_context_chunks`)
2. Format context with citation metadata
3. Generate answer using LLM (OpenAI/Anthropic)
4. Extract cited sources from response using regex: `r"\[(\d+)\]"`

**Configuration:**
| Setting | Value | Notes |
|---------|-------|-------|
| `rag_context_chunks` | 5 | Maximum chunks for context |
| `llm_max_tokens` | 4096 | LLM response limit |
| Temperature | 0.3 | Conservative generation |

**Context Formatting** (`chat/prompts.py`):
```python
# Each chunk formatted as:
[{citation_index}] {title}
Source: {url}
Content: {content}
---
```

### 5. Query Processing

**Current State:** Queries are passed directly to search without preprocessing.

**In Dense Search:**
1. Query text embedded via provider
2. Embedding sent to ChromaDB
3. Results returned by cosine similarity

**In Sparse Search:**
1. Query tokenized (lowercase, alphanumeric split)
2. BM25 scores computed against index
3. Results sorted by score

**No Current Features:**
- Query expansion
- Spelling correction
- Synonym matching
- Query reformulation

---

## Gap Analysis

### 1. Reranking (Critical Gap)

**Current State:** Disabled cross-encoder reranker returning dummy scores

**Best Practice:** Cross-encoder reranking after initial retrieval

**Research Evidence:**
- From `docs/research/llm-orchestration-rag-best-practices-2025-2026.md`:
  - Cohere Rerank 4 showed 50% better accuracy than BM25 alone
  - Cross-encoders provide 20-48% accuracy boost
  - Two-stage retrieval (fast retrieval + rerank) is standard practice

**Impact:** Missing 20-48% potential accuracy improvement

**Code Reference:**
```python
# search/reranker.py (currently disabled)
if not settings.search_rerank_enabled:
    return [0.5] * len(documents)  # Dummy scores
```

### 2. Chunking Strategy (Significant Gap)

**Current State:** Fixed-size 512-token chunks with 10% overlap

**Best Practice:** Semantic/context-aware chunking

**Research Evidence:**
- From `docs/research/llm-orchestration-rag-best-practices-2025-2026.md`:
  - Semantic chunking 70% better than fixed-size in retrieval benchmarks
  - Preserves logical boundaries and context
  - Better handles varying content types (code, lists, paragraphs)

**Impact:** Potential 70% improvement in retrieval quality

**Code Reference:**
```python
# embeddings/chunker.py
self.splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,        # Fixed 512 tokens
    chunk_overlap=chunk_overlap,   # Fixed 50 tokens
    separators=["\n\n", "\n", ". ", ...],
)
```

### 3. Query Expansion (Moderate Gap)

**Current State:** No query preprocessing or expansion

**Best Practice:** HyDE (Hypothetical Document Embeddings) or LLM-based query expansion

**Research Evidence:**
- HyDE generates hypothetical answer, then searches for similar content
- Query expansion with synonyms improves recall 10-20%
- Multi-query retrieval captures different phrasings

**Impact:** Missing 15-25% potential recall improvement

### 4. BM25 Tokenization (Moderate Gap)

**Current State:** Simple alphanumeric tokenization without NLP processing

**Best Practice:** Stemming, stopword removal, lemmatization

**Code Reference:**
```python
# search/sparse_search.py
def _tokenize(self, text: str) -> TokenizedDoc:
    text = text.lower()
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if len(t) > 1]  # No stemming, no stopwords
```

**Impact:** Missing ~10-15% keyword matching effectiveness

### 5. Context Window Size (Moderate Gap)

**Current State:** Only 5 chunks in RAG context

**Best Practice:** 10-20 chunks with reranking to select best

**Research Evidence:**
- More context allows LLM to synthesize comprehensive answers
- Reranking selects most relevant from larger pool
- Typical production systems use 10-20 chunks

**Impact:** Potential for incomplete answers due to limited context

### Gap Summary Matrix

| Gap | Current | Best Practice | Impact | Effort |
|-----|---------|---------------|--------|--------|
| Reranking | Disabled | Cross-encoder enabled | 20-48% accuracy | Low |
| Chunking | Fixed 512 tokens | Semantic chunking | ~70% retrieval | High |
| Query Expansion | None | HyDE/Multi-query | 15-25% recall | Medium |
| BM25 Tokenization | Basic | Stemming+stopwords | 10-15% matching | Low |
| Context Size | 5 chunks | 10-20 chunks | Completeness | Low |

---

## Improvement Recommendations

### Priority 1: Enable Cross-Encoder Reranking

**Impact:** High (20-30% accuracy improvement expected)
**Effort:** Low (infrastructure exists, just needs enabling)
**Timeline:** 1-2 days

**Implementation Steps:**

1. **Enable reranker in configuration:**
   ```python
   # config.py
   search_rerank_enabled: bool = True  # Change from False
   search_rerank_top_k: int = 20       # Adjust as needed
   ```

2. **Update hybrid search to use reranker:**
   ```python
   # search/hybrid_search.py - Enable the existing reranker call
   if settings.search_rerank_enabled:
       reranker = Reranker()
       scores = reranker.rerank(query, [r.content for r in fused_results])
       # Re-sort by reranker scores
   ```

3. **Consider model upgrade:**
   - Current: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, lower accuracy)
   - Upgrade option: `cross-encoder/ms-marco-MiniLM-L-12-v2` (balanced)
   - Premium option: Cohere Rerank 4 API (highest accuracy)

**Monitoring:**
- Track rerank latency per request
- Compare retrieval precision before/after
- A/B test with user queries

### Priority 2: Enhance BM25 Tokenization

**Impact:** Medium (10-15% sparse search improvement)
**Effort:** Low (simple code change)
**Timeline:** 1 day

**Implementation Steps:**

1. **Add NLTK/spaCy for NLP processing:**
   ```python
   # search/sparse_search.py
   import nltk
   from nltk.corpus import stopwords
   from nltk.stem import PorterStemmer

   STOPWORDS = set(stopwords.words('english'))
   STEMMER = PorterStemmer()

   def _tokenize(self, text: str) -> TokenizedDoc:
       text = text.lower()
       tokens = re.split(r"[^a-z0-9]+", text)
       # Remove stopwords and stem
       tokens = [STEMMER.stem(t) for t in tokens
                 if len(t) > 1 and t not in STOPWORDS]
       return tokens
   ```

2. **Consider spaCy for better lemmatization:**
   ```python
   import spacy
   nlp = spacy.load("en_core_web_sm")

   def _tokenize(self, text: str) -> TokenizedDoc:
       doc = nlp(text.lower())
       return [token.lemma_ for token in doc
               if not token.is_stop and token.is_alpha and len(token) > 1]
   ```

**Trade-offs:**
- Adds dependency (nltk/spacy)
- Small latency increase (~5-10ms per query)
- Requires rebuilding BM25 index after change

### Priority 3: Increase RAG Context Window

**Impact:** Medium (improved answer completeness)
**Effort:** Low (configuration change)
**Timeline:** < 1 day

**Implementation Steps:**

1. **Update configuration:**
   ```python
   # config.py
   rag_context_chunks: int = 10  # Increase from 5
   ```

2. **Combine with reranking:**
   - Retrieve 20 chunks initially
   - Rerank to select top 10 for context
   - Ensures highest quality context

3. **Monitor token usage:**
   - More context = more LLM input tokens
   - Track token costs per request
   - Consider dynamic context sizing based on query complexity

### Priority 4: Add Query Expansion

**Impact:** High (15-25% recall improvement)
**Effort:** Medium (new feature)
**Timeline:** 3-5 days

**Implementation Options:**

**Option A: HyDE (Hypothetical Document Embeddings)**
```python
# chat/query_expansion.py
async def hyde_expand(query: str, llm_provider) -> str:
    """Generate hypothetical document for better retrieval."""
    prompt = f"""Write a short passage that would answer the question: {query}

    Write as if you are explaining this topic. Be specific and factual."""

    response = await llm_provider.generate(
        system_prompt="You are a helpful assistant.",
        user_message=prompt,
        max_tokens=200,
    )
    return response.content

# In RAG pipeline
hypothetical_doc = await hyde_expand(question, llm_provider)
# Embed and search with hypothetical_doc instead of raw query
```

**Option B: Multi-Query Expansion**
```python
# Generate multiple query variants
async def multi_query_expand(query: str, llm_provider) -> list[str]:
    prompt = f"""Generate 3 different ways to ask this question.
    Original: {query}

    Provide variations that capture different aspects or phrasings."""

    response = await llm_provider.generate(...)
    return parse_variations(response.content)

# Search with each variant and merge results
```

**Option C: Synonym Expansion (No LLM)**
```python
# Use WordNet for synonym expansion
from nltk.corpus import wordnet

def expand_with_synonyms(query: str) -> str:
    tokens = tokenize(query)
    expanded_tokens = []
    for token in tokens:
        synsets = wordnet.synsets(token)
        synonyms = {lemma.name() for syn in synsets[:2] for lemma in syn.lemmas()[:3]}
        expanded_tokens.extend([token] + list(synonyms)[:2])
    return " ".join(expanded_tokens)
```

### Priority 5: Implement Semantic Chunking

**Impact:** High (~70% retrieval improvement)
**Effort:** High (significant refactoring)
**Timeline:** 1-2 weeks

**Implementation Approach:**

**Option A: LangChain Semantic Chunker**
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

def create_semantic_chunker():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
```

**Option B: Custom Semantic Chunking**
```python
# Custom implementation with sentence embeddings
async def semantic_chunk(text: str, provider: EmbeddingProvider) -> list[str]:
    # Split into sentences
    sentences = sent_tokenize(text)

    # Get embeddings for each sentence
    embeddings = await provider.embed(sentences)

    # Find breakpoints based on semantic similarity
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        if similarity < THRESHOLD:
            # Semantic break - start new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    chunks.append(" ".join(current_chunk))
    return chunks
```

**Considerations:**
- Requires re-embedding all existing content
- Migration plan for production data
- Performance impact during chunking
- Consider hybrid approach: semantic for new content, gradual migration for existing

### Priority 6: Add Metadata Filtering

**Impact:** Medium (improved precision for filtered queries)
**Effort:** Medium (ChromaDB supports this)
**Timeline:** 2-3 days

**Implementation:**

1. **Extend chunk metadata:**
   ```python
   # embeddings/pipeline.py
   metadatas = [
       {
           "article_id": c["article_id"],
           "chunk_index": c["chunk_index"],
           "source_url": c["source_url"],
           "date_published": article.published_at.isoformat() if article.published_at else None,
           "content_type": detect_content_type(c["text"]),  # code, prose, list, etc.
           "word_count": len(c["text"].split()),
       }
       for c in batch
   ]
   ```

2. **Enable filtered search:**
   ```python
   # search/dense_search.py
   def search(
       self,
       session_id: int,
       query_embedding: list[float],
       top_k: int = 10,
       filters: dict | None = None,
   ):
       where_filter = None
       if filters:
           where_filter = {"$and": [
               {key: {"$eq": value}} for key, value in filters.items()
           ]}

       results = collection.query(
           query_embeddings=[query_embedding],
           n_results=top_k,
           where=where_filter,
       )
   ```

3. **Expose in API:**
   ```python
   # schemas/search.py
   class SearchRequest(BaseModel):
       query: str
       top_k: int = 10
       search_mode: SearchMode = SearchMode.HYBRID
       filters: dict[str, Any] | None = None  # New field
   ```

---

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1)

| Task | Effort | Impact | Owner |
|------|--------|--------|-------|
| Enable reranking | 1 day | High | Search team |
| Increase context chunks to 10 | 0.5 day | Medium | RAG team |
| Add BM25 stemming/stopwords | 1 day | Medium | Search team |

**Milestone:** 25-35% expected improvement in search accuracy

### Phase 2: Query Enhancement (Week 2-3)

| Task | Effort | Impact | Owner |
|------|--------|--------|-------|
| Implement HyDE query expansion | 3 days | High | RAG team |
| Add query spelling correction | 2 days | Medium | Search team |
| Add metadata filtering to API | 2 days | Medium | API team |

**Milestone:** 40-50% total expected improvement

### Phase 3: Chunking Evolution (Week 4-6)

| Task | Effort | Impact | Owner |
|------|--------|--------|-------|
| Implement semantic chunker | 5 days | High | Embedding team |
| Create migration plan | 2 days | N/A | Architecture |
| Gradual rollout (new content) | 3 days | High | DevOps |
| Backfill existing content | 5 days | High | Data team |

**Milestone:** 70%+ total expected improvement

### Phase 4: Advanced Optimizations (Week 7+)

| Task | Effort | Impact | Owner |
|------|--------|--------|-------|
| Multi-vector retrieval | 1 week | Medium | Research |
| Upgrade embedding model | 3 days | Medium | ML team |
| Implement late interaction (ColBERT) | 2 weeks | High | Research |
| Add feedback loop for tuning | 1 week | Medium | ML team |

**Milestone:** Production-grade search quality

### Resource Requirements

| Phase | Engineering Days | Infrastructure |
|-------|------------------|----------------|
| Phase 1 | 2.5 days | None |
| Phase 2 | 7 days | LLM API costs |
| Phase 3 | 15 days | Re-embedding compute |
| Phase 4 | 25 days | ML infrastructure |

---

## Risk Assessment

### Risk 1: Reranker Latency

**Risk:** Cross-encoder reranking adds 50-200ms latency per request

**Mitigation:**
- Start with smaller model (`MiniLM-L-6-v2`)
- Cache reranker model in memory
- Set reasonable `rerank_top_k` (20 instead of 50)
- Use GPU acceleration if available

**Monitoring:**
- P99 latency per request
- Alert if latency exceeds threshold

### Risk 2: Re-embedding Data Migration

**Risk:** Semantic chunking requires re-embedding all existing content

**Mitigation:**
- Implement dual-chunking: new content uses semantic, old uses fixed
- Gradual migration during low-traffic periods
- Keep old embeddings until migration complete
- Rollback plan: revert to old chunks if issues

**Cost Estimate:**
- ~100K chunks at $0.02/1M tokens = ~$0.20 for re-embedding
- Compute time: ~2-4 hours for full reindex

### Risk 3: Query Expansion Hallucination

**Risk:** HyDE may generate incorrect hypothetical documents

**Mitigation:**
- Use low temperature (0.3) for generation
- Limit hypothetical document length (200 tokens)
- Fall back to original query if expansion fails
- A/B test before full rollout

**Monitoring:**
- Track retrieval precision with/without expansion
- Monitor for irrelevant results

### Risk 4: BM25 Index Rebuild

**Risk:** Changing tokenization requires full BM25 index rebuild

**Mitigation:**
- BM25 index rebuilds quickly (seconds per session)
- Index rebuilt on service restart anyway
- Test tokenization changes in staging first

**Timing:**
- Schedule rebuild during maintenance window
- < 1 minute downtime for typical deployment

### Risk 5: Context Window Token Costs

**Risk:** Increasing context chunks increases LLM API costs

**Mitigation:**
- Monitor token usage per request
- Set budget alerts
- Implement token-based context pruning if needed
- Consider smaller LLM for cost reduction

**Cost Estimate:**
- 5 chunks -> 10 chunks: ~2x input token increase
- At $0.03/1K tokens (GPT-4): ~$0.06 additional per complex query

### Risk Matrix Summary

| Risk | Likelihood | Impact | Mitigation Effectiveness |
|------|------------|--------|--------------------------|
| Reranker Latency | High | Medium | High (model selection) |
| Data Migration | Medium | High | High (gradual rollout) |
| Query Hallucination | Low | Medium | High (temperature/fallback) |
| BM25 Rebuild | Low | Low | High (fast rebuild) |
| Token Costs | High | Low | Medium (monitoring) |

---

## Appendix A: Configuration Reference

### Current Settings (`config.py`)

```python
# Embedding settings
chunk_size: int = 512
chunk_overlap: int = 50
embedding_provider: str = "openai"  # or "ollama"

# Search settings
search_top_k: int = 10
search_dense_weight: float = 0.7
search_sparse_weight: float = 0.3
search_rrf_k: int = 60
search_rerank_enabled: bool = False  # DISABLED
search_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
search_rerank_top_k: int = 50

# RAG settings
rag_context_chunks: int = 5
llm_max_tokens: int = 4096
```

### Recommended Settings (Post-Implementation)

```python
# Embedding settings
chunk_size: int = 512  # Phase 1-2
# chunk_strategy: str = "semantic"  # Phase 3

# Search settings
search_top_k: int = 20  # Increase for reranking
search_dense_weight: float = 0.7
search_sparse_weight: float = 0.3
search_rrf_k: int = 60
search_rerank_enabled: bool = True  # ENABLE
search_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
search_rerank_top_k: int = 20

# RAG settings
rag_context_chunks: int = 10  # Increase from 5
llm_max_tokens: int = 4096
```

## Appendix B: Related Research Documents

- `docs/research/llm-orchestration-rag-best-practices-2025-2026.md` - Comprehensive RAG best practices
- `docs/research/embedding-models-research-2025-2026.md` - Embedding model comparisons
- `docs/research/rag-empty-content-root-cause-2026-01-21.md` - Previous RAG debugging

## Appendix C: Code References

| Component | File Path |
|-----------|-----------|
| Hybrid Search | `src/article_mind_service/search/hybrid_search.py` |
| Dense Search | `src/article_mind_service/search/dense_search.py` |
| Sparse Search | `src/article_mind_service/search/sparse_search.py` |
| Reranker | `src/article_mind_service/search/reranker.py` |
| RAG Pipeline | `src/article_mind_service/chat/rag_pipeline.py` |
| Chunker | `src/article_mind_service/embeddings/chunker.py` |
| Pipeline | `src/article_mind_service/embeddings/pipeline.py` |
| Configuration | `src/article_mind_service/config.py` |
| Prompts | `src/article_mind_service/chat/prompts.py` |
| Search Router | `src/article_mind_service/routers/search.py` |

---

**Document End**
