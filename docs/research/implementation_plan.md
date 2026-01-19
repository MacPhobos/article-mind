# Body-of-Knowledge System — Phased Implementation Plan (TxDD-friendly)

## Goal
Build a system that ingests user-provided URLs (blog/news/wiki/etc), produces a durable “body of knowledge” on disk, indexes it via `mcp-vector-search`, and answers questions through an orchestrator that uses retrieval + LLM synthesis with grounding and citations.

## Operating principles (non-negotiable)
- **Reproducible ingestion**: store raw + extracted + normalized artifacts so you can reprocess without re-scraping.
- **Provenance everywhere**: every answer must cite URL + chunk IDs.
- **Incremental updates**: avoid re-embedding/re-indexing unchanged documents.
- **Fail loud**: if content is missing/weak, say so; don’t hallucinate.

## Assumptions
- Local-first filesystem storage (can be upgraded to object storage later).
- Index/search served via MCP using `mcp-vector-search`.
- Orchestrator can call tools (scrape, extract, chunk, index, search, rerank, answer).
- LLM can be OpenAI or Ollama (selected at runtime/config).

---

# Phase 0 — Repo skeleton + contracts (1–2 days)
### Outcome
A runnable scaffold with stable interfaces so later phases don’t rewrite everything.

### Scope
- Define data model, directories, tool interfaces, config, logging.

### Implementation tasks (tickets)
- **P0.1** Create repo structure:
    - `ingest/`, `extract/`, `normalize/`, `chunk/`, `index/`, `search/`, `orchestrator/`, `models/`, `cli/`, `tests/`
- **P0.2** Define canonical IDs:
    - `doc_id` = hash(canonical_url) or UUID + canonical_url mapping
    - `content_hash` = hash(normalized_text)
    - `chunk_id` = `{doc_id}:{chunk_index}:{chunk_hash_prefix}`
- **P0.3** Define on-disk schema (see “Storage layout” below)
- **P0.4** Add config system:
    - embedding model selection, LLM provider, rate limits, user agent, storage root
- **P0.5** Logging + run manifests:
    - Every run writes `runs/<timestamp>-<run_id>.json` capturing inputs/outputs

### Acceptance criteria
- `cli` prints config, creates storage directories, writes a sample doc record end-to-end (stubbed content OK).
- Stable JSON schemas documented.

### Non-goals
- No real scraping/indexing yet.

---

# Phase 1 — Ingestion MVP (fetch → raw storage) (2–5 days)
### Outcome
Given a URL, fetch content and store raw artifacts + request metadata.

### Scope
- HTTP fetch with redirects, headers, final URL, timestamps.
- Basic SSRF safety and rate limiting.

### Implementation tasks (tickets)
- **P1.1** Fetcher:
    - follow redirects, capture `final_url`, status, headers, content-type, byte size
    - support gzip/deflate
- **P1.2** URL canonicalization:
    - strip common trackers (`utm_*`), normalize scheme/host/path
- **P1.3** Content-type handling:
    - HTML: store `raw.html`
    - PDF: store `raw.pdf`
    - Other: store `raw.bin` + content-type
- **P1.4** Safety checks:
    - block private IP ranges / localhost (SSRF)
    - max size limits
- **P1.5** Manifest writing:
    - write `metadata.json` with fetch details and a first-pass doc record

### Acceptance criteria
- `ingest <url>` creates a new doc folder with `raw.*` + `metadata.json`.
- Handles redirects and preserves final canonical URL.

### Non-goals
- No extraction/cleanup yet.

### Risks
- Many sites require JS rendering; defer to Phase 2/3.

---

# Phase 2 — Extraction + normalization (main text → Markdown) (3–7 days)
### Outcome
Turn raw HTML/PDF into consistent normalized Markdown + structured metadata.

### Scope
- Readability-like extraction for HTML.
- PDF text extraction.
- Language detection (optional but useful).
- Basic publish-date heuristics (optional).

### Implementation tasks (tickets)
- **P2.1** HTML extraction:
    - remove boilerplate, nav, footers
    - preserve headings, lists, code blocks where possible
- **P2.2** PDF extraction:
    - extract text + basic structure; store extraction warnings if layout is messy
- **P2.3** Normalize to Markdown:
    - stable formatting (avoid nondeterministic whitespace)
- **P2.4** Metadata enrichment:
    - title, author (if present), publish date heuristic, site name, language
- **P2.5** Quality gate:
    - min character count, boilerplate ratio (simple heuristic), “empty/cookie wall” detection
    - mark doc as `status=EXTRACTED_OK|EXTRACTED_WEAK|FAILED`

### Acceptance criteria
- `extract <doc_id>` produces `extracted.md` + updates `metadata.json`.
- A “weak extraction” is explicitly flagged (not silently indexed).

### Non-goals
- No chunking/indexing yet.

---

# Phase 3 — Chunking + stable chunk IDs (2–5 days)
### Outcome
Deterministic chunking that preserves context (title + heading path) per chunk.

### Scope
- Heading-aware chunking + size-based splitting
- Parent context included in each chunk object
- Store chunks as `chunks.jsonl`

### Implementation tasks (tickets)
- **P3.1** Chunker:
    - split by headings → sections → chunks by target tokens/chars
- **P3.2** Context packaging per chunk:
    - `context = [title, heading_path, section_summary(optional)]`
- **P3.3** Stable chunk hashing + IDs
- **P3.4** Store chunk offsets (optional but helpful):
    - byte/line offsets into `extracted.md` for traceability

### Acceptance criteria
- `chunk <doc_id>` writes `chunks.jsonl` with stable `chunk_id` and content hashes.
- Re-running chunking on unchanged `extracted.md` yields identical chunk IDs.

### Non-goals
- No embeddings yet.

---

# Phase 4 — Embeddings + `mcp-vector-search` indexing (3–7 days)
### Outcome
Index chunks into `mcp-vector-search` with incremental updates.

### Scope
- Embedding provider abstraction (OpenAI / Ollama / local)
- Index lifecycle (create/update/delete)
- Incremental: only embed chunks that are new/changed

### Implementation tasks (tickets)
- **P4.1** Embedding interface:
    - `embed(texts[]) -> vectors[]` + metadata (model, dims)
- **P4.2** Incremental embedding:
    - compare `chunk_hash` against stored state
    - cache embeddings on disk (`embeddings/`)
- **P4.3** Index writer:
    - push `{chunk_id, vector, metadata}` to mcp-vector-search
    - metadata includes: canonical_url, doc_id, title, heading_path, scraped_at, published_at(optional)
- **P4.4** Index manifest:
    - store `index_state.json` (which docs/chunks indexed, embedding model version)
- **P4.5** Deletion / tombstones:
    - if doc removed, mark chunks deleted (don’t leave stale results)

### Acceptance criteria
- `index build` indexes all extracted chunks.
- `index update` only processes changed docs/chunks.
- Search returns chunk metadata sufficient to cite sources.

### Non-goals
- Reranking/hybrid search (next phase).

---

# Phase 5 — Retrieval quality: hybrid + rerank + dedupe (4–10 days)
### Outcome
Retrieval stops being “top-k embeddings” and becomes reliably relevant.

### Scope
- Query expansion (lightweight)
- Hybrid recall (vector + lexical fallback)
- Reranking
- Deduplication + diversity of sources

### Implementation tasks (tickets)
- **P5.1** Query planning:
    - generate 2–5 alternative queries (synonyms, entity expansions)
- **P5.2** Hybrid recall:
    - vector search top N per query
    - optional lexical search over `extracted.md` or `chunks.jsonl` (simple TF-IDF/BM25)
- **P5.3** Merge + dedupe:
    - remove near-duplicates by chunk hash / similarity
    - ensure source diversity (don’t return 10 chunks from same page unless necessary)
- **P5.4** Reranker:
    - lightweight: LLM scoring pass “does this chunk answer the question?”
    - keep top K (e.g., 8–20) for synthesis

### Acceptance criteria
- Retrieval returns a ranked set of chunks with reason codes (e.g., `vector_hit`, `lexical_hit`, `rerank_score`).
- For a small golden set, retrieval recall improves over vector-only baseline.

### Non-goals
- Final answer orchestration (next phase).

---

# Phase 6 — Orchestrator for grounded Q&A (4–10 days)
### Outcome
A reliable Q&A flow that cites sources and refuses unsupported claims.

### Scope
- Question classification
- Retrieval (Phase 5)
- Answer synthesis with citations
- Contradiction handling and “not in KB” behavior
- Output structured + human-friendly

### Implementation tasks (tickets)
- **P6.1** Q&A pipeline:
    1) classify question (lookup vs synthesis vs opinion vs not-in-kb)
    2) retrieval plan + execute
    3) generate answer from provided evidence only
- **P6.2** Citation format:
    - include `canonical_url` + `chunk_id` (and optional short excerpt)
- **P6.3** Faithfulness guardrails:
    - require explicit mapping from claims → supporting chunk IDs
- **P6.4** Contradictions:
    - detect conflicting statements; present both with dates and sources
- **P6.5** “Not enough evidence” mode:
    - answer: “Not found in your sources” + suggest what to add

### Acceptance criteria
- For test questions, answers include citations and do not invent unsupported details.
- When KB lacks support, the system explicitly says so.

### Non-goals
- UI / chat frontend (optional later).

---

# Phase 7 — Multi-user / collections / lifecycle (optional but likely) (5–15 days)
### Outcome
Practical management for real usage: namespaces, tags, retention, reindexing.

### Scope
- Namespaces per user/project
- Collections/tags
- Retention policies
- Re-embedding and index migrations

### Implementation tasks (tickets)
- **P7.1** Namespace support:
    - `storage_root/<namespace>/docs/...`
    - index per namespace or metadata filter
- **P7.2** Collections/tags:
    - `collections.json` mapping docs → tags
- **P7.3** Retention:
    - user-selectable keep raw/extracted/chunks/embeddings
- **P7.4** Migration commands:
    - re-embed all with new model
    - rebuild index from disk without re-scrape

### Acceptance criteria
- Multiple namespaces can coexist without cross-contamination.
- One command can rebuild index from stored artifacts.

---

# Phase 8 — Evaluation harness + regression gates (do earlier than you think) (3–8 days)
### Outcome
You can measure if changes improve or degrade retrieval/answering.

### Scope
- Golden set of questions + expected sources
- Basic metrics: retrieval recall, citation precision, faithfulness

### Implementation tasks (tickets)
- **P8.1** Golden set format:
    - YAML: `{question, expected_urls[], expected_chunk_keywords[]}`
- **P8.2** Metrics runner:
    - run retrieval and score coverage
- **P8.3** CI gate:
    - fail build if key metrics drop beyond threshold

### Acceptance criteria
- One command runs eval and prints a report.
- CI can block regressions in retrieval/citation quality.

---

# Storage layout (recommended)
```
storage/
  <namespace>/
    docs/
      <doc_id>/
        raw/
          raw.html | raw.pdf | raw.bin
        extracted/
          extracted.md
        chunks/
          chunks.jsonl
        embeddings/
          <embedding_model>/
            chunk_vectors.jsonl   # or per-chunk files
        metadata.json
    index_state.json
    runs/
      2026-01-08T12-34-56Z-<run_id>.json
```

---

# TxDD Ticket Template (copy/paste)
## Ticket: <Phase X.Y> <Title>
**Goal / user value**
- …

**Scope**
- In:
- Out:

**Interfaces / inputs**
- CLI/API:
- Input files:
- Output files:

**Implementation notes**
- …

**Acceptance criteria**
- …

**Test plan**
- Unit:
- Integration:
- Golden set / eval impact:

**Risks / edge cases**
- …

---

# Suggested sequencing (don’t overbuild early)
- Build Phases **0 → 4** first to get a complete ingest→index loop.
- Then do **6** (grounded Q&A) with **vector-only retrieval** as a baseline.
- Then add **5** (hybrid + rerank) once you can measure improvement.
- Add **7/8** as soon as you feel pain, not before.
