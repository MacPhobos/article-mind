# Plan R4: URL Content Extraction

**Plan ID:** R4-content-extraction
**Created:** 2026-01-19
**Dependencies:** Plan 02 (scaffold-service), Plan 03 (database-setup)
**Estimated Time:** 3-5 days
**Research Reference:** `docs/research/python-content-extraction-libraries-2025.md`

---

## 1. Overview

### Purpose

Implement a robust, multi-format content extraction pipeline that downloads web pages and PDFs from user-provided URLs, extracts clean text content, and stores the results for downstream processing (chunking, embedding, Q&A).

### Scope

- Download web pages (HTML) and PDF documents from URLs
- Extract clean text content using best-in-class libraries
- Handle JavaScript-rendered pages when static extraction fails
- Detect content type automatically (HTML vs PDF)
- Store extracted content with metadata
- Provide async background task processing with status tracking
- Implement retry logic and error handling

### Out of Scope (Future Phases)

- Chunking (Phase 3 of implementation plan)
- Embedding and indexing (Phase 4)
- OCR for scanned PDFs (deferred - can add later)
- Video/audio extraction
- Multi-page crawling (single URL ingestion only)

### Dependencies

- **Plan 02:** Requires FastAPI scaffolding
- **Plan 03:** Requires PostgreSQL database for article storage and status tracking

### Outputs

- Extraction pipeline module: `src/article_mind_service/extraction/`
- Background task processing for async extraction
- Updated Article model with extraction status fields
- API endpoint to trigger extraction
- Comprehensive test suite with fixtures

---

## 2. Technology Stack

### Primary Extraction Libraries

| Content Type | Primary Library | Fallback | License | Notes |
|--------------|-----------------|----------|---------|-------|
| **HTML** | trafilatura 2.0.0 | newspaper4k 0.9.4.1 | GPL-3.0 / MIT | F1: 0.958 accuracy |
| **PDF** | PyMuPDF + pymupdf4llm | pdfplumber | AGPL-3.0 / MIT | 0.12s extraction, markdown output |
| **JS-rendered** | Playwright | - | Apache-2.0 | 35-45% faster than Selenium |

### Supporting Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| **httpx** | Async HTTP client | 0.27+ |
| **chardet/charset-normalizer** | Character encoding detection | Latest |
| **aiofiles** | Async file operations | 24.1+ |

### Why These Choices (from Research)

1. **trafilatura** - Highest F1 score (0.958) in article extraction benchmarks, fast (100-300ms), multilingual support, used by HuggingFace, IBM, Microsoft Research
2. **newspaper4k** - Excellent metadata extraction (author, date, keywords), MIT license, Playwright integration for JS sites
3. **PyMuPDF + pymupdf4llm** - Best speed/quality balance for PDFs, excellent markdown output for LLM ingestion, GNN-based layout analysis
4. **Playwright** - Official Python support, 35-45% faster than Selenium, auto-wait mechanism, network interception

---

## 3. Content Type Detection

### Detection Strategy

Content type detection uses a multi-step approach:

```
1. URL Pattern Check
   ├── *.pdf → PDF
   ├── *.html → HTML
   └── Other → Continue to step 2

2. HEAD Request (lightweight)
   ├── Content-Type: application/pdf → PDF
   ├── Content-Type: text/html → HTML
   └── Unknown → Continue to step 3

3. Response Content Sniffing
   ├── Starts with %PDF → PDF
   ├── Contains <html> or <!DOCTYPE → HTML
   └── Other → Treat as HTML (best effort)
```

### Implementation

```python
from enum import Enum
from urllib.parse import urlparse

class ContentType(str, Enum):
    HTML = "html"
    PDF = "pdf"
    UNKNOWN = "unknown"

async def detect_content_type(url: str, client: httpx.AsyncClient) -> ContentType:
    """Detect content type using URL pattern and HEAD request."""
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Step 1: URL pattern check
    if path.endswith('.pdf'):
        return ContentType.PDF
    if path.endswith(('.html', '.htm')):
        return ContentType.HTML

    # Step 2: HEAD request
    try:
        response = await client.head(url, follow_redirects=True, timeout=10.0)
        content_type = response.headers.get('content-type', '').lower()

        if 'application/pdf' in content_type:
            return ContentType.PDF
        if 'text/html' in content_type:
            return ContentType.HTML
    except Exception:
        pass  # Fall through to content sniffing

    return ContentType.HTML  # Default assumption
```

---

## 4. Extraction Pipeline Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXTRACTION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐ │
│  │ URL  │───►│ Detect Type │───►│   Fetch      │───►│    Extract      │ │
│  └──────┘    └─────────────┘    └──────────────┘    └─────────────────┘ │
│                    │                   │                    │            │
│                    ▼                   ▼                    ▼            │
│              ┌──────────┐       ┌───────────┐       ┌───────────────┐   │
│              │  HTML    │       │  Raw      │       │  HTML         │   │
│              │  PDF     │       │  Content  │       │  Extractor    │   │
│              │  Unknown │       │  Bytes    │       │  (trafilatura │   │
│              └──────────┘       └───────────┘       │  + newspaper) │   │
│                                                     ├───────────────┤   │
│                                                     │  PDF          │   │
│                                                     │  Extractor    │   │
│                                                     │  (PyMuPDF)    │   │
│                                                     ├───────────────┤   │
│                                                     │  JS           │   │
│                                                     │  Extractor    │   │
│                                                     │  (Playwright) │   │
│                                                     └───────────────┘   │
│                                                            │            │
│                    ┌───────────────┐    ┌──────────────────┘            │
│                    │               │    │                               │
│                    ▼               ▼    ▼                               │
│              ┌──────────┐    ┌────────────┐    ┌──────────────────────┐ │
│              │  Clean   │◄───│ Extracted  │───►│  Store in DB         │ │
│              │  Content │    │  Text +    │    │  (article.content,   │ │
│              └──────────┘    │  Metadata  │    │   article.status)    │ │
│                              └────────────┘    └──────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Status Tracking

Articles progress through these statuses:

```
┌──────────┐    ┌─────────────┐    ┌───────────────┐
│ PENDING  │───►│ PROCESSING  │───►│  COMPLETED    │
└──────────┘    └─────────────┘    └───────────────┘
                      │                    ▲
                      │                    │
                      ▼                    │
                ┌──────────┐         (retry with JS)
                │  FAILED  │───────────────┘
                └──────────┘
```

### Error Handling Strategy

| Error Type | Handling | Retry? |
|------------|----------|--------|
| Network timeout | Log, set FAILED, store error message | Yes (3x with backoff) |
| HTTP 4xx | Log, set FAILED, store status code | No |
| HTTP 5xx | Log, set FAILED, store error | Yes (3x with backoff) |
| Empty content | Retry with JS rendering | Once |
| Extraction error | Log, set FAILED, store stack trace | No |
| Rate limited | Wait and retry with exponential backoff | Yes |

---

## 5. Module Structure

```
article-mind-service/
└── src/article_mind_service/
    ├── extraction/
    │   ├── __init__.py           # Public exports
    │   ├── base.py               # Abstract base extractor
    │   ├── html_extractor.py     # trafilatura + newspaper4k
    │   ├── pdf_extractor.py      # PyMuPDF + pymupdf4llm
    │   ├── js_extractor.py       # Playwright for JS rendering
    │   ├── pipeline.py           # Orchestration and routing
    │   ├── content_type.py       # Content type detection
    │   ├── utils.py              # Content cleaning utilities
    │   └── exceptions.py         # Custom exception types
    ├── models/
    │   └── article.py            # UPDATED: Add extraction fields
    ├── schemas/
    │   ├── article.py            # NEW: Article schemas
    │   └── extraction.py         # NEW: Extraction status schemas
    ├── routers/
    │   └── articles.py           # NEW: Article CRUD + extraction trigger
    └── tasks/
        └── extraction.py         # Background task handlers
```

### Module Dependencies

```
pipeline.py
    ├── content_type.py (detect_content_type)
    ├── html_extractor.py (HTMLExtractor)
    ├── pdf_extractor.py (PDFExtractor)
    ├── js_extractor.py (JSExtractor)
    └── utils.py (clean_text, normalize_whitespace)

html_extractor.py
    ├── base.py (BaseExtractor)
    └── utils.py

pdf_extractor.py
    ├── base.py (BaseExtractor)
    └── utils.py

js_extractor.py
    ├── base.py (BaseExtractor)
    ├── html_extractor.py (for post-render extraction)
    └── utils.py
```

---

## 6. Implementation Steps

### Step 1: Add Dependencies (30 min)

Update `pyproject.toml`:

```toml
[project.dependencies]
# ... existing deps ...
# Content extraction
trafilatura = ">=2.0.0"
newspaper4k = ">=0.9.4"
pymupdf = ">=1.26.0"
pymupdf4llm = ">=0.0.17"
playwright = ">=1.50.0"

# HTTP and async
httpx = ">=0.27.0"
aiofiles = ">=24.1.0"
charset-normalizer = ">=3.4.0"

[project.optional-dependencies]
playwright = ["playwright>=1.50.0"]
```

Install Playwright browsers (CI/deployment):

```bash
# After pip install
playwright install chromium
```

### Step 2: Create Exception Types (30 min)

Create `src/article_mind_service/extraction/exceptions.py`:

```python
"""Custom exceptions for content extraction."""

class ExtractionError(Exception):
    """Base exception for extraction errors."""
    pass

class NetworkError(ExtractionError):
    """Network-related errors (timeout, connection refused)."""
    pass

class ContentTypeError(ExtractionError):
    """Unable to determine or handle content type."""
    pass

class EmptyContentError(ExtractionError):
    """Extraction returned empty or minimal content."""
    pass

class RateLimitError(ExtractionError):
    """Rate limited by the target server."""
    pass

class ContentTooLargeError(ExtractionError):
    """Content exceeds maximum size limit."""
    pass
```

### Step 3: Create Base Extractor (1 hour)

Create `src/article_mind_service/extraction/base.py`:

```python
"""Abstract base class for content extractors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ExtractionResult:
    """Result of content extraction."""

    content: str
    title: str | None = None
    author: str | None = None
    published_date: datetime | None = None
    language: str | None = None
    word_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    extraction_method: str = ""
    extraction_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate word count if not provided."""
        if not self.word_count and self.content:
            self.word_count = len(self.content.split())


class BaseExtractor(ABC):
    """Abstract base class for content extractors."""

    @abstractmethod
    async def extract(self, content: bytes | str, url: str) -> ExtractionResult:
        """Extract clean text from content.

        Args:
            content: Raw content (bytes for PDF, str for HTML)
            url: Source URL for context/metadata

        Returns:
            ExtractionResult with extracted text and metadata
        """
        pass

    @abstractmethod
    def can_extract(self, content_type: str) -> bool:
        """Check if this extractor can handle the content type.

        Args:
            content_type: MIME type or content category

        Returns:
            True if this extractor can handle the content
        """
        pass
```

### Step 4: Implement HTML Extractor (2 hours)

Create `src/article_mind_service/extraction/html_extractor.py`:

```python
"""HTML content extraction using trafilatura with newspaper4k fallback."""

import time
from typing import Any

import trafilatura
from newspaper import Article

from .base import BaseExtractor, ExtractionResult
from .exceptions import EmptyContentError
from .utils import clean_text, normalize_whitespace


class HTMLExtractor(BaseExtractor):
    """Extract content from HTML using trafilatura + newspaper4k fallback."""

    MIN_CONTENT_LENGTH = 100  # Minimum characters for valid extraction

    def can_extract(self, content_type: str) -> bool:
        """Check if this extractor handles the content type."""
        return content_type.lower() in ("html", "text/html")

    async def extract(self, content: bytes | str, url: str) -> ExtractionResult:
        """Extract content from HTML.

        Strategy:
        1. Try trafilatura (highest accuracy)
        2. Fallback to newspaper4k if trafilatura fails or returns weak content

        Args:
            content: HTML content as string or bytes
            url: Source URL

        Returns:
            ExtractionResult with extracted text

        Raises:
            EmptyContentError: If no content could be extracted
        """
        start_time = time.perf_counter()

        # Convert bytes to string if needed
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')

        # Try trafilatura first (best accuracy)
        result = await self._extract_with_trafilatura(content, url)

        # Check quality and fallback if needed
        if not result.content or len(result.content) < self.MIN_CONTENT_LENGTH:
            newspaper_result = await self._extract_with_newspaper(content, url)

            if newspaper_result.content and len(newspaper_result.content) > len(result.content or ""):
                result = newspaper_result
                result.warnings.append("Used newspaper4k fallback (trafilatura returned weak content)")

        # Final validation
        if not result.content or len(result.content) < self.MIN_CONTENT_LENGTH:
            raise EmptyContentError(
                f"Extraction returned insufficient content ({len(result.content or '')} chars)"
            )

        result.extraction_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    async def _extract_with_trafilatura(
        self, html: str, url: str
    ) -> ExtractionResult:
        """Extract using trafilatura."""
        # Extract with metadata
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            include_images=False,
            include_links=False,
            output_format='txt',
            with_metadata=True,
        )

        # Get metadata separately
        metadata = trafilatura.extract(
            html,
            url=url,
            output_format='dict',
            with_metadata=True,
        ) or {}

        content = clean_text(extracted) if extracted else ""

        return ExtractionResult(
            content=content,
            title=metadata.get('title'),
            author=metadata.get('author'),
            language=metadata.get('language'),
            metadata={
                'sitename': metadata.get('sitename'),
                'categories': metadata.get('categories'),
                'tags': metadata.get('tags'),
            },
            extraction_method="trafilatura",
        )

    async def _extract_with_newspaper(
        self, html: str, url: str
    ) -> ExtractionResult:
        """Extract using newspaper4k."""
        article = Article(url)
        article.set_html(html)
        article.parse()

        # Optional NLP processing for keywords/summary
        try:
            article.nlp()
            keywords = article.keywords
            summary = article.summary
        except Exception:
            keywords = []
            summary = None

        content = clean_text(article.text) if article.text else ""

        return ExtractionResult(
            content=content,
            title=article.title,
            author=", ".join(article.authors) if article.authors else None,
            published_date=article.publish_date,
            language=article.meta_lang,
            metadata={
                'top_image': article.top_image,
                'keywords': keywords,
                'summary': summary,
                'movies': article.movies,
            },
            extraction_method="newspaper4k",
        )
```

### Step 5: Implement PDF Extractor (2 hours)

Create `src/article_mind_service/extraction/pdf_extractor.py`:

```python
"""PDF content extraction using PyMuPDF + pymupdf4llm."""

import time
import tempfile
import os

import fitz  # PyMuPDF
import pymupdf4llm

from .base import BaseExtractor, ExtractionResult
from .exceptions import EmptyContentError, ExtractionError
from .utils import clean_text


class PDFExtractor(BaseExtractor):
    """Extract content from PDF using PyMuPDF + pymupdf4llm."""

    MIN_CONTENT_LENGTH = 100

    def can_extract(self, content_type: str) -> bool:
        """Check if this extractor handles the content type."""
        return content_type.lower() in ("pdf", "application/pdf")

    async def extract(self, content: bytes | str, url: str) -> ExtractionResult:
        """Extract content from PDF.

        Strategy:
        1. Try pymupdf4llm for markdown output (best for LLMs)
        2. Fallback to basic PyMuPDF text extraction

        Args:
            content: PDF content as bytes
            url: Source URL

        Returns:
            ExtractionResult with extracted text

        Raises:
            EmptyContentError: If no content could be extracted
        """
        start_time = time.perf_counter()

        if isinstance(content, str):
            content = content.encode('utf-8')

        # Write to temp file (PyMuPDF works better with files)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = await self._extract_pdf(tmp_path, url)
        finally:
            os.unlink(tmp_path)

        if not result.content or len(result.content) < self.MIN_CONTENT_LENGTH:
            raise EmptyContentError(
                f"PDF extraction returned insufficient content ({len(result.content or '')} chars)"
            )

        result.extraction_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    async def _extract_pdf(self, file_path: str, url: str) -> ExtractionResult:
        """Extract content from PDF file."""
        warnings: list[str] = []

        # Try pymupdf4llm first (best markdown output)
        try:
            md_text = pymupdf4llm.to_markdown(file_path)
            if md_text and len(md_text.strip()) > self.MIN_CONTENT_LENGTH:
                # Extract metadata
                doc = fitz.open(file_path)
                metadata = doc.metadata or {}
                doc.close()

                return ExtractionResult(
                    content=clean_text(md_text),
                    title=metadata.get('title'),
                    author=metadata.get('author'),
                    metadata={
                        'subject': metadata.get('subject'),
                        'keywords': metadata.get('keywords'),
                        'creator': metadata.get('creator'),
                        'producer': metadata.get('producer'),
                        'page_count': doc.page_count,
                    },
                    extraction_method="pymupdf4llm",
                    warnings=warnings,
                )
        except Exception as e:
            warnings.append(f"pymupdf4llm failed: {e}")

        # Fallback to basic PyMuPDF extraction
        try:
            doc = fitz.open(file_path)
            text_parts: list[str] = []

            for page in doc:
                text_parts.append(page.get_text())

            text = "\n\n".join(text_parts)
            metadata = doc.metadata or {}
            page_count = doc.page_count
            doc.close()

            return ExtractionResult(
                content=clean_text(text),
                title=metadata.get('title'),
                author=metadata.get('author'),
                metadata={
                    'subject': metadata.get('subject'),
                    'keywords': metadata.get('keywords'),
                    'page_count': page_count,
                },
                extraction_method="pymupdf_basic",
                warnings=warnings,
            )
        except Exception as e:
            raise ExtractionError(f"PDF extraction failed: {e}") from e
```

### Step 6: Implement JS Extractor (2 hours)

Create `src/article_mind_service/extraction/js_extractor.py`:

```python
"""JavaScript-rendered content extraction using Playwright."""

import time
from typing import Any

from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeout

from .base import BaseExtractor, ExtractionResult
from .html_extractor import HTMLExtractor
from .exceptions import NetworkError, ExtractionError


class JSExtractor(BaseExtractor):
    """Extract content from JavaScript-rendered pages using Playwright."""

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 30000,
        block_resources: bool = True,
    ) -> None:
        """Initialize JS extractor.

        Args:
            headless: Run browser in headless mode
            timeout_ms: Navigation timeout in milliseconds
            block_resources: Block images/fonts to speed up loading
        """
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.block_resources = block_resources
        self._html_extractor = HTMLExtractor()

    def can_extract(self, content_type: str) -> bool:
        """JS extractor handles HTML that needs rendering."""
        return content_type.lower() in ("html", "text/html", "js-html")

    async def extract(self, content: bytes | str, url: str) -> ExtractionResult:
        """Extract content by rendering the page with Playwright.

        Note: For JS extraction, we ignore the content parameter and
        fetch fresh from the URL with JavaScript execution.

        Args:
            content: Ignored (we fetch fresh)
            url: URL to render and extract

        Returns:
            ExtractionResult with rendered content
        """
        start_time = time.perf_counter()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)

            try:
                page = await browser.new_page()

                # Block unnecessary resources for speed
                if self.block_resources:
                    await page.route(
                        "**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf,eot,ico}",
                        lambda route: route.abort()
                    )

                # Navigate and wait for content
                try:
                    await page.goto(
                        url,
                        wait_until="networkidle",
                        timeout=self.timeout_ms,
                    )
                except PlaywrightTimeout as e:
                    # Try with domcontentloaded if networkidle times out
                    await page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=self.timeout_ms,
                    )

                # Wait a bit more for lazy-loaded content
                await page.wait_for_timeout(1000)

                # Get rendered HTML
                html = await page.content()

                # Extract using HTML extractor
                result = await self._html_extractor.extract(html, url)
                result.extraction_method = f"playwright+{result.extraction_method}"
                result.metadata['rendered'] = True

            finally:
                await browser.close()

        result.extraction_time_ms = (time.perf_counter() - start_time) * 1000
        return result
```

### Step 7: Create Content Utilities (1 hour)

Create `src/article_mind_service/extraction/utils.py`:

```python
"""Content cleaning and processing utilities."""

import re
import unicodedata


def clean_text(text: str | None) -> str:
    """Clean and normalize extracted text.

    - Normalizes Unicode
    - Removes excessive whitespace
    - Removes control characters
    - Normalizes line endings

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Unicode normalization (NFC form)
    text = unicodedata.normalize('NFC', text)

    # Remove control characters except newlines and tabs
    text = ''.join(
        char for char in text
        if unicodedata.category(char) != 'Cc' or char in '\n\t'
    )

    # Normalize whitespace
    text = normalize_whitespace(text)

    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    - Replaces multiple spaces with single space
    - Replaces multiple newlines with double newline (paragraph break)
    - Removes trailing whitespace from lines

    Args:
        text: Text to normalize

    Returns:
        Text with normalized whitespace
    """
    # Replace tabs with spaces
    text = text.replace('\t', ' ')

    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)

    # Replace 3+ newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def is_boilerplate(text: str, threshold: float = 0.3) -> bool:
    """Check if text is likely boilerplate content.

    Uses lexical diversity as a heuristic - boilerplate tends to
    have low unique word ratio.

    Args:
        text: Text to check
        threshold: Minimum unique word ratio (default 0.3)

    Returns:
        True if text appears to be boilerplate
    """
    words = text.lower().split()
    if len(words) < 20:
        return False  # Too short to judge

    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < threshold


def estimate_reading_time(text: str, wpm: int = 200) -> int:
    """Estimate reading time in minutes.

    Args:
        text: Content text
        wpm: Words per minute (default 200)

    Returns:
        Estimated reading time in minutes
    """
    word_count = len(text.split())
    return max(1, round(word_count / wpm))
```

### Step 8: Implement Pipeline Orchestrator (2 hours)

Create `src/article_mind_service/extraction/pipeline.py`:

```python
"""Content extraction pipeline orchestration."""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from .base import ExtractionResult
from .content_type import ContentType, detect_content_type
from .html_extractor import HTMLExtractor
from .pdf_extractor import PDFExtractor
from .js_extractor import JSExtractor
from .exceptions import (
    ExtractionError,
    NetworkError,
    EmptyContentError,
    RateLimitError,
    ContentTooLargeError,
)


class ExtractionStatus(str, Enum):
    """Extraction status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for extraction pipeline."""

    timeout_seconds: int = 30
    max_retries: int = 3
    max_content_size_mb: int = 50
    user_agent: str = "ArticleMind/1.0 (Content Extraction Bot)"
    playwright_headless: bool = True
    retry_with_js: bool = True

    @property
    def max_content_size_bytes(self) -> int:
        return self.max_content_size_mb * 1024 * 1024


class ExtractionPipeline:
    """Orchestrates content extraction from URLs."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        """Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        self._html_extractor = HTMLExtractor()
        self._pdf_extractor = PDFExtractor()
        self._js_extractor: JSExtractor | None = None

    def _get_js_extractor(self) -> JSExtractor:
        """Lazy-initialize JS extractor (expensive to create)."""
        if self._js_extractor is None:
            self._js_extractor = JSExtractor(
                headless=self.config.playwright_headless,
                timeout_ms=self.config.timeout_seconds * 1000,
            )
        return self._js_extractor

    async def extract(self, url: str) -> ExtractionResult:
        """Extract content from URL.

        Flow:
        1. Detect content type
        2. Fetch content
        3. Route to appropriate extractor
        4. If HTML extraction fails/weak, retry with JS rendering

        Args:
            url: URL to extract content from

        Returns:
            ExtractionResult with extracted content

        Raises:
            NetworkError: If network request fails
            ExtractionError: If extraction fails
            ContentTooLargeError: If content exceeds size limit
        """
        async with httpx.AsyncClient(
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": self.config.user_agent},
        ) as client:
            # Step 1: Detect content type
            content_type = await detect_content_type(url, client)

            # Step 2: Fetch content with retry
            content, final_url = await self._fetch_with_retry(client, url)

            # Step 3: Route to appropriate extractor
            try:
                if content_type == ContentType.PDF:
                    return await self._pdf_extractor.extract(content, final_url)
                else:
                    return await self._extract_html(content, final_url)
            except EmptyContentError as e:
                # Step 4: Retry with JS rendering for HTML
                if content_type == ContentType.HTML and self.config.retry_with_js:
                    return await self._extract_with_js(final_url)
                raise

    async def _fetch_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> tuple[bytes, str]:
        """Fetch content with retry logic.

        Args:
            client: HTTP client
            url: URL to fetch

        Returns:
            Tuple of (content bytes, final URL after redirects)
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                response = await client.get(url)

                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(retry_after)
                        continue
                    raise RateLimitError(f"Rate limited by {url}")

                response.raise_for_status()

                # Check content size
                content_length = len(response.content)
                if content_length > self.config.max_content_size_bytes:
                    raise ContentTooLargeError(
                        f"Content size {content_length} exceeds limit "
                        f"{self.config.max_content_size_bytes}"
                    )

                return response.content, str(response.url)

            except httpx.TimeoutException as e:
                last_error = NetworkError(f"Timeout fetching {url}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    last_error = NetworkError(f"Server error {e.response.status_code}: {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                else:
                    raise NetworkError(f"HTTP error {e.response.status_code}: {e}") from e
            except httpx.RequestError as e:
                last_error = NetworkError(f"Request error: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        raise last_error or NetworkError(f"Failed to fetch {url}")

    async def _extract_html(self, content: bytes, url: str) -> ExtractionResult:
        """Extract from HTML content."""
        return await self._html_extractor.extract(content, url)

    async def _extract_with_js(self, url: str) -> ExtractionResult:
        """Extract using JavaScript rendering."""
        js_extractor = self._get_js_extractor()
        result = await js_extractor.extract(b"", url)
        result.warnings.append("Used JavaScript rendering (static extraction failed)")
        return result
```

### Step 9: Create Content Type Detection (30 min)

Create `src/article_mind_service/extraction/content_type.py`:

```python
"""Content type detection utilities."""

from enum import Enum
from urllib.parse import urlparse

import httpx


class ContentType(str, Enum):
    """Supported content types."""
    HTML = "html"
    PDF = "pdf"
    UNKNOWN = "unknown"


async def detect_content_type(url: str, client: httpx.AsyncClient) -> ContentType:
    """Detect content type from URL and HEAD request.

    Detection strategy:
    1. URL pattern (*.pdf, *.html)
    2. HEAD request Content-Type header
    3. Default to HTML

    Args:
        url: URL to check
        client: HTTP client for HEAD request

    Returns:
        Detected ContentType
    """
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Step 1: URL pattern
    if path.endswith('.pdf'):
        return ContentType.PDF
    if path.endswith(('.html', '.htm')):
        return ContentType.HTML

    # Step 2: HEAD request
    try:
        response = await client.head(url, follow_redirects=True, timeout=10.0)
        content_type_header = response.headers.get('content-type', '').lower()

        if 'application/pdf' in content_type_header:
            return ContentType.PDF
        if 'text/html' in content_type_header:
            return ContentType.HTML
    except Exception:
        pass  # Fall through to default

    # Default to HTML
    return ContentType.HTML


def detect_content_type_from_bytes(content: bytes) -> ContentType:
    """Detect content type by inspecting content bytes.

    Args:
        content: Content bytes to inspect

    Returns:
        Detected ContentType
    """
    if content.startswith(b'%PDF'):
        return ContentType.PDF

    # Check for HTML markers in first 1KB
    preview = content[:1024].lower()
    if b'<!doctype html' in preview or b'<html' in preview:
        return ContentType.HTML

    return ContentType.UNKNOWN
```

### Step 10: Module Exports (15 min)

Create `src/article_mind_service/extraction/__init__.py`:

```python
"""Content extraction module.

Provides extraction of clean text content from URLs (HTML and PDF).

Usage:
    from article_mind_service.extraction import ExtractionPipeline, PipelineConfig

    config = PipelineConfig(timeout_seconds=60)
    pipeline = ExtractionPipeline(config)

    result = await pipeline.extract("https://example.com/article")
    print(result.content)
"""

from .base import BaseExtractor, ExtractionResult
from .content_type import ContentType, detect_content_type
from .exceptions import (
    ExtractionError,
    NetworkError,
    EmptyContentError,
    RateLimitError,
    ContentTooLargeError,
    ContentTypeError,
)
from .html_extractor import HTMLExtractor
from .pdf_extractor import PDFExtractor
from .js_extractor import JSExtractor
from .pipeline import ExtractionPipeline, PipelineConfig, ExtractionStatus
from .utils import clean_text, normalize_whitespace, is_boilerplate, estimate_reading_time

__all__ = [
    # Base classes
    "BaseExtractor",
    "ExtractionResult",
    # Content types
    "ContentType",
    "detect_content_type",
    # Extractors
    "HTMLExtractor",
    "PDFExtractor",
    "JSExtractor",
    # Pipeline
    "ExtractionPipeline",
    "PipelineConfig",
    "ExtractionStatus",
    # Exceptions
    "ExtractionError",
    "NetworkError",
    "EmptyContentError",
    "RateLimitError",
    "ContentTooLargeError",
    "ContentTypeError",
    # Utilities
    "clean_text",
    "normalize_whitespace",
    "is_boilerplate",
    "estimate_reading_time",
]
```

### Step 11: Update Article Model (1 hour)

Update `src/article_mind_service/models/article.py`:

```python
"""Article model for storing extracted content."""

from datetime import datetime
from typing import Any
from enum import Enum

from sqlalchemy import String, Text, DateTime, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ExtractionStatus(str, Enum):
    """Article extraction status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Article(Base):
    """Article model representing an extracted web page or document."""

    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(primary_key=True)

    # URL information
    url: Mapped[str] = mapped_column(String(2048), index=True, unique=True)
    canonical_url: Mapped[str | None] = mapped_column(String(2048))

    # Extraction status
    status: Mapped[str] = mapped_column(
        String(20),
        default=ExtractionStatus.PENDING.value,
        index=True,
    )
    extraction_method: Mapped[str | None] = mapped_column(String(50))
    extraction_error: Mapped[str | None] = mapped_column(Text)

    # Extracted content
    title: Mapped[str | None] = mapped_column(String(1024))
    content: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)

    # Metadata
    author: Mapped[str | None] = mapped_column(String(512))
    published_date: Mapped[datetime | None] = mapped_column(DateTime)
    language: Mapped[str | None] = mapped_column(String(10))
    word_count: Mapped[int | None] = mapped_column(Integer)
    reading_time_minutes: Mapped[int | None] = mapped_column(Integer)

    # Extended metadata (JSON)
    metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    extracted_at: Mapped[datetime | None] = mapped_column(DateTime)

    def __repr__(self) -> str:
        return f"<Article id={self.id} url={self.url[:50]}... status={self.status}>"
```

### Step 12: Create Article Schemas (1 hour)

Create `src/article_mind_service/schemas/article.py`:

```python
"""Article request/response schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class ArticleCreate(BaseModel):
    """Schema for creating an article."""

    url: HttpUrl = Field(..., description="URL to extract content from")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"url": "https://example.com/article"}
            ]
        }
    }


class ArticleResponse(BaseModel):
    """Schema for article response."""

    id: int
    url: str
    canonical_url: str | None = None
    status: str
    extraction_method: str | None = None
    extraction_error: str | None = None

    title: str | None = None
    content: str | None = None

    author: str | None = None
    published_date: datetime | None = None
    language: str | None = None
    word_count: int | None = None
    reading_time_minutes: int | None = None

    metadata: dict[str, Any] | None = None

    created_at: datetime
    updated_at: datetime
    extracted_at: datetime | None = None

    model_config = {"from_attributes": True}


class ArticleListResponse(BaseModel):
    """Schema for article list response."""

    items: list[ArticleResponse]
    total: int
    page: int
    page_size: int


class ExtractionStatusResponse(BaseModel):
    """Schema for extraction status response."""

    article_id: int
    url: str
    status: str
    extraction_method: str | None = None
    error: str | None = None

    model_config = {"from_attributes": True}
```

### Step 13: Create Background Task Handler (2 hours)

Create `src/article_mind_service/tasks/extraction.py`:

```python
"""Background task handlers for content extraction."""

import asyncio
import hashlib
import logging
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.extraction import (
    ExtractionPipeline,
    PipelineConfig,
    ExtractionResult,
    ExtractionError,
    NetworkError,
)
from article_mind_service.models.article import Article, ExtractionStatus
from article_mind_service.config import settings

logger = logging.getLogger(__name__)


async def extract_article_content(
    article_id: int,
    db: AsyncSession,
    config: PipelineConfig | None = None,
) -> None:
    """Background task to extract content from an article's URL.

    Updates the article record with extraction results.

    Args:
        article_id: ID of article to extract
        db: Database session
        config: Optional pipeline configuration
    """
    # Load article
    article = await db.get(Article, article_id)
    if not article:
        logger.error(f"Article {article_id} not found")
        return

    # Update status to processing
    article.status = ExtractionStatus.PROCESSING.value
    article.extraction_error = None
    await db.commit()

    # Initialize pipeline
    pipeline_config = config or PipelineConfig(
        timeout_seconds=settings.extraction_timeout_seconds,
        max_retries=settings.extraction_max_retries,
        user_agent=settings.extraction_user_agent,
        playwright_headless=settings.playwright_headless,
    )
    pipeline = ExtractionPipeline(pipeline_config)

    try:
        # Run extraction
        logger.info(f"Starting extraction for article {article_id}: {article.url}")
        result = await pipeline.extract(str(article.url))

        # Update article with results
        article.status = ExtractionStatus.COMPLETED.value
        article.title = result.title
        article.content = result.content
        article.content_hash = _compute_hash(result.content)
        article.author = result.author
        article.published_date = result.published_date
        article.language = result.language
        article.word_count = result.word_count
        article.reading_time_minutes = _estimate_reading_time(result.word_count)
        article.metadata = result.metadata
        article.extraction_method = result.extraction_method
        article.extracted_at = datetime.utcnow()

        if result.warnings:
            article.metadata = article.metadata or {}
            article.metadata['extraction_warnings'] = result.warnings

        logger.info(
            f"Extraction completed for article {article_id}: "
            f"{result.word_count} words, method={result.extraction_method}"
        )

    except NetworkError as e:
        logger.warning(f"Network error extracting article {article_id}: {e}")
        article.status = ExtractionStatus.FAILED.value
        article.extraction_error = f"Network error: {e}"

    except ExtractionError as e:
        logger.warning(f"Extraction error for article {article_id}: {e}")
        article.status = ExtractionStatus.FAILED.value
        article.extraction_error = f"Extraction error: {e}"

    except Exception as e:
        logger.exception(f"Unexpected error extracting article {article_id}")
        article.status = ExtractionStatus.FAILED.value
        article.extraction_error = f"Unexpected error: {e}"

    finally:
        article.updated_at = datetime.utcnow()
        await db.commit()


def _compute_hash(content: str | None) -> str | None:
    """Compute SHA-256 hash of content."""
    if not content:
        return None
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _estimate_reading_time(word_count: int | None, wpm: int = 200) -> int | None:
    """Estimate reading time in minutes."""
    if not word_count:
        return None
    return max(1, round(word_count / wpm))
```

Create `src/article_mind_service/tasks/__init__.py`:

```python
"""Background tasks module."""

from .extraction import extract_article_content

__all__ = ["extract_article_content"]
```

### Step 14: Create Articles Router (2 hours)

Create `src/article_mind_service/routers/articles.py`:

```python
"""Article CRUD and extraction endpoints."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.database import get_db
from article_mind_service.models.article import Article, ExtractionStatus
from article_mind_service.schemas.article import (
    ArticleCreate,
    ArticleResponse,
    ArticleListResponse,
    ExtractionStatusResponse,
)
from article_mind_service.tasks import extract_article_content

router = APIRouter(prefix="/api/v1/articles", tags=["articles"])


@router.post(
    "/",
    response_model=ArticleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create article and start extraction",
    description="Creates an article record and starts background extraction.",
)
async def create_article(
    article_in: ArticleCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> Article:
    """Create a new article and trigger content extraction.

    The extraction runs as a background task. Poll the article status
    to check extraction progress.
    """
    # Check if URL already exists
    existing = await db.execute(
        select(Article).where(Article.url == str(article_in.url))
    )
    existing_article = existing.scalar_one_or_none()

    if existing_article:
        # Return existing article, optionally re-trigger extraction
        if existing_article.status == ExtractionStatus.FAILED.value:
            background_tasks.add_task(
                extract_article_content, existing_article.id, db
            )
        return existing_article

    # Create new article
    article = Article(
        url=str(article_in.url),
        status=ExtractionStatus.PENDING.value,
    )
    db.add(article)
    await db.commit()
    await db.refresh(article)

    # Start background extraction
    background_tasks.add_task(extract_article_content, article.id, db)

    return article


@router.get(
    "/{article_id}",
    response_model=ArticleResponse,
    summary="Get article by ID",
)
async def get_article(
    article_id: int,
    db: AsyncSession = Depends(get_db),
) -> Article:
    """Get article by ID."""
    article = await db.get(Article, article_id)
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article {article_id} not found",
        )
    return article


@router.get(
    "/{article_id}/status",
    response_model=ExtractionStatusResponse,
    summary="Get extraction status",
)
async def get_extraction_status(
    article_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Get extraction status for an article."""
    article = await db.get(Article, article_id)
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article {article_id} not found",
        )
    return {
        "article_id": article.id,
        "url": article.url,
        "status": article.status,
        "extraction_method": article.extraction_method,
        "error": article.extraction_error,
    }


@router.get(
    "/",
    response_model=ArticleListResponse,
    summary="List articles",
)
async def list_articles(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    status_filter: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """List articles with pagination."""
    offset = (page - 1) * page_size

    # Build query
    query = select(Article)
    count_query = select(func.count(Article.id))

    if status_filter:
        query = query.where(Article.status == status_filter)
        count_query = count_query.where(Article.status == status_filter)

    query = query.order_by(Article.created_at.desc())
    query = query.offset(offset).limit(page_size)

    # Execute
    result = await db.execute(query)
    articles = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    return {
        "items": articles,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.post(
    "/{article_id}/reextract",
    response_model=ExtractionStatusResponse,
    summary="Re-run extraction",
)
async def reextract_article(
    article_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Re-run extraction for an article."""
    article = await db.get(Article, article_id)
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article {article_id} not found",
        )

    # Reset status and trigger extraction
    article.status = ExtractionStatus.PENDING.value
    article.extraction_error = None
    await db.commit()

    background_tasks.add_task(extract_article_content, article.id, db)

    return {
        "article_id": article.id,
        "url": article.url,
        "status": article.status,
        "extraction_method": None,
        "error": None,
    }


@router.delete(
    "/{article_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete article",
)
async def delete_article(
    article_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete an article."""
    article = await db.get(Article, article_id)
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article {article_id} not found",
        )

    await db.delete(article)
    await db.commit()
```

### Step 15: Update Configuration (30 min)

Update `src/article_mind_service/config.py` to add extraction settings:

```python
# Add to Settings class:

class Settings(BaseSettings):
    # ... existing settings ...

    # Extraction configuration
    extraction_timeout_seconds: int = Field(
        default=30,
        description="Timeout for content extraction in seconds",
    )
    extraction_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for failed extractions",
    )
    extraction_user_agent: str = Field(
        default="ArticleMind/1.0 (Content Extraction Bot)",
        description="User-Agent header for HTTP requests",
    )
    playwright_headless: bool = Field(
        default=True,
        description="Run Playwright in headless mode",
    )
    extraction_max_content_size_mb: int = Field(
        default=50,
        description="Maximum content size in MB",
    )
```

Update `.env.example`:

```env
# Extraction configuration
EXTRACTION_TIMEOUT_SECONDS=30
EXTRACTION_MAX_RETRIES=3
EXTRACTION_USER_AGENT="ArticleMind/1.0 (Content Extraction Bot)"
PLAYWRIGHT_HEADLESS=true
EXTRACTION_MAX_CONTENT_SIZE_MB=50
```

### Step 16: Register Router (15 min)

Update `src/article_mind_service/main.py`:

```python
# Add import
from .routers import health_router, articles_router

# Add router
app.include_router(articles_router)
```

Update `src/article_mind_service/routers/__init__.py`:

```python
from .health import router as health_router
from .articles import router as articles_router

__all__ = ["health_router", "articles_router"]
```

---

## 7. Configuration

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EXTRACTION_TIMEOUT_SECONDS` | int | 30 | HTTP request timeout |
| `EXTRACTION_MAX_RETRIES` | int | 3 | Max retry attempts for failed requests |
| `EXTRACTION_USER_AGENT` | str | ArticleMind/1.0 | User-Agent header |
| `PLAYWRIGHT_HEADLESS` | bool | true | Run Playwright headless |
| `EXTRACTION_MAX_CONTENT_SIZE_MB` | int | 50 | Max content size |

### Sample .env

```env
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/articleMind

# Extraction
EXTRACTION_TIMEOUT_SECONDS=30
EXTRACTION_MAX_RETRIES=3
EXTRACTION_USER_AGENT="ArticleMind/1.0 (Content Extraction Bot)"
PLAYWRIGHT_HEADLESS=true
EXTRACTION_MAX_CONTENT_SIZE_MB=50
```

---

## 8. Background Task Processing

### Approach: Simple Async with Status Tracking

For MVP, we use FastAPI's built-in BackgroundTasks with status tracking in PostgreSQL. This avoids the complexity of Celery/Redis while providing:

- Immediate response to client (article ID + PENDING status)
- Background extraction without blocking
- Status polling via `/articles/{id}/status`
- Automatic retry on failure (via re-extract endpoint)

### Future Scaling Options

If extraction volume increases significantly:

1. **Task Queue (Celery + Redis):** For distributed processing
2. **Worker Processes:** Multiple extraction workers
3. **Priority Queues:** Different priorities for different content types

---

## 9. Testing

### Test Structure

```
tests/
├── extraction/
│   ├── __init__.py
│   ├── conftest.py              # Fixtures and test data
│   ├── test_html_extractor.py
│   ├── test_pdf_extractor.py
│   ├── test_js_extractor.py
│   ├── test_pipeline.py
│   ├── test_content_type.py
│   └── test_utils.py
├── routers/
│   └── test_articles.py
└── fixtures/
    ├── sample_article.html
    ├── sample_news.html
    ├── sample_document.pdf
    └── js_rendered_page.html
```

### Sample Test Cases

```python
# tests/extraction/test_html_extractor.py

import pytest
from article_mind_service.extraction import HTMLExtractor, ExtractionResult


@pytest.fixture
def html_extractor():
    return HTMLExtractor()


@pytest.fixture
def sample_article_html():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Test Article</title></head>
    <body>
        <nav>Navigation links</nav>
        <article>
            <h1>Test Article Title</h1>
            <p class="author">By John Doe</p>
            <div class="content">
                <p>This is the main article content.</p>
                <p>It has multiple paragraphs with meaningful text.</p>
                <p>The content extraction should capture this text.</p>
            </div>
        </article>
        <footer>Footer content</footer>
    </body>
    </html>
    """


@pytest.mark.asyncio
async def test_html_extraction_returns_content(
    html_extractor: HTMLExtractor,
    sample_article_html: str,
) -> None:
    """Test that HTML extractor returns content."""
    result = await html_extractor.extract(sample_article_html, "https://example.com")

    assert isinstance(result, ExtractionResult)
    assert len(result.content) > 100
    assert "main article content" in result.content.lower()


@pytest.mark.asyncio
async def test_html_extraction_removes_boilerplate(
    html_extractor: HTMLExtractor,
    sample_article_html: str,
) -> None:
    """Test that boilerplate is removed."""
    result = await html_extractor.extract(sample_article_html, "https://example.com")

    assert "navigation links" not in result.content.lower()
    assert "footer content" not in result.content.lower()


@pytest.mark.asyncio
async def test_html_extraction_extracts_title(
    html_extractor: HTMLExtractor,
    sample_article_html: str,
) -> None:
    """Test that title is extracted."""
    result = await html_extractor.extract(sample_article_html, "https://example.com")

    assert result.title is not None
    assert "test article" in result.title.lower()


@pytest.mark.asyncio
async def test_html_extraction_calculates_word_count(
    html_extractor: HTMLExtractor,
    sample_article_html: str,
) -> None:
    """Test that word count is calculated."""
    result = await html_extractor.extract(sample_article_html, "https://example.com")

    assert result.word_count > 0


@pytest.mark.asyncio
async def test_html_extraction_empty_content_raises(
    html_extractor: HTMLExtractor,
) -> None:
    """Test that empty content raises EmptyContentError."""
    from article_mind_service.extraction import EmptyContentError

    empty_html = "<html><body></body></html>"

    with pytest.raises(EmptyContentError):
        await html_extractor.extract(empty_html, "https://example.com")
```

### Integration Tests

```python
# tests/routers/test_articles.py

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_article(async_client: AsyncClient) -> None:
    """Test creating an article triggers extraction."""
    response = await async_client.post(
        "/api/v1/articles/",
        json={"url": "https://example.com/test-article"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["url"] == "https://example.com/test-article"
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_get_article(async_client: AsyncClient) -> None:
    """Test getting article by ID."""
    # Create article first
    create_response = await async_client.post(
        "/api/v1/articles/",
        json={"url": "https://example.com/test-get"},
    )
    article_id = create_response.json()["id"]

    # Get article
    response = await async_client.get(f"/api/v1/articles/{article_id}")

    assert response.status_code == 200
    assert response.json()["id"] == article_id


@pytest.mark.asyncio
async def test_list_articles(async_client: AsyncClient) -> None:
    """Test listing articles with pagination."""
    response = await async_client.get("/api/v1/articles/?page=1&page_size=10")

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data


@pytest.mark.asyncio
async def test_get_extraction_status(async_client: AsyncClient) -> None:
    """Test getting extraction status."""
    # Create article
    create_response = await async_client.post(
        "/api/v1/articles/",
        json={"url": "https://example.com/test-status"},
    )
    article_id = create_response.json()["id"]

    # Get status
    response = await async_client.get(f"/api/v1/articles/{article_id}/status")

    assert response.status_code == 200
    data = response.json()
    assert data["article_id"] == article_id
    assert data["status"] in ["pending", "processing", "completed", "failed"]
```

---

## 10. Acceptance Criteria

### Functional Requirements

- [ ] **F1:** POST to `/api/v1/articles/` with URL creates article and starts extraction
- [ ] **F2:** GET `/api/v1/articles/{id}` returns article with extracted content
- [ ] **F3:** GET `/api/v1/articles/{id}/status` returns extraction status
- [ ] **F4:** HTML pages are extracted using trafilatura with newspaper4k fallback
- [ ] **F5:** PDF documents are extracted using PyMuPDF/pymupdf4llm
- [ ] **F6:** JS-rendered pages are extracted using Playwright when static fails
- [ ] **F7:** Content type is auto-detected from URL and Content-Type header
- [ ] **F8:** Extraction runs as background task (non-blocking)
- [ ] **F9:** Failed extractions can be re-triggered via `/reextract` endpoint

### Non-Functional Requirements

- [ ] **NF1:** Extraction timeout configurable via environment variable
- [ ] **NF2:** Retry logic with exponential backoff for transient failures
- [ ] **NF3:** Rate limit handling (429 responses)
- [ ] **NF4:** Content size limit enforced
- [ ] **NF5:** All extraction results stored with metadata
- [ ] **NF6:** Clean text with normalized whitespace
- [ ] **NF7:** Word count and reading time calculated

### Test Requirements

- [ ] **T1:** Unit tests for each extractor (HTML, PDF, JS)
- [ ] **T2:** Unit tests for content type detection
- [ ] **T3:** Unit tests for text cleaning utilities
- [ ] **T4:** Integration tests for article endpoints
- [ ] **T5:** Test fixtures for HTML, PDF content
- [ ] **T6:** All tests pass in CI

### Documentation Requirements

- [ ] **D1:** Extraction module has docstrings
- [ ] **D2:** API endpoints documented in OpenAPI
- [ ] **D3:** Configuration documented in .env.example
- [ ] **D4:** README updated with extraction usage

---

## 11. Verification Steps

### 1. Run Unit Tests

```bash
cd article-mind-service
make test
```

**Expected:** All tests pass, including new extraction tests.

### 2. Start Dev Server

```bash
make dev
```

### 3. Test Article Creation

```bash
curl -X POST http://localhost:8000/api/v1/articles/ \
  -H "Content-Type: application/json" \
  -d '{"url": "https://en.wikipedia.org/wiki/Python_(programming_language)"}'
```

**Expected:** 201 Created with article ID and PENDING status.

### 4. Check Extraction Status

```bash
curl http://localhost:8000/api/v1/articles/1/status
```

**Expected:** Status transitions from PENDING -> PROCESSING -> COMPLETED.

### 5. Get Extracted Content

```bash
curl http://localhost:8000/api/v1/articles/1
```

**Expected:** Article with extracted content, title, word count.

### 6. Test PDF Extraction

```bash
curl -X POST http://localhost:8000/api/v1/articles/ \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.w3.org/WAI/WCAG21/Techniques/pdf/PDF1.pdf"}'
```

**Expected:** PDF extracted with markdown content.

### 7. Verify OpenAPI Documentation

Visit: http://localhost:8000/docs

**Expected:** Article endpoints documented with schemas.

---

## 12. Common Pitfalls

### Issue 1: Playwright Browser Not Installed

**Symptom:** JS extraction fails with "Browser not found"

**Solution:**
```bash
playwright install chromium
```

### Issue 2: trafilatura Returns Empty Content

**Symptom:** Extraction returns empty for some pages

**Solution:** Fallback to newspaper4k or JS rendering is working correctly. Check logs for which fallback was used.

### Issue 3: PDF Extraction Memory Issues

**Symptom:** High memory usage with large PDFs

**Solution:** Check `EXTRACTION_MAX_CONTENT_SIZE_MB` limit. Consider chunking large PDFs.

### Issue 4: Rate Limiting

**Symptom:** Extraction fails with 429 status

**Solution:** Exponential backoff is implemented. Adjust `EXTRACTION_MAX_RETRIES` if needed.

### Issue 5: Encoding Issues

**Symptom:** Content has garbled characters

**Solution:** charset-normalizer handles most cases. Check content encoding detection logs.

---

## 13. Next Steps

After completing this plan:

1. **Plan 05:** Chunking - Split extracted content into chunks with heading awareness
2. **Plan 06:** Embeddings - Generate embeddings for chunks
3. **Plan 07:** Indexing - Index chunks in mcp-vector-search
4. **Plan 08:** Q&A - Implement grounded Q&A with citations

---

## 14. References

- Research: `docs/research/python-content-extraction-libraries-2025.md`
- Implementation Plan: `docs/research/implementation_plan.md` (Phase 1-2)
- trafilatura docs: https://trafilatura.readthedocs.io/
- PyMuPDF docs: https://pymupdf.readthedocs.io/
- Playwright Python docs: https://playwright.dev/python/

---

**Plan Status:** Ready for implementation
**Last Updated:** 2026-01-19
