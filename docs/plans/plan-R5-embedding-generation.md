# Plan R5: Embedding Generation Pipeline

**Plan ID:** R5-embedding-generation
**Created:** 2026-01-19
**Dependencies:** Plan 03 (database-setup), Plan 04 (service-health-api)
**Estimated Time:** 6-8 hours
**Research Reference:** `docs/research/embedding-models-research-2025-2026.md`

---

## 1. Overview

### Purpose

Implement a production-ready embedding generation pipeline that transforms extracted article text into vector embeddings for semantic search. The system supports dual-mode operation with OpenAI cloud API and Ollama local inference, enabling both development flexibility and production scalability.

### Scope

- Abstract embedding provider interface supporting multiple backends
- OpenAI text-embedding-3-small provider implementation (cloud)
- Ollama nomic-embed-text provider implementation (local)
- Text chunking with langchain-text-splitters
- ChromaDB vector storage integration
- Database schema updates for embedding status tracking
- Configuration via environment variables
- Comprehensive testing with mock providers

### Dependencies

- **Plan 03:** Requires PostgreSQL database for article status tracking
- **Plan 04:** Follows established FastAPI patterns from health check

### Outputs

- Abstract `EmbeddingProvider` base class
- `OpenAIEmbeddingProvider` implementation (1536 dimensions)
- `OllamaEmbeddingProvider` implementation (1024 dimensions)
- `TextChunker` class using RecursiveCharacterTextSplitter
- `EmbeddingPipeline` orchestrator
- ChromaDB collection management
- Database migration adding embedding status fields
- Configuration schema for embedding settings
- Unit and integration tests

---

## 2. Technology Stack

### Embedding Providers

| Provider | Model | Dimensions | Cost | Speed | Use Case |
|----------|-------|-----------|------|-------|----------|
| **OpenAI** | text-embedding-3-small | 1536 | $0.02/1M tokens | API latency | Production, cloud |
| **Ollama** | nomic-embed-text | 1024 | Free | 12,450 tok/s (GPU) | Development, privacy |

### Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `openai` | 1.60+ | OpenAI API client |
| `ollama` | 0.4+ | Ollama Python SDK |
| `chromadb` | 0.5+ | Vector database |
| `langchain-text-splitters` | 0.3+ | Text chunking |

### Key Research Findings

From `docs/research/embedding-models-research-2025-2026.md`:

**OpenAI text-embedding-3-small:**
- MTEB Score: 62.3%
- Context Window: 8,192 tokens
- 5x cost reduction over ada-002
- Best cost-performance balance for production

**Ollama nomic-embed-text:**
- Surpasses OpenAI ada-002 and text-embedding-3-small on short/long context
- 8,192 token context window
- 12,450 tokens/sec on RTX 4090
- 0.5GB memory footprint
- Completely free, local operation

---

## 3. Architecture Design

### High-Level Flow

```
Article Text
     │
     ▼
┌─────────────────┐
│   TextChunker   │  RecursiveCharacterTextSplitter
│   (chunking)    │  512 tokens, 50 overlap
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EmbeddingProvider│  OpenAI or Ollama (configurable)
│   (embedding)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    ChromaDB     │  Collection per session
│   (storage)     │  Persistent, with metadata
└─────────────────┘
```

### Provider Abstraction

```python
from abc import ABC, abstractmethod
from typing import Protocol


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Design Decision: Use ABC over Protocol because we need shared
    implementation logic (retry handling, batching) in the base class.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of embeddings produced by this provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier for this provider."""
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Return the maximum context length in tokens."""
        pass
```

---

## 4. Module Structure

```
article-mind-service/
└── src/article_mind_service/
    ├── embeddings/                 # NEW: Embedding pipeline module
    │   ├── __init__.py            # Public exports
    │   ├── base.py                # EmbeddingProvider ABC
    │   ├── openai_provider.py     # OpenAI implementation
    │   ├── ollama_provider.py     # Ollama implementation
    │   ├── chunker.py             # TextChunker (langchain)
    │   ├── pipeline.py            # EmbeddingPipeline orchestrator
    │   ├── chromadb_store.py      # ChromaDB integration
    │   └── exceptions.py          # Custom exceptions
    ├── models/
    │   └── article.py             # UPDATED: Add embedding status fields
    ├── schemas/
    │   └── embedding.py           # NEW: Embedding-related schemas
    └── config.py                  # UPDATED: Add embedding configuration
```

---

## 5. Chunking Strategy

### RecursiveCharacterTextSplitter Configuration

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """Text chunking using RecursiveCharacterTextSplitter.

    Design Decisions:

    1. Chunk Size (512 tokens):
       - Balances context preservation with embedding quality
       - Both OpenAI and Ollama support 8,192 tokens, but smaller
         chunks enable more precise retrieval
       - Research shows 256-512 optimal for RAG

    2. Overlap (50 tokens, ~10%):
       - Preserves context across chunk boundaries
       - Prevents information loss at splits
       - Standard practice in production RAG systems

    3. Separators:
       - Prioritize paragraph boundaries (\n\n)
       - Fall back to sentences (. ! ?)
       - Last resort: words and characters
       - Preserves semantic coherence within chunks
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        length_function: callable = len,  # Will use tiktoken in implementation
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[
                "\n\n",      # Paragraph boundaries (highest priority)
                "\n",        # Line breaks
                ". ",        # Sentence endings
                "! ",
                "? ",
                "; ",        # Clause boundaries
                ", ",
                " ",         # Word boundaries
                "",          # Character-level (last resort)
            ],
            keep_separator=True,
        )

    def chunk(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: The full text to chunk.

        Returns:
            List of text chunks.
        """
        return self.splitter.split_text(text)

    def chunk_with_metadata(
        self,
        text: str,
        source_metadata: dict
    ) -> list[dict]:
        """Split text into chunks with position metadata.

        Args:
            text: The full text to chunk.
            source_metadata: Metadata to attach to each chunk.

        Returns:
            List of dicts with 'text', 'chunk_index', and source metadata.
        """
        chunks = self.splitter.split_text(text)
        return [
            {
                "text": chunk,
                "chunk_index": i,
                **source_metadata,
            }
            for i, chunk in enumerate(chunks)
        ]
```

### Token Counting with tiktoken

```python
import tiktoken


def get_token_counter(model: str = "cl100k_base") -> callable:
    """Get a token counting function for the specified model.

    Args:
        model: The tiktoken encoding to use.
               - cl100k_base: OpenAI text-embedding-3-* models
               - gpt2: Fallback for Ollama (approximate)

    Returns:
        Function that counts tokens in a string.
    """
    encoding = tiktoken.get_encoding(model)
    return lambda text: len(encoding.encode(text))
```

---

## 6. Embedding Provider Implementations

### OpenAI Provider

```python
# src/article_mind_service/embeddings/openai_provider.py
"""OpenAI embedding provider implementation."""

from openai import AsyncOpenAI

from .base import EmbeddingProvider
from .exceptions import EmbeddingError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text-embedding-3-small provider.

    Model Details:
        - Dimensions: 1536
        - Max tokens: 8,192
        - Cost: $0.02 per 1M tokens
        - MTEB Score: 62.3%

    Design Decisions:

    1. Async Client:
       - Uses AsyncOpenAI for non-blocking I/O
       - Integrates with FastAPI's async ecosystem

    2. Batch Processing:
       - OpenAI API supports batching (up to 2048 inputs)
       - Reduces API calls and latency
       - Implement chunked batching if > 2048 texts

    3. Error Handling:
       - Retry with exponential backoff on rate limits
       - Wrap API errors in EmbeddingError
    """

    MODEL = "text-embedding-3-small"
    DIMENSIONS = 1536
    MAX_TOKENS = 8192
    MAX_BATCH_SIZE = 2048

    def __init__(self, api_key: str):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (from environment).
        """
        self.client = AsyncOpenAI(api_key=api_key)

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS

    @property
    def model_name(self) -> str:
        return self.MODEL

    @property
    def max_tokens(self) -> int:
        return self.MAX_TOKENS

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors (1536 dimensions each).

        Raises:
            EmbeddingError: If API call fails.
        """
        if not texts:
            return []

        try:
            # Process in batches if necessary
            all_embeddings = []
            for i in range(0, len(texts), self.MAX_BATCH_SIZE):
                batch = texts[i:i + self.MAX_BATCH_SIZE]
                response = await self.client.embeddings.create(
                    model=self.MODEL,
                    input=batch,
                )
                batch_embeddings = [
                    item.embedding for item in response.data
                ]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}") from e
```

### Ollama Provider

```python
# src/article_mind_service/embeddings/ollama_provider.py
"""Ollama embedding provider implementation."""

import ollama
from ollama import AsyncClient

from .base import EmbeddingProvider
from .exceptions import EmbeddingError


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama nomic-embed-text provider.

    Model Details:
        - Dimensions: 1024
        - Max tokens: 8,192
        - Speed: 12,450 tokens/sec (RTX 4090)
        - Memory: 0.5GB
        - Cost: Free (local)

    Design Decisions:

    1. Local-First:
       - No API keys required
       - Complete data privacy
       - Works offline

    2. Async Client:
       - Uses ollama.AsyncClient
       - Ollama server must be running locally

    3. Sequential Processing:
       - Ollama API embeds one text at a time
       - Use asyncio.gather for parallel requests
       - Batch size configurable for memory management
    """

    MODEL = "nomic-embed-text"
    DIMENSIONS = 1024
    MAX_TOKENS = 8192
    DEFAULT_BATCH_SIZE = 32  # Process in parallel batches

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """Initialize Ollama provider.

        Args:
            base_url: Ollama server URL.
            model: Model name (default: nomic-embed-text).
            batch_size: Number of texts to process in parallel.
        """
        self.client = AsyncClient(host=base_url)
        self.model = model
        self.batch_size = batch_size

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def max_tokens(self) -> int:
        return self.MAX_TOKENS

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Ollama.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors (1024 dimensions each).

        Raises:
            EmbeddingError: If Ollama API call fails.
        """
        import asyncio

        if not texts:
            return []

        try:
            all_embeddings = []

            # Process in batches for memory efficiency
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]

                # Parallel embedding within batch
                tasks = [
                    self.client.embeddings(
                        model=self.model,
                        prompt=text,
                    )
                    for text in batch
                ]
                responses = await asyncio.gather(*tasks)

                batch_embeddings = [
                    response["embedding"] for response in responses
                ]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            raise EmbeddingError(f"Ollama embedding failed: {e}") from e

    async def health_check(self) -> bool:
        """Check if Ollama server is available.

        Returns:
            True if server is reachable and model is available.
        """
        try:
            # Check if model is available
            models = await self.client.list()
            model_names = [m["name"] for m in models.get("models", [])]
            return self.model in model_names or f"{self.model}:latest" in model_names
        except Exception:
            return False
```

---

## 7. ChromaDB Integration

### Collection Management

```python
# src/article_mind_service/embeddings/chromadb_store.py
"""ChromaDB vector storage integration."""

import chromadb
from chromadb.config import Settings as ChromaSettings

from .base import EmbeddingProvider


class ChromaDBStore:
    """ChromaDB vector store for embedding storage.

    Design Decisions:

    1. Collection per Session:
       - Isolates embeddings by session for multi-tenant use
       - Collection name: session_{session_id}
       - Enables easy cleanup when session deleted

    2. Persistent Storage:
       - Uses DuckDB + Parquet backend
       - Path configurable via CHROMADB_PATH
       - Survives service restarts

    3. Metadata Strategy:
       - article_id: Reference back to PostgreSQL
       - chunk_index: Position in original document
       - source_url: Original article URL
       - Enables filtered retrieval

    4. Dimension Handling:
       - ChromaDB auto-detects dimensions from first insert
       - Must be consistent within collection
       - Store dimension info in metadata for validation
    """

    def __init__(
        self,
        persist_path: str = "./data/chromadb",
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """Initialize ChromaDB store.

        Args:
            persist_path: Directory for persistent storage.
            embedding_provider: Optional provider for query embedding.
        """
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        self.embedding_provider = embedding_provider

    def get_or_create_collection(
        self,
        session_id: str,
        dimensions: int,
    ) -> chromadb.Collection:
        """Get or create a collection for a session.

        Args:
            session_id: Unique session identifier.
            dimensions: Expected embedding dimensions.

        Returns:
            ChromaDB Collection instance.
        """
        collection_name = f"session_{session_id}"

        # Create with metadata to track dimensions
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "dimensions": dimensions,
                "session_id": session_id,
            },
        )

    def add_embeddings(
        self,
        collection: chromadb.Collection,
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """Add embeddings to collection.

        Args:
            collection: Target ChromaDB collection.
            embeddings: List of embedding vectors.
            texts: Original text chunks (stored for retrieval).
            metadatas: Metadata for each chunk.
            ids: Unique IDs for each chunk.
        """
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

    def query(
        self,
        collection: chromadb.Collection,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> dict:
        """Query collection for similar embeddings.

        Args:
            collection: ChromaDB collection to query.
            query_embedding: Embedding vector for query.
            n_results: Number of results to return.
            where: Optional metadata filter.

        Returns:
            Query results with ids, documents, distances, metadatas.
        """
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def delete_collection(self, session_id: str) -> None:
        """Delete a session's collection.

        Args:
            session_id: Session to delete embeddings for.
        """
        collection_name = f"session_{session_id}"
        try:
            self.client.delete_collection(collection_name)
        except ValueError:
            # Collection doesn't exist, ignore
            pass

    def get_collection_stats(self, session_id: str) -> dict:
        """Get statistics for a session's collection.

        Args:
            session_id: Session to get stats for.

        Returns:
            Dict with count, dimensions, etc.
        """
        collection_name = f"session_{session_id}"
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "count": collection.count(),
                "metadata": collection.metadata,
            }
        except ValueError:
            return {"name": collection_name, "count": 0, "exists": False}
```

---

## 8. Configuration

### Environment Variables

```env
# .env additions

# Embedding Provider Selection
EMBEDDING_PROVIDER=openai  # openai | ollama

# OpenAI Configuration
OPENAI_API_KEY=sk-...

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text

# ChromaDB Configuration
CHROMADB_PATH=./data/chromadb

# Chunking Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

### Pydantic Settings Update

```python
# src/article_mind_service/config.py (additions)

from typing import Literal


class Settings(BaseSettings):
    # ... existing settings ...

    # Embedding Provider
    embedding_provider: Literal["openai", "ollama"] = "openai"

    # OpenAI
    openai_api_key: str | None = None

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"

    # ChromaDB
    chromadb_path: str = "./data/chromadb"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    @property
    def embedding_dimensions(self) -> int:
        """Get dimensions based on configured provider."""
        if self.embedding_provider == "openai":
            return 1536
        return 1024  # ollama nomic-embed-text
```

---

## 9. Processing Pipeline

### Pipeline Orchestrator

```python
# src/article_mind_service/embeddings/pipeline.py
"""Embedding pipeline orchestrator."""

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.config import settings

from .base import EmbeddingProvider
from .chunker import TextChunker
from .chromadb_store import ChromaDBStore
from .exceptions import EmbeddingError


class EmbeddingPipeline:
    """Orchestrates text chunking, embedding, and storage.

    Pipeline Flow:
        1. Receive article text and metadata
        2. Chunk text using TextChunker
        3. Generate embeddings in batches
        4. Store in ChromaDB with metadata
        5. Update article status in PostgreSQL

    Design Decisions:

    1. Batch Processing:
       - Process in batches of 100 chunks
       - Balance memory usage and efficiency
       - Progress updates after each batch

    2. Error Recovery:
       - Store progress in database
       - Resume from last successful batch
       - Mark failed articles for retry

    3. Status Tracking:
       - pending: Article queued for embedding
       - processing: Currently generating embeddings
       - completed: All chunks embedded
       - failed: Error during processing
    """

    BATCH_SIZE = 100

    def __init__(
        self,
        provider: EmbeddingProvider,
        store: ChromaDBStore,
        chunker: TextChunker,
    ):
        """Initialize pipeline.

        Args:
            provider: Embedding provider (OpenAI or Ollama).
            store: ChromaDB store instance.
            chunker: Text chunking instance.
        """
        self.provider = provider
        self.store = store
        self.chunker = chunker

    async def process_article(
        self,
        article_id: int,
        session_id: str,
        text: str,
        source_url: str,
        db: AsyncSession,
    ) -> int:
        """Process a single article through the embedding pipeline.

        Args:
            article_id: Database ID of the article.
            session_id: Session this article belongs to.
            text: Extracted article text.
            source_url: Original article URL.
            db: Database session for status updates.

        Returns:
            Number of chunks created.

        Raises:
            EmbeddingError: If processing fails.
        """
        # Update status to processing
        await self._update_status(db, article_id, "processing")

        try:
            # Step 1: Chunk the text
            chunks = self.chunker.chunk_with_metadata(
                text,
                source_metadata={
                    "article_id": article_id,
                    "source_url": source_url,
                },
            )

            if not chunks:
                await self._update_status(
                    db, article_id, "completed", chunk_count=0
                )
                return 0

            # Step 2: Get or create collection
            collection = self.store.get_or_create_collection(
                session_id=session_id,
                dimensions=self.provider.dimensions,
            )

            # Step 3: Process in batches
            total_chunks = len(chunks)
            for batch_start in range(0, total_chunks, self.BATCH_SIZE):
                batch_end = min(batch_start + self.BATCH_SIZE, total_chunks)
                batch = chunks[batch_start:batch_end]

                # Extract texts for embedding
                batch_texts = [c["text"] for c in batch]

                # Generate embeddings
                embeddings = await self.provider.embed(batch_texts)

                # Prepare metadata and IDs
                ids = [
                    f"article_{article_id}_chunk_{c['chunk_index']}"
                    for c in batch
                ]
                metadatas = [
                    {
                        "article_id": c["article_id"],
                        "chunk_index": c["chunk_index"],
                        "source_url": c["source_url"],
                    }
                    for c in batch
                ]

                # Store in ChromaDB
                self.store.add_embeddings(
                    collection=collection,
                    embeddings=embeddings,
                    texts=batch_texts,
                    metadatas=metadatas,
                    ids=ids,
                )

            # Update status to completed
            await self._update_status(
                db, article_id, "completed", chunk_count=total_chunks
            )

            return total_chunks

        except Exception as e:
            await self._update_status(db, article_id, "failed")
            raise EmbeddingError(f"Pipeline failed for article {article_id}: {e}") from e

    async def _update_status(
        self,
        db: AsyncSession,
        article_id: int,
        status: str,
        chunk_count: int | None = None,
    ) -> None:
        """Update article embedding status in database.

        Args:
            db: Database session.
            article_id: Article to update.
            status: New embedding status.
            chunk_count: Optional chunk count to set.
        """
        from sqlalchemy import update
        from article_mind_service.models.article import Article

        values = {"embedding_status": status}
        if chunk_count is not None:
            values["chunk_count"] = chunk_count

        stmt = (
            update(Article)
            .where(Article.id == article_id)
            .values(**values)
        )
        await db.execute(stmt)
        await db.commit()
```

### Factory Function

```python
# src/article_mind_service/embeddings/__init__.py
"""Embedding module public exports."""

from .base import EmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
from .ollama_provider import OllamaEmbeddingProvider
from .chunker import TextChunker
from .chromadb_store import ChromaDBStore
from .pipeline import EmbeddingPipeline
from .exceptions import EmbeddingError

from article_mind_service.config import settings


def get_embedding_provider() -> EmbeddingProvider:
    """Factory function to get configured embedding provider.

    Returns:
        EmbeddingProvider instance based on EMBEDDING_PROVIDER setting.

    Raises:
        ValueError: If provider is not configured correctly.
    """
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI provider")
        return OpenAIEmbeddingProvider(api_key=settings.openai_api_key)

    elif settings.embedding_provider == "ollama":
        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )

    else:
        raise ValueError(f"Unknown provider: {settings.embedding_provider}")


def get_embedding_pipeline() -> EmbeddingPipeline:
    """Factory function to create configured pipeline.

    Returns:
        EmbeddingPipeline ready for use.
    """
    provider = get_embedding_provider()
    store = ChromaDBStore(
        persist_path=settings.chromadb_path,
        embedding_provider=provider,
    )
    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return EmbeddingPipeline(
        provider=provider,
        store=store,
        chunker=chunker,
    )


__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "TextChunker",
    "ChromaDBStore",
    "EmbeddingPipeline",
    "EmbeddingError",
    "get_embedding_provider",
    "get_embedding_pipeline",
]
```

---

## 10. Database Schema Updates

### Article Model Updates

```python
# src/article_mind_service/models/article.py
"""Article model with embedding status tracking."""

from datetime import datetime
from typing import Literal

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from article_mind_service.database import Base


class Article(Base):
    """Article model for storing processed articles.

    Embedding Status Values:
        - pending: Queued for embedding generation
        - processing: Currently being embedded
        - completed: All chunks successfully embedded
        - failed: Embedding generation failed
    """

    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    url: Mapped[str] = mapped_column(String(2048))
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Embedding status tracking
    embedding_status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        index=True,
    )
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:
        return f"<Article(id={self.id}, url={self.url[:50]}, status={self.embedding_status})>"
```

### Alembic Migration

```python
# alembic/versions/xxxx_add_embedding_fields.py
"""add embedding status fields to articles

Revision ID: xxxx
Revises: 608494810fe8
Create Date: 2026-01-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'xxxx'  # Will be auto-generated
down_revision: Union[str, Sequence[str], None] = '608494810fe8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add articles table with embedding status fields."""
    op.create_table(
        'articles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.String(length=36), nullable=False),
        sa.Column('url', sa.String(length=2048), nullable=False),
        sa.Column('title', sa.String(length=512), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('embedding_status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.Column('chunk_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_articles_session_id'), 'articles', ['session_id'], unique=False)
    op.create_index(op.f('ix_articles_embedding_status'), 'articles', ['embedding_status'], unique=False)


def downgrade() -> None:
    """Remove articles table."""
    op.drop_index(op.f('ix_articles_embedding_status'), table_name='articles')
    op.drop_index(op.f('ix_articles_session_id'), table_name='articles')
    op.drop_table('articles')
```

---

## 11. Implementation Steps

### Phase 1: Core Infrastructure (Steps 1-4)

#### Step 1: Add Dependencies

Update `pyproject.toml`:

```toml
dependencies = [
    # ... existing dependencies ...
    "openai>=1.60.0",
    "ollama>=0.4.0",
    "chromadb>=0.5.0",
    "langchain-text-splitters>=0.3.0",
    "tiktoken>=0.8.0",
]
```

Install:

```bash
cd article-mind-service
make install
```

#### Step 2: Create Module Structure

```bash
mkdir -p src/article_mind_service/embeddings
touch src/article_mind_service/embeddings/__init__.py
touch src/article_mind_service/embeddings/base.py
touch src/article_mind_service/embeddings/exceptions.py
touch src/article_mind_service/embeddings/openai_provider.py
touch src/article_mind_service/embeddings/ollama_provider.py
touch src/article_mind_service/embeddings/chunker.py
touch src/article_mind_service/embeddings/chromadb_store.py
touch src/article_mind_service/embeddings/pipeline.py
```

#### Step 3: Implement Base Classes

1. Create `exceptions.py` with `EmbeddingError`
2. Create `base.py` with `EmbeddingProvider` ABC

#### Step 4: Update Configuration

1. Add embedding settings to `config.py`
2. Update `.env.example` with new variables

### Phase 2: Provider Implementations (Steps 5-7)

#### Step 5: Implement OpenAI Provider

1. Create `openai_provider.py`
2. Add unit tests in `tests/unit/test_openai_provider.py`
3. Verify with manual test (requires API key)

#### Step 6: Implement Ollama Provider

1. Create `ollama_provider.py`
2. Add unit tests in `tests/unit/test_ollama_provider.py`
3. Add health check method
4. Verify with local Ollama (if available)

#### Step 7: Implement Text Chunker

1. Create `chunker.py` with `TextChunker`
2. Add tiktoken token counting
3. Add unit tests in `tests/unit/test_chunker.py`

### Phase 3: Storage and Pipeline (Steps 8-10)

#### Step 8: Implement ChromaDB Store

1. Create `chromadb_store.py`
2. Implement collection management
3. Add unit tests in `tests/unit/test_chromadb_store.py`

#### Step 9: Create Database Migration

1. Create Article model in `models/article.py`
2. Import in `models/__init__.py`
3. Generate migration: `make migrate-create MSG="add articles with embedding status"`
4. Apply migration: `make migrate`

#### Step 10: Implement Pipeline Orchestrator

1. Create `pipeline.py`
2. Implement `process_article` method
3. Add status tracking integration

### Phase 4: Integration and Testing (Steps 11-13)

#### Step 11: Create Module Exports

1. Update `embeddings/__init__.py` with all exports
2. Add factory functions

#### Step 12: Write Integration Tests

1. Create `tests/integration/test_embedding_pipeline.py`
2. Test full pipeline with mock providers
3. Test ChromaDB integration

#### Step 13: Verify and Document

1. Run full test suite: `make test`
2. Update `.env.example`
3. Update CLAUDE.md with embedding module documentation

---

## 12. Testing Strategy

### Unit Tests

#### Test Provider Mock

```python
# tests/unit/test_embedding_providers.py
"""Unit tests for embedding providers."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from article_mind_service.embeddings import (
    OpenAIEmbeddingProvider,
    OllamaEmbeddingProvider,
    EmbeddingError,
)


class TestOpenAIProvider:
    """Tests for OpenAI embedding provider."""

    @pytest.fixture
    def provider(self) -> OpenAIEmbeddingProvider:
        """Create provider with test API key."""
        return OpenAIEmbeddingProvider(api_key="test-key")

    def test_dimensions(self, provider: OpenAIEmbeddingProvider) -> None:
        """Test dimensions property."""
        assert provider.dimensions == 1536

    def test_model_name(self, provider: OpenAIEmbeddingProvider) -> None:
        """Test model_name property."""
        assert provider.model_name == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, provider: OpenAIEmbeddingProvider) -> None:
        """Test embedding empty list returns empty list."""
        result = await provider.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_single_text(self, provider: OpenAIEmbeddingProvider) -> None:
        """Test embedding single text."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch.object(
            provider.client.embeddings,
            'create',
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await provider.embed(["test text"])

            assert len(result) == 1
            assert len(result[0]) == 1536

    @pytest.mark.asyncio
    async def test_embed_api_error(self, provider: OpenAIEmbeddingProvider) -> None:
        """Test API error raises EmbeddingError."""
        with patch.object(
            provider.client.embeddings,
            'create',
            new_callable=AsyncMock,
            side_effect=Exception("API Error"),
        ):
            with pytest.raises(EmbeddingError) as exc_info:
                await provider.embed(["test"])

            assert "OpenAI embedding failed" in str(exc_info.value)


class TestOllamaProvider:
    """Tests for Ollama embedding provider."""

    @pytest.fixture
    def provider(self) -> OllamaEmbeddingProvider:
        """Create provider with default settings."""
        return OllamaEmbeddingProvider()

    def test_dimensions(self, provider: OllamaEmbeddingProvider) -> None:
        """Test dimensions property."""
        assert provider.dimensions == 1024

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, provider: OllamaEmbeddingProvider) -> None:
        """Test embedding empty list returns empty list."""
        result = await provider.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_health_check_model_not_found(
        self, provider: OllamaEmbeddingProvider
    ) -> None:
        """Test health check when model not found."""
        with patch.object(
            provider.client,
            'list',
            new_callable=AsyncMock,
            return_value={"models": []},
        ):
            result = await provider.health_check()
            assert result is False
```

### Integration Tests

```python
# tests/integration/test_embedding_pipeline.py
"""Integration tests for embedding pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from article_mind_service.embeddings import (
    EmbeddingPipeline,
    TextChunker,
    ChromaDBStore,
    EmbeddingProvider,
)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock provider for testing."""

    @property
    def dimensions(self) -> int:
        return 128

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def max_tokens(self) -> int:
        return 8192

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings."""
        return [[0.1] * 128 for _ in texts]


@pytest.fixture
def mock_provider() -> MockEmbeddingProvider:
    """Create mock provider."""
    return MockEmbeddingProvider()


@pytest.fixture
def chunker() -> TextChunker:
    """Create chunker with small chunks for testing."""
    return TextChunker(chunk_size=100, chunk_overlap=10)


@pytest.fixture
def store(tmp_path) -> ChromaDBStore:
    """Create store with temp directory."""
    return ChromaDBStore(persist_path=str(tmp_path / "chromadb"))


@pytest.fixture
def pipeline(
    mock_provider: MockEmbeddingProvider,
    store: ChromaDBStore,
    chunker: TextChunker,
) -> EmbeddingPipeline:
    """Create pipeline with mock components."""
    return EmbeddingPipeline(
        provider=mock_provider,
        store=store,
        chunker=chunker,
    )


class TestEmbeddingPipeline:
    """Tests for embedding pipeline."""

    @pytest.mark.asyncio
    async def test_process_article_creates_chunks(
        self,
        pipeline: EmbeddingPipeline,
        mock_db_session: AsyncMock,
    ) -> None:
        """Test processing article creates chunks in ChromaDB."""
        text = "This is a test article. " * 50  # ~200 words

        chunk_count = await pipeline.process_article(
            article_id=1,
            session_id="test-session",
            text=text,
            source_url="https://example.com/article",
            db=mock_db_session,
        )

        assert chunk_count > 0

        # Verify chunks in ChromaDB
        stats = pipeline.store.get_collection_stats("test-session")
        assert stats["count"] == chunk_count

    @pytest.mark.asyncio
    async def test_process_empty_text(
        self,
        pipeline: EmbeddingPipeline,
        mock_db_session: AsyncMock,
    ) -> None:
        """Test processing empty text returns 0 chunks."""
        chunk_count = await pipeline.process_article(
            article_id=1,
            session_id="test-session",
            text="",
            source_url="https://example.com",
            db=mock_db_session,
        )

        assert chunk_count == 0
```

### Test Coverage Requirements

- **Unit Tests:** 95%+ coverage for all modules
- **Integration Tests:** Full pipeline flow
- **Mock Strategy:** Use mock providers to avoid API costs
- **Edge Cases:** Empty text, very long text, special characters

---

## 13. Acceptance Criteria

### Functional Requirements

- [ ] `EmbeddingProvider` ABC defined with required methods
- [ ] `OpenAIEmbeddingProvider` generates 1536-dim embeddings
- [ ] `OllamaEmbeddingProvider` generates 1024-dim embeddings
- [ ] `TextChunker` splits text with configurable size/overlap
- [ ] `ChromaDBStore` persists embeddings with metadata
- [ ] `EmbeddingPipeline` orchestrates full flow
- [ ] Provider configurable via `EMBEDDING_PROVIDER` env var
- [ ] Database tracks embedding status per article

### Configuration Requirements

- [ ] `.env.example` updated with all new variables
- [ ] Settings validated on startup
- [ ] Provider selection works for both openai and ollama

### Quality Requirements

- [ ] All tests pass: `make test`
- [ ] Type checking passes: `make type-check`
- [ ] Linting passes: `make lint`
- [ ] Coverage > 90% for embedding module
- [ ] No security warnings from bandit

### Documentation Requirements

- [ ] Module docstrings explain design decisions
- [ ] CLAUDE.md updated with embedding module docs
- [ ] API contract (if endpoints added) documented

### Performance Requirements

- [ ] Batch processing handles 100+ chunks efficiently
- [ ] ChromaDB queries return in < 100ms
- [ ] Pipeline processes average article in < 10 seconds

---

## 14. Common Pitfalls

### Pitfall 1: OpenAI Rate Limits

**Problem:** Too many API calls too fast causes rate limit errors.

**Solution:**
- Implement exponential backoff retry
- Use batch API (2048 max inputs)
- Add rate limiting in provider

### Pitfall 2: Ollama Not Running

**Problem:** Pipeline fails because Ollama server not started.

**Solution:**
- Add health check before processing
- Clear error message: "Ollama server not available at {url}"
- Document required setup

### Pitfall 3: Dimension Mismatch

**Problem:** Switching providers causes dimension mismatch in ChromaDB.

**Solution:**
- Store dimension info in collection metadata
- Validate dimensions before adding embeddings
- Create new collection if dimensions change

### Pitfall 4: Large Article Memory Issues

**Problem:** Very large articles cause memory issues during chunking.

**Solution:**
- Process in streaming fashion for large texts
- Set maximum article size limit
- Monitor memory usage

### Pitfall 5: ChromaDB Path Permissions

**Problem:** ChromaDB can't write to persist path.

**Solution:**
- Create directory if not exists
- Check write permissions on startup
- Use sensible default path

---

## 15. Next Steps

After completing this plan:

1. **Plan R6:** Article ingestion API endpoint
   - Accept URLs for article processing
   - Trigger embedding pipeline
   - Return processing status

2. **Plan R7:** Semantic search endpoint
   - Query ChromaDB with user query
   - Return ranked results with metadata
   - Integrate with chat API

3. **Plan R8:** Background job processing
   - Queue-based embedding generation
   - Handle large batches asynchronously
   - Progress tracking and retry logic

---

**Plan Status:** Ready for implementation
**Last Updated:** 2026-01-19
**Research Reference:** `docs/research/embedding-models-research-2025-2026.md`
