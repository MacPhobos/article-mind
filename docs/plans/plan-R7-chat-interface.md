# Plan R7: Chat Interface for Knowledge Q&A

**Plan ID:** R7-chat-interface
**Created:** 2026-01-19
**Dependencies:** R6 (Semantic Search), Plan 03 (PostgreSQL), Plan 04 (Health API patterns)
**Estimated Time:** 3-5 days

---

## 1. Overview

### Purpose
Implement a chat-like interface for querying session knowledge using a RAG (Retrieval-Augmented Generation) pipeline. Users enter questions, the system retrieves relevant content from R6's semantic search, augments a prompt with context, generates an LLM-powered answer, and returns the response with source citations.

### Scope
- RAG pipeline: Retrieve (R6) -> Augment -> Generate -> Cite
- Dual LLM provider support (OpenAI GPT-4o-mini for cost, Claude Sonnet 4.5 for quality)
- Chat history persistence in PostgreSQL
- Chat UI components in SvelteKit with Svelte 5 Runes
- Source citation display with expandable details

### Architecture Decisions
| Decision | Choice | Rationale |
|----------|--------|-----------|
| **LLM Provider** | Dual mode (OpenAI + Anthropic) | GPT-4o-mini for cost ($0.15/$0.60 per 1M tokens), Claude Sonnet 4.5 for quality |
| **Search Backend** | R6 Hybrid Search API | Leverages existing semantic + keyword retrieval infrastructure |
| **Auth Model** | Single-user | MVP scope, no multi-tenancy required |
| **Persistence** | PostgreSQL | Store Q&A history with session association for reload |
| **Orchestration** | Custom pipeline | Simple RAG flow, no framework overhead needed |

### Dependencies
- **R6 (Semantic Search):** Provides `/api/v1/sessions/{id}/search` endpoint for retrieval
- **Plan 03 (PostgreSQL):** Database for chat history persistence
- **Plan 04 (Health API):** Pydantic schema and router patterns

### Outputs
- Working chat interface at session detail page
- LLM provider abstraction (OpenAI/Anthropic)
- RAG pipeline orchestration module
- Chat history API endpoints
- Reusable chat UI components

---

## 2. RAG Pipeline Architecture

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline Flow                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   User Query                                                             │
│       │                                                                  │
│       ▼                                                                  │
│   ┌───────────────────┐                                                  │
│   │  1. RETRIEVE      │  Call R6 Search API                              │
│   │     (R6 API)      │  GET /api/v1/sessions/{id}/search?q=...          │
│   └─────────┬─────────┘  Returns: ranked chunks with metadata            │
│             │                                                            │
│             ▼                                                            │
│   ┌───────────────────┐                                                  │
│   │  2. AUGMENT       │  Build prompt with context                       │
│   │     (Prompt Eng)  │  - System instructions                           │
│   └─────────┬─────────┘  - Numbered context chunks                       │
│             │            - Citation format instructions                  │
│             ▼                                                            │
│   ┌───────────────────┐                                                  │
│   │  3. GENERATE      │  Call LLM provider                               │
│   │     (LLM API)     │  - OpenAI GPT-4o-mini (cost mode)                │
│   └─────────┬─────────┘  - Claude Sonnet 4.5 (quality mode)              │
│             │                                                            │
│             ▼                                                            │
│   ┌───────────────────┐                                                  │
│   │  4. FORMAT        │  Parse response                                  │
│   │     (Response)    │  - Extract inline citations [1], [2]             │
│   └─────────┬─────────┘  - Map citations to source metadata              │
│             │                                                            │
│             ▼                                                            │
│   Response with Citations                                                │
│   {content: "...", sources: [...]}                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Retrieval Strategy

```
Retrieval Configuration:
├── Top-K: 5 chunks (configurable via RAG_CONTEXT_CHUNKS)
├── Search Mode: Hybrid (semantic + keyword from R6)
├── Deduplication: By chunk_id (R6 handles this)
└── Diversity: Prefer chunks from different articles
```

### Context Injection Format

```
System Prompt:
├── Role: "You are a research assistant..."
├── Constraint: "Answer based ONLY on the provided context"
├── Citation: "Cite sources using [1], [2] format"
├── Uncertainty: "Say 'I don't have enough information' if unsure"
└── Format: "Be concise but thorough"

Context Block:
├── [1] {chunk_content} (Source: {title}, URL: {url})
├── [2] {chunk_content} (Source: {title}, URL: {url})
└── ...

User Message:
└── {original_question}
```

---

## 3. Database Schema

### Chat Message Model

```python
# src/article_mind_service/models/chat.py
"""Chat message models for Q&A history."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from article_mind_service.database import Base

if TYPE_CHECKING:
    from article_mind_service.models.session import Session


class ChatMessage(Base):
    """Chat message in a session Q&A conversation.

    Each message represents either a user question or an assistant response.
    Assistant responses include source citations for grounding.

    Attributes:
        id: Auto-incrementing primary key
        session_id: Foreign key to the session this chat belongs to
        role: Message role ("user" or "assistant")
        content: Message text content
        sources: JSON array of source citations (assistant messages only)
        llm_provider: Which LLM provider generated this response
        llm_model: Specific model used (e.g., "gpt-4o-mini")
        tokens_used: Total tokens consumed for this message
        created_at: When the message was created
    """
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="user or assistant",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Message text content",
    )
    sources: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
        comment="Source citations for assistant messages",
    )
    llm_provider: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="LLM provider used (openai, anthropic)",
    )
    llm_model: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Specific model identifier",
    )
    tokens_used: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Total tokens consumed",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    session: Mapped["Session"] = relationship(back_populates="chat_messages")

    def __repr__(self) -> str:
        return f"<ChatMessage(id={self.id}, role={self.role}, session_id={self.session_id})>"
```

### Sources JSON Schema

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "citation_index": { "type": "integer", "description": "Citation number [1], [2], etc." },
      "article_id": { "type": "integer" },
      "chunk_id": { "type": "string" },
      "title": { "type": "string", "nullable": true },
      "url": { "type": "string", "nullable": true },
      "excerpt": { "type": "string", "description": "Relevant excerpt from chunk" }
    },
    "required": ["citation_index", "article_id", "chunk_id"]
  }
}
```

### Migration

```bash
# Create migration
make migrate-create MSG="add chat_messages table"

# Migration file content (alembic/versions/xxx_add_chat_messages_table.py)
```

```python
"""add chat_messages table

Revision ID: xxx
Revises: previous_revision
Create Date: 2026-01-19
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'xxx'
down_revision: Union[str, None] = 'previous_revision'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'chat_messages',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(length=20), nullable=False, comment='user or assistant'),
        sa.Column('content', sa.Text(), nullable=False, comment='Message text content'),
        sa.Column('sources', postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment='Source citations for assistant messages'),
        sa.Column('llm_provider', sa.String(length=50), nullable=True, comment='LLM provider used (openai, anthropic)'),
        sa.Column('llm_model', sa.String(length=100), nullable=True, comment='Specific model identifier'),
        sa.Column('tokens_used', sa.Integer(), nullable=True, comment='Total tokens consumed'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_chat_messages_session_id'), 'chat_messages', ['session_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_chat_messages_session_id'), table_name='chat_messages')
    op.drop_table('chat_messages')
```

---

## 4. LLM Provider Abstraction

### Provider Interface

```python
# src/article_mind_service/chat/llm_providers.py
"""LLM provider abstraction for chat generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from article_mind_service.config import settings


@dataclass
class LLMResponse:
    """Response from LLM provider.

    Attributes:
        content: Generated text response
        tokens_input: Input tokens consumed
        tokens_output: Output tokens generated
        model: Model identifier used
        provider: Provider name
    """
    content: str
    tokens_input: int
    tokens_output: int
    model: str
    provider: str

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.tokens_input + self.tokens_output


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Implementations must provide async generate() method for chat completion.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier (e.g., 'openai', 'anthropic')."""
        pass

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        context_chunks: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate response using LLM.

        Args:
            system_prompt: System instructions for the model
            user_message: User's question
            context_chunks: List of context strings to include
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)

        Returns:
            LLMResponse with generated content and usage stats

        Raises:
            LLMProviderError: If generation fails
        """
        pass


class LLMProviderError(Exception):
    """Exception raised when LLM provider fails."""

    def __init__(self, provider: str, message: str, original_error: Exception | None = None):
        self.provider = provider
        self.message = message
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")
```

### OpenAI Provider

```python
# src/article_mind_service/chat/providers/openai.py
"""OpenAI LLM provider implementation."""

import openai
from openai import AsyncOpenAI

from article_mind_service.chat.llm_providers import LLMProvider, LLMProviderError, LLMResponse
from article_mind_service.config import settings


class OpenAIProvider(LLMProvider):
    """OpenAI GPT-4o-mini provider for cost-optimized generation.

    Pricing (as of 2026):
    - GPT-4o-mini: $0.15 input / $0.60 output per 1M tokens
    - 128K context window

    Best for:
    - High-volume, cost-sensitive RAG applications
    - Fast response times required
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or settings.openai_api_key
        self.model = model
        self._client: AsyncOpenAI | None = None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise LLMProviderError(
                    provider="openai",
                    message="OPENAI_API_KEY not configured"
                )
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        context_chunks: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate response using OpenAI GPT-4o-mini.

        Constructs messages in OpenAI chat format:
        - System message with instructions
        - User message with context + question
        """
        # Build context block
        context_block = self._format_context(context_chunks)

        # Construct full user message with context
        full_user_message = f"""Context:
{context_block}

Question: {user_message}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_message},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            choice = response.choices[0]
            usage = response.usage

            return LLMResponse(
                content=choice.message.content or "",
                tokens_input=usage.prompt_tokens if usage else 0,
                tokens_output=usage.completion_tokens if usage else 0,
                model=self.model,
                provider=self.provider_name,
            )

        except openai.APIError as e:
            raise LLMProviderError(
                provider="openai",
                message=f"API error: {e.message}",
                original_error=e,
            )
        except openai.AuthenticationError as e:
            raise LLMProviderError(
                provider="openai",
                message="Invalid API key",
                original_error=e,
            )
        except Exception as e:
            raise LLMProviderError(
                provider="openai",
                message=str(e),
                original_error=e,
            )

    def _format_context(self, chunks: list[str]) -> str:
        """Format context chunks with citation numbers."""
        if not chunks:
            return "No relevant context found."

        formatted = []
        for i, chunk in enumerate(chunks, start=1):
            formatted.append(f"[{i}] {chunk}")

        return "\n\n".join(formatted)
```

### Anthropic Provider

```python
# src/article_mind_service/chat/providers/anthropic.py
"""Anthropic Claude provider implementation."""

import anthropic
from anthropic import AsyncAnthropic

from article_mind_service.chat.llm_providers import LLMProvider, LLMProviderError, LLMResponse
from article_mind_service.config import settings


class AnthropicProvider(LLMProvider):
    """Anthropic Claude Sonnet 4.5 provider for quality-optimized generation.

    Pricing (as of 2026):
    - Claude Sonnet 4.5: $3.00 input / $15.00 output per 1M tokens
    - 200K context window (1M at premium rates)
    - 90% savings with prompt caching

    Best for:
    - Complex reasoning tasks
    - High-quality synthesis
    - Coding and technical content
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20241022",
    ):
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model
        self._client: AsyncAnthropic | None = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def client(self) -> AsyncAnthropic:
        """Lazy-initialize Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise LLMProviderError(
                    provider="anthropic",
                    message="ANTHROPIC_API_KEY not configured"
                )
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        context_chunks: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate response using Claude Sonnet 4.5.

        Constructs messages in Anthropic format:
        - System parameter for instructions
        - Messages array with user content
        """
        # Build context block
        context_block = self._format_context(context_chunks)

        # Construct full user message with context
        full_user_message = f"""Context:
{context_block}

Question: {user_message}"""

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": full_user_message},
                ],
                temperature=temperature,
            )

            # Extract text from content blocks
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text

            return LLMResponse(
                content=content,
                tokens_input=response.usage.input_tokens,
                tokens_output=response.usage.output_tokens,
                model=self.model,
                provider=self.provider_name,
            )

        except anthropic.APIError as e:
            raise LLMProviderError(
                provider="anthropic",
                message=f"API error: {str(e)}",
                original_error=e,
            )
        except anthropic.AuthenticationError as e:
            raise LLMProviderError(
                provider="anthropic",
                message="Invalid API key",
                original_error=e,
            )
        except Exception as e:
            raise LLMProviderError(
                provider="anthropic",
                message=str(e),
                original_error=e,
            )

    def _format_context(self, chunks: list[str]) -> str:
        """Format context chunks with citation numbers."""
        if not chunks:
            return "No relevant context found."

        formatted = []
        for i, chunk in enumerate(chunks, start=1):
            formatted.append(f"[{i}] {chunk}")

        return "\n\n".join(formatted)
```

### Provider Factory

```python
# src/article_mind_service/chat/providers/__init__.py
"""LLM provider factory and exports."""

from typing import Literal

from article_mind_service.chat.llm_providers import LLMProvider, LLMProviderError, LLMResponse
from article_mind_service.chat.providers.openai import OpenAIProvider
from article_mind_service.chat.providers.anthropic import AnthropicProvider
from article_mind_service.config import settings


ProviderName = Literal["openai", "anthropic"]


def get_llm_provider(provider: ProviderName | None = None) -> LLMProvider:
    """Get LLM provider instance based on configuration.

    Args:
        provider: Explicit provider name, or None to use configured default

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider name is invalid
    """
    provider_name = provider or settings.llm_provider

    if provider_name == "openai":
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.llm_model if settings.llm_provider == "openai" else "gpt-4o-mini",
        )
    elif provider_name == "anthropic":
        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model if settings.llm_provider == "anthropic" else "claude-sonnet-4-5-20241022",
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")


__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "LLMResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "get_llm_provider",
    "ProviderName",
]
```

---

## 5. Prompt Engineering

### System Prompts

```python
# src/article_mind_service/chat/prompts.py
"""System prompts for RAG Q&A pipeline."""

# Main research assistant prompt
RESEARCH_ASSISTANT_PROMPT = """You are a research assistant helping users understand their saved articles and content. Your task is to answer questions based ONLY on the provided context.

## Instructions

1. **Answer from context only**: Base your answers solely on the numbered context passages provided. Do not use external knowledge.

2. **Cite your sources**: Use inline citations in the format [1], [2], etc. to reference the context passages that support your statements.

3. **Be honest about limitations**: If the context doesn't contain enough information to fully answer the question, clearly state what information is missing.

4. **Be concise but complete**: Provide thorough answers without unnecessary verbosity.

5. **Handle contradictions**: If sources contradict each other, acknowledge this and present both perspectives with their citations.

## Citation Format

- Use [1], [2], etc. corresponding to the numbered context passages
- Place citations at the end of the relevant sentence or claim
- Multiple sources supporting the same point can be cited together: [1][2]

## Response Guidelines

- Start with a direct answer when possible
- Support claims with citations
- If you cannot answer from the context, say: "Based on the available sources, I don't have enough information to answer this question."
- Never make up information or cite non-existent sources"""


# Fallback prompt when no context is found
NO_CONTEXT_PROMPT = """You are a research assistant. The user asked a question, but no relevant content was found in their saved articles.

Respond helpfully by:
1. Acknowledging that no relevant sources were found
2. Suggesting what kind of content they might need to add
3. Offering to help refine their question if it might be too specific

Be friendly and constructive."""


def build_system_prompt(has_context: bool = True) -> str:
    """Get appropriate system prompt based on context availability.

    Args:
        has_context: Whether context chunks were retrieved

    Returns:
        Appropriate system prompt string
    """
    return RESEARCH_ASSISTANT_PROMPT if has_context else NO_CONTEXT_PROMPT


def format_context_with_metadata(
    chunks: list[dict],
) -> tuple[str, list[dict]]:
    """Format context chunks with metadata for prompt injection.

    Args:
        chunks: List of chunk dictionaries with content and metadata

    Returns:
        Tuple of (formatted_context_string, source_metadata_list)

    Each chunk dict expected to have:
        - content: str (chunk text)
        - article_id: int
        - chunk_id: str
        - title: str | None
        - url: str | None
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

        # Track source metadata for response
        source_metadata.append({
            "citation_index": i,
            "article_id": chunk.get("article_id"),
            "chunk_id": chunk.get("chunk_id"),
            "title": title,
            "url": url,
            "excerpt": content[:200] + "..." if len(content) > 200 else content,
        })

    return "\n\n".join(formatted_lines), source_metadata
```

---

## 6. RAG Pipeline Orchestration

```python
# src/article_mind_service/chat/rag_pipeline.py
"""RAG pipeline orchestration for chat Q&A."""

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.chat.llm_providers import LLMResponse
from article_mind_service.chat.providers import get_llm_provider, ProviderName
from article_mind_service.chat.prompts import (
    build_system_prompt,
    format_context_with_metadata,
)
from article_mind_service.config import settings


@dataclass
class RAGResponse:
    """Response from RAG pipeline.

    Attributes:
        content: Generated answer text
        sources: List of source citation metadata
        llm_provider: Provider used for generation
        llm_model: Model used for generation
        tokens_used: Total tokens consumed
        chunks_retrieved: Number of chunks retrieved from search
    """
    content: str
    sources: list[dict]
    llm_provider: str
    llm_model: str
    tokens_used: int
    chunks_retrieved: int


class RAGPipeline:
    """Orchestrates RAG (Retrieve-Augment-Generate) pipeline.

    Flow:
    1. Retrieve relevant chunks via R6 search API
    2. Format context with citation metadata
    3. Generate answer using LLM provider
    4. Extract and map citations to sources

    Usage:
        pipeline = RAGPipeline()
        response = await pipeline.query(
            session_id=123,
            question="What is...",
            db=db_session,
        )
    """

    def __init__(
        self,
        provider: ProviderName | None = None,
        max_context_chunks: int | None = None,
    ):
        """Initialize RAG pipeline.

        Args:
            provider: LLM provider to use (default from settings)
            max_context_chunks: Max chunks to include in context (default from settings)
        """
        self.provider_name = provider or settings.llm_provider
        self.max_context_chunks = max_context_chunks or settings.rag_context_chunks
        self._llm_provider = None

    @property
    def llm_provider(self):
        """Lazy-initialize LLM provider."""
        if self._llm_provider is None:
            self._llm_provider = get_llm_provider(self.provider_name)
        return self._llm_provider

    async def query(
        self,
        session_id: int,
        question: str,
        db: AsyncSession,
        search_client: Any | None = None,
    ) -> RAGResponse:
        """Execute RAG pipeline for a question.

        Args:
            session_id: Session ID to search within
            question: User's question
            db: Database session
            search_client: Optional search client (for testing)

        Returns:
            RAGResponse with answer and sources
        """
        # Step 1: Retrieve relevant chunks
        chunks = await self._retrieve_chunks(
            session_id=session_id,
            query=question,
            limit=self.max_context_chunks,
            search_client=search_client,
        )

        # Step 2: Format context with metadata
        context_str, source_metadata = format_context_with_metadata(chunks)
        has_context = len(chunks) > 0

        # Step 3: Generate answer
        system_prompt = build_system_prompt(has_context=has_context)

        llm_response = await self.llm_provider.generate(
            system_prompt=system_prompt,
            user_message=question,
            context_chunks=[context_str] if has_context else [],
            max_tokens=settings.llm_max_tokens,
            temperature=0.3,
        )

        # Step 4: Extract cited sources only
        cited_sources = self._extract_cited_sources(
            content=llm_response.content,
            source_metadata=source_metadata,
        )

        return RAGResponse(
            content=llm_response.content,
            sources=cited_sources,
            llm_provider=llm_response.provider,
            llm_model=llm_response.model,
            tokens_used=llm_response.total_tokens,
            chunks_retrieved=len(chunks),
        )

    async def _retrieve_chunks(
        self,
        session_id: int,
        query: str,
        limit: int,
        search_client: Any | None = None,
    ) -> list[dict]:
        """Retrieve relevant chunks from R6 search API.

        Args:
            session_id: Session to search within
            query: Search query
            limit: Max results
            search_client: Optional client for testing

        Returns:
            List of chunk dictionaries with content and metadata
        """
        # TODO: Integrate with R6 search API when available
        # For now, return empty to demonstrate flow
        #
        # Expected R6 API call:
        # GET /api/v1/sessions/{session_id}/search?q={query}&limit={limit}
        #
        # Expected response:
        # {
        #     "results": [
        #         {
        #             "chunk_id": "...",
        #             "article_id": 123,
        #             "content": "...",
        #             "score": 0.85,
        #             "article": {
        #                 "title": "...",
        #                 "url": "..."
        #             }
        #         }
        #     ]
        # }

        if search_client:
            # Use injected client for testing
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
                    "title": r.get("article", {}).get("title"),
                    "url": r.get("article", {}).get("url"),
                }
                for r in results.get("results", [])
            ]

        # Placeholder until R6 is implemented
        return []

    def _extract_cited_sources(
        self,
        content: str,
        source_metadata: list[dict],
    ) -> list[dict]:
        """Extract only the sources that were actually cited in the response.

        Args:
            content: LLM response content
            source_metadata: All available source metadata

        Returns:
            List of source metadata for cited sources only
        """
        # Find all citation numbers in the response
        citation_pattern = r'\[(\d+)\]'
        cited_numbers = set(int(m) for m in re.findall(citation_pattern, content))

        # Filter to only cited sources
        cited_sources = [
            source for source in source_metadata
            if source.get("citation_index") in cited_numbers
        ]

        return cited_sources
```

---

## 7. API Endpoints

### Pydantic Schemas

```python
# src/article_mind_service/schemas/chat.py
"""Pydantic schemas for chat API endpoints."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatSource(BaseModel):
    """Source citation in chat response.

    Represents a reference to an article chunk that was used
    to generate part of the response.
    """
    citation_index: int = Field(
        ...,
        description="Citation number [1], [2], etc.",
        examples=[1, 2],
    )
    article_id: int = Field(
        ...,
        description="ID of the source article",
    )
    chunk_id: str | None = Field(
        default=None,
        description="ID of the specific chunk",
    )
    title: str | None = Field(
        default=None,
        description="Article title",
        examples=["Introduction to RAG Systems"],
    )
    url: str | None = Field(
        default=None,
        description="Article URL",
        examples=["https://example.com/article"],
    )
    excerpt: str | None = Field(
        default=None,
        description="Brief excerpt from the cited content",
    )


class ChatRequest(BaseModel):
    """Request body for sending a chat message."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User's question or message",
        examples=["What are the key points about embeddings?"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "What are the main topics covered in my saved articles?"},
                {"message": "Summarize what I've read about vector databases."},
            ]
        }
    }


class ChatMessageResponse(BaseModel):
    """Single chat message in response."""
    id: int = Field(..., description="Message ID")
    role: Literal["user", "assistant"] = Field(
        ...,
        description="Message role",
    )
    content: str = Field(..., description="Message content")
    sources: list[ChatSource] | None = Field(
        default=None,
        description="Source citations (assistant messages only)",
    )
    created_at: datetime = Field(..., description="When the message was created")

    model_config = {"from_attributes": True}


class ChatResponse(BaseModel):
    """Response from sending a chat message.

    Contains the assistant's response with sources and usage metadata.
    """
    message_id: int = Field(..., description="ID of the assistant's response message")
    content: str = Field(..., description="Assistant's response text")
    sources: list[ChatSource] = Field(
        default_factory=list,
        description="Sources cited in the response",
    )
    llm_provider: str | None = Field(
        default=None,
        description="LLM provider used",
        examples=["openai", "anthropic"],
    )
    llm_model: str | None = Field(
        default=None,
        description="Specific model used",
        examples=["gpt-4o-mini", "claude-sonnet-4-5-20241022"],
    )
    tokens_used: int | None = Field(
        default=None,
        description="Total tokens consumed",
    )
    created_at: datetime = Field(..., description="When the response was generated")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message_id": 42,
                    "content": "Based on your saved articles, embeddings are... [1]",
                    "sources": [
                        {
                            "citation_index": 1,
                            "article_id": 123,
                            "title": "Understanding Embeddings",
                            "url": "https://example.com/embeddings",
                        }
                    ],
                    "llm_provider": "openai",
                    "llm_model": "gpt-4o-mini",
                    "tokens_used": 1250,
                    "created_at": "2026-01-19T12:00:00Z",
                }
            ]
        }
    }


class ChatHistoryResponse(BaseModel):
    """Response containing chat history for a session."""
    session_id: int = Field(..., description="Session ID")
    messages: list[ChatMessageResponse] = Field(
        default_factory=list,
        description="List of chat messages in chronological order",
    )
    total_messages: int = Field(..., description="Total number of messages")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": 1,
                    "messages": [
                        {
                            "id": 1,
                            "role": "user",
                            "content": "What are embeddings?",
                            "sources": None,
                            "created_at": "2026-01-19T12:00:00Z",
                        },
                        {
                            "id": 2,
                            "role": "assistant",
                            "content": "Embeddings are vector representations... [1]",
                            "sources": [{"citation_index": 1, "article_id": 123, "title": "..."}],
                            "created_at": "2026-01-19T12:00:01Z",
                        },
                    ],
                    "total_messages": 2,
                }
            ]
        }
    }
```

### Router Implementation

```python
# src/article_mind_service/routers/chat.py
"""Chat API endpoints for Q&A functionality."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.database import get_db
from article_mind_service.models.chat import ChatMessage
from article_mind_service.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatHistoryResponse,
    ChatMessageResponse,
    ChatSource,
)
from article_mind_service.chat.rag_pipeline import RAGPipeline

router = APIRouter(
    prefix="/api/v1/sessions/{session_id}/chat",
    tags=["chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send chat message and get response",
    description="""
Send a question about the session's content and receive an LLM-generated
answer with source citations.

The RAG pipeline:
1. Retrieves relevant content chunks from the session's articles
2. Augments the prompt with context
3. Generates an answer using the configured LLM
4. Returns the response with inline citations [1], [2], etc.

Both the user message and assistant response are persisted to the database.
""",
)
async def send_chat_message(
    session_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """Send a chat message and get an AI-generated response.

    Args:
        session_id: ID of the session to query
        request: Chat message request body
        db: Database session

    Returns:
        ChatResponse with assistant's answer and citations

    Raises:
        HTTPException 404: If session not found
        HTTPException 500: If LLM generation fails
    """
    # TODO: Verify session exists (when Session model is implemented)
    # result = await db.execute(select(Session).where(Session.id == session_id))
    # session = result.scalar_one_or_none()
    # if not session:
    #     raise HTTPException(status_code=404, detail="Session not found")

    # Save user message
    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=request.message,
    )
    db.add(user_message)
    await db.flush()  # Get user_message.id

    # Execute RAG pipeline
    try:
        pipeline = RAGPipeline()
        rag_response = await pipeline.query(
            session_id=session_id,
            question=request.message,
            db=db,
        )
    except Exception as e:
        # Log error and return helpful message
        print(f"RAG pipeline error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}",
        )

    # Save assistant response
    assistant_message = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=rag_response.content,
        sources=rag_response.sources,
        llm_provider=rag_response.llm_provider,
        llm_model=rag_response.llm_model,
        tokens_used=rag_response.tokens_used,
    )
    db.add(assistant_message)
    await db.flush()

    # Build response
    return ChatResponse(
        message_id=assistant_message.id,
        content=assistant_message.content,
        sources=[ChatSource(**s) for s in rag_response.sources],
        llm_provider=rag_response.llm_provider,
        llm_model=rag_response.llm_model,
        tokens_used=rag_response.tokens_used,
        created_at=assistant_message.created_at,
    )


@router.get(
    "/history",
    response_model=ChatHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get chat history",
    description="Retrieve all chat messages for a session in chronological order.",
)
async def get_chat_history(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> ChatHistoryResponse:
    """Get chat history for a session.

    Args:
        session_id: ID of the session
        db: Database session

    Returns:
        ChatHistoryResponse with all messages
    """
    # Fetch messages
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    messages = result.scalars().all()

    # Count total
    count_result = await db.execute(
        select(func.count())
        .select_from(ChatMessage)
        .where(ChatMessage.session_id == session_id)
    )
    total = count_result.scalar_one()

    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            ChatMessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                sources=[ChatSource(**s) for s in msg.sources] if msg.sources else None,
                created_at=msg.created_at,
            )
            for msg in messages
        ],
        total_messages=total,
    )


@router.delete(
    "/history",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear chat history",
    description="Delete all chat messages for a session.",
)
async def clear_chat_history(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Clear all chat history for a session.

    Args:
        session_id: ID of the session
        db: Database session
    """
    await db.execute(
        delete(ChatMessage).where(ChatMessage.session_id == session_id)
    )
```

### Router Registration

Update `src/article_mind_service/main.py`:

```python
# Add import
from .routers import chat_router

# Add router registration (after health_router)
app.include_router(chat_router)
```

Update `src/article_mind_service/routers/__init__.py`:

```python
from .health import router as health_router
from .chat import router as chat_router

__all__ = [
    "health_router",
    "chat_router",
]
```

---

## 8. Configuration

### Environment Variables

Update `.env.example`:

```env
# Existing config...

# ===== LLM Configuration =====
# Provider: "openai" or "anthropic"
LLM_PROVIDER=openai

# API Keys (only one needed based on provider)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...

# Model selection (provider-specific)
# OpenAI: gpt-4o-mini, gpt-4o, gpt-4-turbo
# Anthropic: claude-sonnet-4-5-20241022, claude-opus-4-5-20251101
LLM_MODEL=gpt-4o-mini

# Generation settings
LLM_MAX_TOKENS=2048

# ===== RAG Configuration =====
# Number of context chunks to include in prompt
RAG_CONTEXT_CHUNKS=5
```

### Settings Update

Update `src/article_mind_service/config.py`:

```python
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Existing settings...

    # LLM Configuration
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: 'openai' or 'anthropic'",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model identifier",
    )
    llm_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for LLM generation",
    )

    # RAG Configuration
    rag_context_chunks: int = Field(
        default=5,
        description="Number of context chunks for RAG",
    )
```

---

## 9. Module Structure

```
article-mind-service/
└── src/article_mind_service/
    ├── chat/
    │   ├── __init__.py           # Module exports
    │   ├── llm_providers.py      # Abstract base class, types
    │   ├── rag_pipeline.py       # RAG orchestration
    │   ├── prompts.py            # System prompts
    │   └── providers/
    │       ├── __init__.py       # Provider factory
    │       ├── openai.py         # OpenAI implementation
    │       └── anthropic.py      # Anthropic implementation
    ├── models/
    │   ├── __init__.py           # Import ChatMessage
    │   └── chat.py               # ChatMessage model
    ├── schemas/
    │   ├── __init__.py           # Export chat schemas
    │   └── chat.py               # Chat Pydantic schemas
    └── routers/
        ├── __init__.py           # Export chat_router
        └── chat.py               # Chat API endpoints
```

### Module Exports

```python
# src/article_mind_service/chat/__init__.py
"""Chat module for Q&A functionality."""

from article_mind_service.chat.llm_providers import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
)
from article_mind_service.chat.rag_pipeline import RAGPipeline, RAGResponse
from article_mind_service.chat.providers import (
    get_llm_provider,
    OpenAIProvider,
    AnthropicProvider,
)

__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "LLMResponse",
    "RAGPipeline",
    "RAGResponse",
    "get_llm_provider",
    "OpenAIProvider",
    "AnthropicProvider",
]
```

---

## 10. UI Components (Svelte 5 Runes)

### ChatContainer Component

```svelte
<!-- src/lib/components/chat/ChatContainer.svelte -->
<script lang="ts">
  import { onMount } from 'svelte';
  import type { components } from '$lib/api/generated';
  import { getChatHistory, sendChatMessage, clearChatHistory } from '$lib/api/chat';
  import MessageBubble from './MessageBubble.svelte';
  import ChatInput from './ChatInput.svelte';

  type ChatMessage = components['schemas']['ChatMessageResponse'];
  type ChatSource = components['schemas']['ChatSource'];

  // Props
  interface Props {
    sessionId: number;
  }
  let { sessionId }: Props = $props();

  // State
  let messages = $state<ChatMessage[]>([]);
  let isLoading = $state(false);
  let isSending = $state(false);
  let error = $state<string | null>(null);
  let messagesContainer: HTMLDivElement | null = $state(null);

  // Load chat history on mount
  onMount(async () => {
    await loadHistory();
  });

  /**
   * Load chat history from API
   */
  async function loadHistory() {
    isLoading = true;
    error = null;
    try {
      const response = await getChatHistory(sessionId);
      messages = response.messages;
      scrollToBottom();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load chat history';
    } finally {
      isLoading = false;
    }
  }

  /**
   * Send a new message
   */
  async function handleSendMessage(content: string) {
    if (!content.trim() || isSending) return;

    isSending = true;
    error = null;

    // Optimistically add user message
    const userMessage: ChatMessage = {
      id: Date.now(), // Temporary ID
      role: 'user',
      content: content.trim(),
      sources: null,
      created_at: new Date().toISOString(),
    };
    messages = [...messages, userMessage];
    scrollToBottom();

    try {
      const response = await sendChatMessage(sessionId, { message: content });

      // Add assistant response
      const assistantMessage: ChatMessage = {
        id: response.message_id,
        role: 'assistant',
        content: response.content,
        sources: response.sources,
        created_at: response.created_at,
      };
      messages = [...messages, assistantMessage];
      scrollToBottom();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to send message';
      // Remove optimistic user message on error
      messages = messages.slice(0, -1);
    } finally {
      isSending = false;
    }
  }

  /**
   * Clear all chat history
   */
  async function handleClearHistory() {
    if (!confirm('Clear all chat history for this session?')) return;

    try {
      await clearChatHistory(sessionId);
      messages = [];
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to clear history';
    }
  }

  /**
   * Scroll to bottom of messages
   */
  function scrollToBottom() {
    setTimeout(() => {
      if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }
    }, 100);
  }
</script>

<div class="chat-container">
  <div class="chat-header">
    <h3>Knowledge Q&A</h3>
    {#if messages.length > 0}
      <button class="clear-btn" onclick={handleClearHistory}>
        Clear History
      </button>
    {/if}
  </div>

  {#if error}
    <div class="error-banner">
      {error}
      <button onclick={() => error = null}>Dismiss</button>
    </div>
  {/if}

  <div class="messages" bind:this={messagesContainer}>
    {#if isLoading}
      <div class="loading">Loading chat history...</div>
    {:else if messages.length === 0}
      <div class="empty-state">
        <p>Ask questions about your saved articles.</p>
        <p class="hint">The AI will answer based on your content with citations.</p>
      </div>
    {:else}
      {#each messages as message (message.id)}
        <MessageBubble
          role={message.role}
          content={message.content}
          sources={message.sources}
          timestamp={message.created_at}
        />
      {/each}
    {/if}

    {#if isSending}
      <div class="typing-indicator">
        <span></span><span></span><span></span>
      </div>
    {/if}
  </div>

  <ChatInput
    onSend={handleSendMessage}
    disabled={isSending || isLoading}
    placeholder={isSending ? 'Generating response...' : 'Ask a question...'}
  />
</div>

<style>
  .chat-container {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 400px;
    max-height: 80vh;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background: white;
  }

  .chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    border-bottom: 1px solid #e5e7eb;
  }

  .chat-header h3 {
    margin: 0;
    font-size: 1.125rem;
    font-weight: 600;
  }

  .clear-btn {
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
    color: #6b7280;
    background: transparent;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    cursor: pointer;
  }

  .clear-btn:hover {
    background: #f3f4f6;
  }

  .error-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    background: #fef2f2;
    color: #dc2626;
    font-size: 0.875rem;
  }

  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .loading,
  .empty-state {
    text-align: center;
    color: #6b7280;
    padding: 2rem;
  }

  .empty-state .hint {
    font-size: 0.875rem;
    color: #9ca3af;
    margin-top: 0.5rem;
  }

  .typing-indicator {
    display: flex;
    gap: 4px;
    padding: 1rem;
    align-self: flex-start;
  }

  .typing-indicator span {
    width: 8px;
    height: 8px;
    background: #d1d5db;
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out;
  }

  .typing-indicator span:nth-child(1) { animation-delay: 0s; }
  .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
  .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

  @keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-6px); }
  }
</style>
```

### MessageBubble Component

```svelte
<!-- src/lib/components/chat/MessageBubble.svelte -->
<script lang="ts">
  import type { components } from '$lib/api/generated';
  import SourceCitations from './SourceCitations.svelte';

  type ChatSource = components['schemas']['ChatSource'];

  // Props
  interface Props {
    role: 'user' | 'assistant';
    content: string;
    sources?: ChatSource[] | null;
    timestamp: string;
  }
  let { role, content, sources, timestamp }: Props = $props();

  // Format timestamp
  let formattedTime = $derived(
    new Date(timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    })
  );

  // Process content to highlight citations
  let processedContent = $derived(
    role === 'assistant'
      ? content.replace(/\[(\d+)\]/g, '<span class="citation">[$1]</span>')
      : content
  );
</script>

<div class="message" class:user={role === 'user'} class:assistant={role === 'assistant'}>
  <div class="bubble">
    <div class="content">
      {#if role === 'assistant'}
        {@html processedContent}
      {:else}
        {content}
      {/if}
    </div>

    {#if sources && sources.length > 0}
      <SourceCitations {sources} />
    {/if}
  </div>

  <span class="timestamp">{formattedTime}</span>
</div>

<style>
  .message {
    display: flex;
    flex-direction: column;
    max-width: 80%;
  }

  .message.user {
    align-self: flex-end;
  }

  .message.assistant {
    align-self: flex-start;
  }

  .bubble {
    padding: 0.75rem 1rem;
    border-radius: 12px;
    line-height: 1.5;
  }

  .user .bubble {
    background: #3b82f6;
    color: white;
    border-bottom-right-radius: 4px;
  }

  .assistant .bubble {
    background: #f3f4f6;
    color: #1f2937;
    border-bottom-left-radius: 4px;
  }

  .content {
    white-space: pre-wrap;
    word-break: break-word;
  }

  .content :global(.citation) {
    color: #3b82f6;
    font-weight: 600;
    cursor: pointer;
  }

  .timestamp {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 0.25rem;
    padding: 0 0.5rem;
  }

  .user .timestamp {
    text-align: right;
  }
</style>
```

### ChatInput Component

```svelte
<!-- src/lib/components/chat/ChatInput.svelte -->
<script lang="ts">
  // Props
  interface Props {
    onSend: (message: string) => void;
    disabled?: boolean;
    placeholder?: string;
  }
  let { onSend, disabled = false, placeholder = 'Type a message...' }: Props = $props();

  // State
  let inputValue = $state('');
  let textareaRef: HTMLTextAreaElement | null = $state(null);

  /**
   * Handle form submission
   */
  function handleSubmit(e: Event) {
    e.preventDefault();
    if (inputValue.trim() && !disabled) {
      onSend(inputValue);
      inputValue = '';
      resizeTextarea();
    }
  }

  /**
   * Handle keyboard shortcuts
   */
  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  }

  /**
   * Auto-resize textarea based on content
   */
  function resizeTextarea() {
    if (textareaRef) {
      textareaRef.style.height = 'auto';
      textareaRef.style.height = Math.min(textareaRef.scrollHeight, 150) + 'px';
    }
  }
</script>

<form class="chat-input" onsubmit={handleSubmit}>
  <textarea
    bind:this={textareaRef}
    bind:value={inputValue}
    oninput={resizeTextarea}
    onkeydown={handleKeyDown}
    {placeholder}
    {disabled}
    rows="1"
  ></textarea>

  <button type="submit" {disabled} aria-label="Send message">
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M22 2L11 13" />
      <path d="M22 2L15 22L11 13L2 9L22 2Z" />
    </svg>
  </button>
</form>

<style>
  .chat-input {
    display: flex;
    align-items: flex-end;
    gap: 0.5rem;
    padding: 1rem;
    border-top: 1px solid #e5e7eb;
    background: white;
  }

  textarea {
    flex: 1;
    padding: 0.75rem 1rem;
    border: 1px solid #d1d5db;
    border-radius: 20px;
    resize: none;
    font-family: inherit;
    font-size: 0.9375rem;
    line-height: 1.5;
    max-height: 150px;
    overflow-y: auto;
  }

  textarea:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
  }

  textarea:disabled {
    background: #f9fafb;
    color: #9ca3af;
  }

  button {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  button:hover:not(:disabled) {
    background: #2563eb;
  }

  button:disabled {
    background: #d1d5db;
    cursor: not-allowed;
  }
</style>
```

### SourceCitations Component

```svelte
<!-- src/lib/components/chat/SourceCitations.svelte -->
<script lang="ts">
  import type { components } from '$lib/api/generated';

  type ChatSource = components['schemas']['ChatSource'];

  // Props
  interface Props {
    sources: ChatSource[];
  }
  let { sources }: Props = $props();

  // State
  let isExpanded = $state(false);
</script>

<div class="sources">
  <button
    class="toggle-btn"
    onclick={() => isExpanded = !isExpanded}
    aria-expanded={isExpanded}
  >
    <span class="icon">{isExpanded ? '▼' : '▶'}</span>
    {sources.length} source{sources.length !== 1 ? 's' : ''}
  </button>

  {#if isExpanded}
    <ul class="sources-list">
      {#each sources as source}
        <li>
          <span class="citation-number">[{source.citation_index}]</span>
          <div class="source-details">
            {#if source.url}
              <a href={source.url} target="_blank" rel="noopener noreferrer">
                {source.title || 'Untitled'}
              </a>
            {:else}
              <span class="title">{source.title || 'Untitled'}</span>
            {/if}
            {#if source.excerpt}
              <p class="excerpt">"{source.excerpt}"</p>
            {/if}
          </div>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .sources {
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid rgba(0, 0, 0, 0.1);
  }

  .toggle-btn {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0.5rem;
    font-size: 0.8125rem;
    color: #6b7280;
    background: transparent;
    border: none;
    cursor: pointer;
    border-radius: 4px;
  }

  .toggle-btn:hover {
    background: rgba(0, 0, 0, 0.05);
  }

  .icon {
    font-size: 0.625rem;
  }

  .sources-list {
    list-style: none;
    margin: 0.5rem 0 0 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .sources-list li {
    display: flex;
    gap: 0.5rem;
    font-size: 0.8125rem;
  }

  .citation-number {
    color: #3b82f6;
    font-weight: 600;
    flex-shrink: 0;
  }

  .source-details {
    flex: 1;
    min-width: 0;
  }

  .source-details a {
    color: #2563eb;
    text-decoration: none;
  }

  .source-details a:hover {
    text-decoration: underline;
  }

  .title {
    color: #374151;
  }

  .excerpt {
    margin: 0.25rem 0 0 0;
    font-size: 0.75rem;
    color: #6b7280;
    font-style: italic;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
  }
</style>
```

### Chat API Client

```typescript
// src/lib/api/chat.ts
/**
 * Chat API client functions
 */

import { apiClient } from './client';
import type { components } from './generated';

type ChatRequest = components['schemas']['ChatRequest'];
type ChatResponse = components['schemas']['ChatResponse'];
type ChatHistoryResponse = components['schemas']['ChatHistoryResponse'];

/**
 * Send a chat message and get response
 */
export async function sendChatMessage(
  sessionId: number,
  request: ChatRequest
): Promise<ChatResponse> {
  return apiClient.post<ChatResponse>(
    `/api/v1/sessions/${sessionId}/chat`,
    request
  );
}

/**
 * Get chat history for a session
 */
export async function getChatHistory(
  sessionId: number
): Promise<ChatHistoryResponse> {
  return apiClient.get<ChatHistoryResponse>(
    `/api/v1/sessions/${sessionId}/chat/history`
  );
}

/**
 * Clear chat history for a session
 */
export async function clearChatHistory(sessionId: number): Promise<void> {
  return apiClient.delete(`/api/v1/sessions/${sessionId}/chat/history`);
}
```

---

## 11. Implementation Steps

### Phase 1: Backend Foundation (Day 1)

| Step | Task | Est. Time | Dependencies |
|------|------|-----------|--------------|
| 1.1 | Add dependencies to `pyproject.toml` (openai, anthropic) | 15m | - |
| 1.2 | Update `config.py` with LLM settings | 30m | 1.1 |
| 1.3 | Create `chat/` module structure | 15m | - |
| 1.4 | Implement `llm_providers.py` base classes | 30m | - |
| 1.5 | Implement OpenAI provider | 1h | 1.4 |
| 1.6 | Implement Anthropic provider | 1h | 1.4 |
| 1.7 | Write provider unit tests | 1h | 1.5, 1.6 |

### Phase 2: RAG Pipeline (Day 2)

| Step | Task | Est. Time | Dependencies |
|------|------|-----------|--------------|
| 2.1 | Create `prompts.py` with system prompts | 30m | - |
| 2.2 | Implement `rag_pipeline.py` | 2h | 1.5, 2.1 |
| 2.3 | Add R6 search integration (stub) | 1h | 2.2 |
| 2.4 | Write RAG pipeline unit tests | 1h | 2.2 |

### Phase 3: Database & API (Day 3)

| Step | Task | Est. Time | Dependencies |
|------|------|-----------|--------------|
| 3.1 | Create `models/chat.py` SQLAlchemy model | 30m | - |
| 3.2 | Create and run migration | 30m | 3.1 |
| 3.3 | Create `schemas/chat.py` Pydantic schemas | 1h | - |
| 3.4 | Implement `routers/chat.py` endpoints | 2h | 3.2, 3.3 |
| 3.5 | Write API integration tests | 1.5h | 3.4 |
| 3.6 | Update OpenAPI documentation | 30m | 3.4 |

### Phase 4: Frontend UI (Day 4-5)

| Step | Task | Est. Time | Dependencies |
|------|------|-----------|--------------|
| 4.1 | Generate TypeScript types from OpenAPI | 15m | 3.6 |
| 4.2 | Create `ChatInput.svelte` component | 1h | 4.1 |
| 4.3 | Create `MessageBubble.svelte` component | 1h | 4.1 |
| 4.4 | Create `SourceCitations.svelte` component | 1h | 4.1 |
| 4.5 | Create `ChatContainer.svelte` component | 2h | 4.2-4.4 |
| 4.6 | Implement chat API client | 30m | 4.1 |
| 4.7 | Integrate chat into session detail page | 1h | 4.5, 4.6 |
| 4.8 | Write component tests | 2h | 4.5 |

### Phase 5: Polish & Testing (Day 5)

| Step | Task | Est. Time | Dependencies |
|------|------|-----------|--------------|
| 5.1 | End-to-end testing | 2h | 4.7 |
| 5.2 | Error handling improvements | 1h | 5.1 |
| 5.3 | Loading states and UX polish | 1h | 5.1 |
| 5.4 | Documentation update | 1h | - |

---

## 12. Testing Strategy

### Backend Tests

#### Unit Tests

```python
# tests/unit/chat/test_providers.py
"""Unit tests for LLM providers."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from article_mind_service.chat.providers import (
    OpenAIProvider,
    AnthropicProvider,
    get_llm_provider,
)
from article_mind_service.chat.llm_providers import LLMProviderError


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    @pytest.fixture
    def provider(self):
        return OpenAIProvider(api_key="test-key", model="gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        with patch.object(provider, 'client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await provider.generate(
                system_prompt="You are helpful.",
                user_message="Test question?",
                context_chunks=["Context 1"],
            )

            assert result.content == "Test response"
            assert result.tokens_input == 100
            assert result.tokens_output == 50
            assert result.provider == "openai"

    @pytest.mark.asyncio
    async def test_generate_api_error(self, provider):
        """Test handling of API errors."""
        import openai

        with patch.object(provider, 'client') as mock_client:
            mock_client.chat.completions.create = AsyncMock(
                side_effect=openai.APIError("Rate limited", None, None)
            )

            with pytest.raises(LLMProviderError) as exc_info:
                await provider.generate(
                    system_prompt="You are helpful.",
                    user_message="Test?",
                    context_chunks=[],
                )

            assert exc_info.value.provider == "openai"


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    @pytest.fixture
    def provider(self):
        return AnthropicProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful generation."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Claude response")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        with patch.object(provider, 'client') as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_response)

            result = await provider.generate(
                system_prompt="You are helpful.",
                user_message="Test question?",
                context_chunks=["Context 1"],
            )

            assert result.content == "Claude response"
            assert result.provider == "anthropic"


class TestProviderFactory:
    """Tests for provider factory."""

    def test_get_openai_provider(self):
        """Test getting OpenAI provider."""
        with patch('article_mind_service.chat.providers.settings') as mock_settings:
            mock_settings.openai_api_key = "test-key"
            mock_settings.llm_provider = "openai"
            mock_settings.llm_model = "gpt-4o-mini"

            provider = get_llm_provider("openai")
            assert isinstance(provider, OpenAIProvider)

    def test_invalid_provider(self):
        """Test invalid provider name."""
        with pytest.raises(ValueError):
            get_llm_provider("invalid")
```

#### Integration Tests

```python
# tests/integration/test_chat_api.py
"""Integration tests for chat API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_send_chat_message(async_client: AsyncClient, test_session):
    """Test sending a chat message."""
    response = await async_client.post(
        f"/api/v1/sessions/{test_session.id}/chat",
        json={"message": "What is the main topic?"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "message_id" in data
    assert "content" in data
    assert "sources" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_get_chat_history(async_client: AsyncClient, test_session, test_chat_message):
    """Test getting chat history."""
    response = await async_client.get(
        f"/api/v1/sessions/{test_session.id}/chat/history"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == test_session.id
    assert len(data["messages"]) > 0


@pytest.mark.asyncio
async def test_clear_chat_history(async_client: AsyncClient, test_session, test_chat_message):
    """Test clearing chat history."""
    response = await async_client.delete(
        f"/api/v1/sessions/{test_session.id}/chat/history"
    )

    assert response.status_code == 204

    # Verify history is empty
    history_response = await async_client.get(
        f"/api/v1/sessions/{test_session.id}/chat/history"
    )
    assert len(history_response.json()["messages"]) == 0


@pytest.mark.asyncio
async def test_chat_message_validation(async_client: AsyncClient, test_session):
    """Test message validation."""
    # Empty message
    response = await async_client.post(
        f"/api/v1/sessions/{test_session.id}/chat",
        json={"message": ""},
    )
    assert response.status_code == 422

    # Message too long
    response = await async_client.post(
        f"/api/v1/sessions/{test_session.id}/chat",
        json={"message": "x" * 5000},
    )
    assert response.status_code == 422
```

### Frontend Tests

```typescript
// tests/unit/chat/ChatContainer.test.ts
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import ChatContainer from '$lib/components/chat/ChatContainer.svelte';
import * as chatApi from '$lib/api/chat';

vi.mock('$lib/api/chat');

describe('ChatContainer', () => {
  const mockHistory = {
    session_id: 1,
    messages: [
      {
        id: 1,
        role: 'user',
        content: 'Hello',
        sources: null,
        created_at: '2026-01-19T12:00:00Z',
      },
      {
        id: 2,
        role: 'assistant',
        content: 'Hi! How can I help? [1]',
        sources: [{ citation_index: 1, article_id: 1, title: 'Test' }],
        created_at: '2026-01-19T12:00:01Z',
      },
    ],
    total_messages: 2,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(chatApi.getChatHistory).mockResolvedValue(mockHistory);
  });

  it('loads and displays chat history', async () => {
    render(ChatContainer, { props: { sessionId: 1 } });

    await waitFor(() => {
      expect(screen.getByText('Hello')).toBeDefined();
      expect(screen.getByText(/Hi! How can I help/)).toBeDefined();
    });
  });

  it('sends a message', async () => {
    const mockResponse = {
      message_id: 3,
      content: 'Response',
      sources: [],
      created_at: '2026-01-19T12:00:02Z',
    };
    vi.mocked(chatApi.sendChatMessage).mockResolvedValue(mockResponse);

    render(ChatContainer, { props: { sessionId: 1 } });
    await waitFor(() => expect(screen.getByPlaceholderText(/Ask a question/)).toBeDefined());

    const input = screen.getByRole('textbox');
    const form = input.closest('form');

    await fireEvent.input(input, { target: { value: 'New question' } });
    await fireEvent.submit(form!);

    await waitFor(() => {
      expect(chatApi.sendChatMessage).toHaveBeenCalledWith(1, { message: 'New question' });
    });
  });

  it('displays empty state when no messages', async () => {
    vi.mocked(chatApi.getChatHistory).mockResolvedValue({
      session_id: 1,
      messages: [],
      total_messages: 0,
    });

    render(ChatContainer, { props: { sessionId: 1 } });

    await waitFor(() => {
      expect(screen.getByText(/Ask questions about your saved articles/)).toBeDefined();
    });
  });
});
```

### Mock LLM Provider for Testing

```python
# tests/conftest.py (additions)
"""Test fixtures for chat functionality."""

import pytest
from unittest.mock import AsyncMock

from article_mind_service.chat.llm_providers import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or ["This is a test response. [1]"]
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        context_chunks: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> LLMResponse:
        response_text = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1

        return LLMResponse(
            content=response_text,
            tokens_input=100,
            tokens_output=50,
            model="mock-model",
            provider="mock",
        )


@pytest.fixture
def mock_llm_provider():
    """Provide mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_search_client():
    """Provide mock search client."""
    client = AsyncMock()
    client.search.return_value = {
        "results": [
            {
                "chunk_id": "chunk-1",
                "article_id": 1,
                "content": "This is test content about embeddings.",
                "score": 0.95,
                "article": {
                    "title": "Understanding Embeddings",
                    "url": "https://example.com/embeddings",
                },
            },
        ]
    }
    return client
```

---

## 13. Acceptance Criteria

### Functional Requirements

- [ ] User can send a question via chat input
- [ ] System retrieves relevant chunks from R6 search (when integrated)
- [ ] LLM generates answer based on retrieved context
- [ ] Response includes inline citations [1], [2], etc.
- [ ] Citations are expandable to show source details
- [ ] Chat history is persisted to PostgreSQL
- [ ] Chat history loads on page refresh
- [ ] User can clear chat history
- [ ] Both OpenAI and Anthropic providers work

### Non-Functional Requirements

- [ ] Response time < 5 seconds (excluding LLM latency)
- [ ] UI remains responsive during LLM generation
- [ ] Graceful error handling with user-friendly messages
- [ ] Mobile-responsive chat interface
- [ ] Accessible (keyboard navigation, screen reader support)

### API Contract Compliance

- [ ] POST `/api/v1/sessions/{id}/chat` returns ChatResponse
- [ ] GET `/api/v1/sessions/{id}/chat/history` returns ChatHistoryResponse
- [ ] DELETE `/api/v1/sessions/{id}/chat/history` returns 204
- [ ] Error responses follow ErrorResponse schema
- [ ] OpenAPI spec includes all chat endpoints

### Testing Coverage

- [ ] LLM provider unit tests (OpenAI, Anthropic)
- [ ] RAG pipeline unit tests
- [ ] API endpoint integration tests
- [ ] Pydantic schema validation tests
- [ ] UI component tests
- [ ] End-to-end flow test

---

## 14. Future Enhancements

### Streaming Responses
- Implement Server-Sent Events (SSE) for real-time token streaming
- Progressive display of LLM output
- Cancel generation in progress

### Conversation Context
- Include previous messages in LLM context
- Configurable conversation memory window
- Topic-based conversation threading

### Advanced RAG Features
- Query rewriting and expansion
- Multi-query retrieval fusion
- Adaptive chunk selection based on question type
- Confidence scoring for answers

### User Experience
- Suggested follow-up questions
- Copy response to clipboard
- Export conversation to markdown
- Search within chat history
- Keyboard shortcuts

### Analytics
- Token usage tracking and billing
- Response quality feedback (thumbs up/down)
- Popular questions analytics
- Source coverage reports

---

## 15. Dependencies to Add

### Backend (pyproject.toml)

```toml
[project]
dependencies = [
    # Existing...
    "openai>=1.58.0",
    "anthropic>=0.43.0",
]
```

### Frontend (package.json)

No additional dependencies required - uses existing SvelteKit and fetch API.

---

**Plan Status:** Ready for implementation
**Last Updated:** 2026-01-19
**Estimated Total Time:** 3-5 days
