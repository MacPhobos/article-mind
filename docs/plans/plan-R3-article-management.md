# Plan R3: Article Management API/UI

**Plan ID:** R3-article-management
**Created:** 2026-01-19
**Dependencies:** Plan 03 (database-setup), Plan 04 (service-health-api), Plan 05 (ui-health-display)
**Estimated Time:** 8-12 hours

---

## 1. Overview

### Purpose
Implement complete article management functionality allowing users to add articles via URL or file upload, view article lists within research sessions, and manage article lifecycle including deletion. This is the core data management feature of Article Mind.

### Architecture Decisions (Pre-determined)
- **File Storage:** PostgreSQL + filesystem (metadata in DB, content/files on disk)
- **Session Lifecycle:** Full lifecycle with soft delete
- **Auth Model:** Single-user (no authentication required)

### Scope
- Database schema for articles with soft delete support
- File upload handling with configurable storage paths
- API endpoints for CRUD operations on articles within sessions
- Svelte 5 UI components with Runes API for article management
- Extraction status tracking (for future content extraction integration)

### Dependencies
- **Plan 03:** Requires PostgreSQL database and Alembic migrations
- **Plan 04:** Requires FastAPI scaffolding and Pydantic schema patterns
- **Plan 05:** Requires SvelteKit scaffolding and API client patterns

### Outputs
- SQLAlchemy async models for `research_sessions` and `articles` tables
- Pydantic schemas for all request/response types
- FastAPI routers with CRUD endpoints
- Svelte 5 components for article management UI
- Comprehensive test coverage (API and UI)
- Alembic migration for new tables

---

## 2. Database Schema

### SQLAlchemy 2.0 Async Models

#### Research Sessions Model

Create `src/article_mind_service/models/research_session.py`:

```python
"""Research session model."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from article_mind_service.database import Base

if TYPE_CHECKING:
    from .article import Article


class ResearchSession(Base):
    """Research session model.

    A research session groups related articles together for focused research.
    Supports soft delete via deleted_at timestamp.

    Design Decision: Soft Delete Pattern
    -------------------------------------
    Using deleted_at timestamp instead of hard delete because:
    - Allows data recovery if needed
    - Maintains referential integrity with articles
    - Enables audit trail of user actions
    - Articles within deleted sessions are preserved

    Trade-offs:
    - Requires filtering deleted records in all queries
    - Slight storage overhead for deleted records
    - Need to handle cascade soft delete for articles
    """

    __tablename__ = "research_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    # Relationships
    articles: Mapped[list["Article"]] = relationship(
        "Article",
        back_populates="session",
        lazy="selectin",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ResearchSession(id={self.id}, name='{self.name}')>"

    @property
    def is_deleted(self) -> bool:
        """Check if session is soft deleted."""
        return self.deleted_at is not None
```

#### Articles Model

Create `src/article_mind_service/models/article.py`:

```python
"""Article model."""

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy import DateTime, Enum, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from article_mind_service.database import Base

if TYPE_CHECKING:
    from .research_session import ResearchSession


# Article type enum
ArticleType = Literal["url", "file"]

# Extraction status enum
ExtractionStatus = Literal["pending", "processing", "completed", "failed"]


class Article(Base):
    """Article model.

    Represents a single article (URL or uploaded file) within a research session.
    Tracks extraction status for content processing workflow.

    Design Decision: Separate Storage Paths
    ----------------------------------------
    Files are stored on the filesystem with paths tracked in database because:
    - Large files should not bloat the database
    - Filesystem storage is simpler to manage and backup
    - Path structure enables easy cleanup by session
    - Content can be served directly without DB round-trip

    Directory Structure:
    data/uploads/{session_id}/{article_id}/
    ├── original.{ext}     # Original uploaded file
    └── extracted.txt      # Extracted text content (after processing)

    Trade-offs:
    - Requires filesystem + DB synchronization
    - Need to handle orphaned files on failed operations
    - Backup strategy must include both DB and filesystem
    """

    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("research_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Article type and source
    type: Mapped[str] = mapped_column(
        Enum("url", "file", name="article_type", create_constraint=True),
        nullable=False,
    )
    original_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Storage
    storage_path: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Filesystem path for uploaded files (relative to UPLOAD_BASE_PATH)",
    )

    # Metadata
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    extraction_status: Mapped[str] = mapped_column(
        Enum(
            "pending", "processing", "completed", "failed",
            name="extraction_status",
            create_constraint=True,
        ),
        nullable=False,
        default="pending",
        index=True,
    )

    # Extracted content (stored directly for smaller text content)
    content_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    # Relationships
    session: Mapped["ResearchSession"] = relationship(
        "ResearchSession",
        back_populates="articles",
    )

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_articles_session_status", "session_id", "extraction_status"),
        Index("ix_articles_session_deleted", "session_id", "deleted_at"),
    )

    def __repr__(self) -> str:
        source = self.original_url or self.original_filename or "unknown"
        return f"<Article(id={self.id}, type='{self.type}', source='{source[:50]}')>"

    @property
    def is_deleted(self) -> bool:
        """Check if article is soft deleted."""
        return self.deleted_at is not None

    @property
    def display_name(self) -> str:
        """Get display name for the article."""
        if self.title:
            return self.title
        if self.type == "url" and self.original_url:
            return self.original_url[:100]
        if self.type == "file" and self.original_filename:
            return self.original_filename
        return f"Article #{self.id}"
```

#### Update Models __init__.py

Update `src/article_mind_service/models/__init__.py`:

```python
"""SQLAlchemy models for database schema."""

# Import all models here to ensure they are registered with Base.metadata
# This is required for Alembic autogenerate to detect models

from .article import Article
from .research_session import ResearchSession

__all__ = [
    "Article",
    "ResearchSession",
]
```

---

## 3. Pydantic Schemas

Create `src/article_mind_service/schemas/article.py`:

```python
"""Article and research session Pydantic schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


# =============================================================================
# Enums (as Literal types for better type safety)
# =============================================================================

ArticleType = Literal["url", "file"]
ExtractionStatus = Literal["pending", "processing", "completed", "failed"]


# =============================================================================
# Research Session Schemas
# =============================================================================


class SessionCreate(BaseModel):
    """Request schema for creating a research session."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the research session",
        examples=["ML Research Papers"],
    )
    description: str | None = Field(
        default=None,
        max_length=5000,
        description="Optional description of the session",
        examples=["Papers about transformer architectures"],
    )


class SessionUpdate(BaseModel):
    """Request schema for updating a research session."""

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Updated name",
    )
    description: str | None = Field(
        default=None,
        max_length=5000,
        description="Updated description",
    )


class SessionResponse(BaseModel):
    """Response schema for a research session."""

    id: int = Field(..., description="Unique session identifier")
    name: str = Field(..., description="Session name")
    description: str | None = Field(None, description="Session description")
    article_count: int = Field(0, description="Number of articles in session")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "name": "ML Research Papers",
                    "description": "Papers about transformer architectures",
                    "article_count": 5,
                    "created_at": "2026-01-19T10:00:00Z",
                    "updated_at": "2026-01-19T10:00:00Z",
                }
            ]
        },
    }


class SessionListResponse(BaseModel):
    """Response schema for listing research sessions."""

    items: list[SessionResponse] = Field(..., description="List of sessions")
    total: int = Field(..., description="Total number of sessions")


# =============================================================================
# Article Schemas
# =============================================================================


class AddUrlRequest(BaseModel):
    """Request schema for adding an article via URL.

    Design Decision: URL Validation
    --------------------------------
    Using Pydantic's HttpUrl type for validation because:
    - Automatic scheme validation (http/https)
    - Well-tested URL parsing
    - Clear error messages for invalid URLs

    The title field is optional and will be extracted from the page if not provided.
    """

    url: HttpUrl = Field(
        ...,
        description="URL of the article to add",
        examples=["https://arxiv.org/abs/2301.00001"],
    )
    title: str | None = Field(
        default=None,
        max_length=512,
        description="Optional title (auto-extracted if not provided)",
        examples=["Attention Is All You Need"],
    )

    @field_validator("url")
    @classmethod
    def validate_url_scheme(cls, v: HttpUrl) -> HttpUrl:
        """Ensure URL uses http or https scheme."""
        if v.scheme not in ("http", "https"):
            raise ValueError("URL must use http or https scheme")
        return v


class UploadFileResponse(BaseModel):
    """Response schema after file upload.

    Returned immediately after upload, before content extraction.
    """

    id: int = Field(..., description="Article ID")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    extraction_status: ExtractionStatus = Field(
        "pending",
        description="Content extraction status",
    )
    created_at: datetime = Field(..., description="Upload timestamp")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 42,
                    "filename": "research_paper.pdf",
                    "size_bytes": 1048576,
                    "extraction_status": "pending",
                    "created_at": "2026-01-19T10:00:00Z",
                }
            ]
        },
    }


class ArticleResponse(BaseModel):
    """Response schema for a single article.

    Used for both URL and file articles with appropriate fields populated.
    """

    id: int = Field(..., description="Unique article identifier")
    session_id: int = Field(..., description="Parent session ID")
    type: ArticleType = Field(..., description="Article type (url or file)")
    original_url: str | None = Field(None, description="Source URL (for url type)")
    original_filename: str | None = Field(
        None,
        description="Original filename (for file type)",
    )
    title: str | None = Field(None, description="Article title")
    extraction_status: ExtractionStatus = Field(
        ...,
        description="Content extraction status",
    )
    has_content: bool = Field(
        False,
        description="Whether extracted content is available",
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "session_id": 1,
                    "type": "url",
                    "original_url": "https://arxiv.org/abs/2301.00001",
                    "original_filename": None,
                    "title": "Attention Is All You Need",
                    "extraction_status": "completed",
                    "has_content": True,
                    "created_at": "2026-01-19T10:00:00Z",
                    "updated_at": "2026-01-19T10:30:00Z",
                }
            ]
        },
    }


class ArticleListResponse(BaseModel):
    """Response schema for listing articles in a session."""

    items: list[ArticleResponse] = Field(..., description="List of articles")
    total: int = Field(..., description="Total number of articles in session")
    session_id: int = Field(..., description="Session ID")


class ArticleContentResponse(BaseModel):
    """Response schema for article extracted content.

    Returns the extracted text content from the article.
    Only available when extraction_status is 'completed'.
    """

    id: int = Field(..., description="Article ID")
    title: str | None = Field(None, description="Article title")
    content_text: str = Field(..., description="Extracted text content")
    extraction_status: ExtractionStatus = Field(..., description="Extraction status")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "title": "Attention Is All You Need",
                    "content_text": "Abstract: The dominant sequence transduction models...",
                    "extraction_status": "completed",
                }
            ]
        },
    }


# =============================================================================
# Error Schemas
# =============================================================================


class ErrorDetail(BaseModel):
    """Standard error detail schema."""

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict | None = Field(None, description="Additional error context")


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: ErrorDetail

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "Article not found",
                        "details": {"article_id": 999},
                    }
                }
            ]
        },
    }
```

#### Update Schemas __init__.py

Update `src/article_mind_service/schemas/__init__.py`:

```python
"""Pydantic schemas for API request/response models."""

from .article import (
    AddUrlRequest,
    ArticleContentResponse,
    ArticleListResponse,
    ArticleResponse,
    ArticleType,
    ErrorDetail,
    ErrorResponse,
    ExtractionStatus,
    SessionCreate,
    SessionListResponse,
    SessionResponse,
    SessionUpdate,
    UploadFileResponse,
)
from .health import HealthResponse

__all__ = [
    # Health
    "HealthResponse",
    # Sessions
    "SessionCreate",
    "SessionUpdate",
    "SessionResponse",
    "SessionListResponse",
    # Articles
    "AddUrlRequest",
    "UploadFileResponse",
    "ArticleResponse",
    "ArticleListResponse",
    "ArticleContentResponse",
    # Enums
    "ArticleType",
    "ExtractionStatus",
    # Errors
    "ErrorDetail",
    "ErrorResponse",
]
```

---

## 4. API Endpoints

### Sessions Router

Create `src/article_mind_service/routers/sessions.py`:

```python
"""Research session API endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.database import get_db
from article_mind_service.models import Article, ResearchSession
from article_mind_service.schemas import (
    SessionCreate,
    SessionListResponse,
    SessionResponse,
    SessionUpdate,
)

router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["sessions"],
)


def _session_to_response(session: ResearchSession, article_count: int) -> SessionResponse:
    """Convert SQLAlchemy model to response schema."""
    return SessionResponse(
        id=session.id,
        name=session.name,
        description=session.description,
        article_count=article_count,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create research session",
    description="Create a new research session to group articles together.",
)
async def create_session(
    data: SessionCreate,
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Create a new research session."""
    session = ResearchSession(
        name=data.name,
        description=data.description,
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)

    return _session_to_response(session, article_count=0)


@router.get(
    "",
    response_model=SessionListResponse,
    summary="List research sessions",
    description="Get all active research sessions (excludes soft-deleted).",
)
async def list_sessions(
    db: AsyncSession = Depends(get_db),
) -> SessionListResponse:
    """List all active research sessions."""
    # Query sessions with article count
    stmt = (
        select(
            ResearchSession,
            func.count(Article.id).filter(Article.deleted_at.is_(None)).label("article_count"),
        )
        .outerjoin(Article, Article.session_id == ResearchSession.id)
        .where(ResearchSession.deleted_at.is_(None))
        .group_by(ResearchSession.id)
        .order_by(ResearchSession.updated_at.desc())
    )

    result = await db.execute(stmt)
    rows = result.all()

    items = [_session_to_response(row[0], row[1]) for row in rows]

    return SessionListResponse(items=items, total=len(items))


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get research session",
    description="Get a specific research session by ID.",
    responses={404: {"description": "Session not found"}},
)
async def get_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Get a specific research session."""
    stmt = (
        select(
            ResearchSession,
            func.count(Article.id).filter(Article.deleted_at.is_(None)).label("article_count"),
        )
        .outerjoin(Article, Article.session_id == ResearchSession.id)
        .where(
            ResearchSession.id == session_id,
            ResearchSession.deleted_at.is_(None),
        )
        .group_by(ResearchSession.id)
    )

    result = await db.execute(stmt)
    row = result.first()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    return _session_to_response(row[0], row[1])


@router.patch(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Update research session",
    description="Update session name and/or description.",
    responses={404: {"description": "Session not found"}},
)
async def update_session(
    session_id: int,
    data: SessionUpdate,
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Update a research session."""
    stmt = select(ResearchSession).where(
        ResearchSession.id == session_id,
        ResearchSession.deleted_at.is_(None),
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Update fields if provided
    if data.name is not None:
        session.name = data.name
    if data.description is not None:
        session.description = data.description

    await db.flush()
    await db.refresh(session)

    # Get article count
    count_stmt = select(func.count(Article.id)).where(
        Article.session_id == session_id,
        Article.deleted_at.is_(None),
    )
    count_result = await db.execute(count_stmt)
    article_count = count_result.scalar() or 0

    return _session_to_response(session, article_count)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete research session",
    description="Soft delete a research session and all its articles.",
    responses={404: {"description": "Session not found"}},
)
async def delete_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Soft delete a research session."""
    stmt = select(ResearchSession).where(
        ResearchSession.id == session_id,
        ResearchSession.deleted_at.is_(None),
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Soft delete session
    now = datetime.now(timezone.utc)
    session.deleted_at = now

    # Soft delete all articles in session
    articles_stmt = select(Article).where(
        Article.session_id == session_id,
        Article.deleted_at.is_(None),
    )
    articles_result = await db.execute(articles_stmt)
    for article in articles_result.scalars():
        article.deleted_at = now

    await db.flush()
```

### Articles Router

Create `src/article_mind_service/routers/articles.py`:

```python
"""Article API endpoints."""

import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.config import settings
from article_mind_service.database import get_db
from article_mind_service.models import Article, ResearchSession
from article_mind_service.schemas import (
    AddUrlRequest,
    ArticleContentResponse,
    ArticleListResponse,
    ArticleResponse,
    UploadFileResponse,
)

router = APIRouter(
    prefix="/api/v1/sessions/{session_id}/articles",
    tags=["articles"],
)


# =============================================================================
# Helper Functions
# =============================================================================


async def _get_session_or_404(
    session_id: int,
    db: AsyncSession,
) -> ResearchSession:
    """Get session or raise 404."""
    stmt = select(ResearchSession).where(
        ResearchSession.id == session_id,
        ResearchSession.deleted_at.is_(None),
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    return session


async def _get_article_or_404(
    session_id: int,
    article_id: int,
    db: AsyncSession,
) -> Article:
    """Get article or raise 404."""
    stmt = select(Article).where(
        Article.id == article_id,
        Article.session_id == session_id,
        Article.deleted_at.is_(None),
    )
    result = await db.execute(stmt)
    article = result.scalar_one_or_none()

    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article {article_id} not found in session {session_id}",
        )

    return article


def _article_to_response(article: Article) -> ArticleResponse:
    """Convert SQLAlchemy model to response schema."""
    return ArticleResponse(
        id=article.id,
        session_id=article.session_id,
        type=article.type,
        original_url=article.original_url,
        original_filename=article.original_filename,
        title=article.title,
        extraction_status=article.extraction_status,
        has_content=article.content_text is not None and len(article.content_text) > 0,
        created_at=article.created_at,
        updated_at=article.updated_at,
    )


def _get_upload_dir(session_id: int, article_id: int) -> Path:
    """Get upload directory for an article."""
    base_path = Path(settings.upload_base_path)
    return base_path / str(session_id) / str(article_id)


def _ensure_upload_dir(session_id: int, article_id: int) -> Path:
    """Create and return upload directory."""
    upload_dir = _get_upload_dir(session_id, article_id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/url",
    response_model=ArticleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add article from URL",
    description="Add an article to the session by providing its URL. Content extraction happens asynchronously.",
    responses={404: {"description": "Session not found"}},
)
async def add_url_article(
    session_id: int,
    data: AddUrlRequest,
    db: AsyncSession = Depends(get_db),
) -> ArticleResponse:
    """Add an article from a URL."""
    # Verify session exists
    await _get_session_or_404(session_id, db)

    # Create article
    article = Article(
        session_id=session_id,
        type="url",
        original_url=str(data.url),
        title=data.title,
        extraction_status="pending",
    )
    db.add(article)
    await db.flush()
    await db.refresh(article)

    return _article_to_response(article)


@router.post(
    "/upload",
    response_model=UploadFileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload article file",
    description="Upload a file (PDF, DOCX, TXT, etc.) as an article. Content extraction happens asynchronously.",
    responses={
        404: {"description": "Session not found"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
    },
)
async def upload_article_file(
    session_id: int,
    file: UploadFile = File(..., description="File to upload"),
    db: AsyncSession = Depends(get_db),
) -> UploadFileResponse:
    """Upload a file as an article.

    Design Decision: File Storage Strategy
    --------------------------------------
    Files are stored in: data/uploads/{session_id}/{article_id}/original.{ext}

    This structure enables:
    - Easy cleanup when session is deleted
    - Isolation between sessions
    - Simple file serving without DB lookup
    - Future: multiple files per article (e.g., attachments)

    The original file extension is preserved for content-type detection
    during extraction.
    """
    # Verify session exists
    await _get_session_or_404(session_id, db)

    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    # Check file size (read size from file object)
    file_content = await file.read()
    file_size = len(file_content)

    max_size = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB",
        )

    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    # Create article record first to get ID
    article = Article(
        session_id=session_id,
        type="file",
        original_filename=file.filename,
        extraction_status="pending",
    )
    db.add(article)
    await db.flush()
    await db.refresh(article)

    # Save file to filesystem
    try:
        upload_dir = _ensure_upload_dir(session_id, article.id)
        file_path = upload_dir / f"original{file_ext}"

        with open(file_path, "wb") as f:
            f.write(file_content)

        # Update article with storage path (relative to base path)
        article.storage_path = f"{session_id}/{article.id}/original{file_ext}"
        await db.flush()

    except OSError as e:
        # Rollback article creation on file save failure
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {e}",
        )

    return UploadFileResponse(
        id=article.id,
        filename=file.filename,
        size_bytes=file_size,
        extraction_status=article.extraction_status,
        created_at=article.created_at,
    )


@router.get(
    "",
    response_model=ArticleListResponse,
    summary="List articles in session",
    description="Get all active articles in a research session.",
    responses={404: {"description": "Session not found"}},
)
async def list_articles(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> ArticleListResponse:
    """List all articles in a session."""
    # Verify session exists
    await _get_session_or_404(session_id, db)

    # Query articles
    stmt = (
        select(Article)
        .where(
            Article.session_id == session_id,
            Article.deleted_at.is_(None),
        )
        .order_by(Article.created_at.desc())
    )
    result = await db.execute(stmt)
    articles = result.scalars().all()

    return ArticleListResponse(
        items=[_article_to_response(a) for a in articles],
        total=len(articles),
        session_id=session_id,
    )


@router.get(
    "/{article_id}",
    response_model=ArticleResponse,
    summary="Get article details",
    description="Get detailed information about a specific article.",
    responses={404: {"description": "Article not found"}},
)
async def get_article(
    session_id: int,
    article_id: int,
    db: AsyncSession = Depends(get_db),
) -> ArticleResponse:
    """Get a specific article."""
    article = await _get_article_or_404(session_id, article_id, db)
    return _article_to_response(article)


@router.delete(
    "/{article_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete article",
    description="Soft delete an article from the session.",
    responses={404: {"description": "Article not found"}},
)
async def delete_article(
    session_id: int,
    article_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Soft delete an article."""
    article = await _get_article_or_404(session_id, article_id, db)

    # Soft delete
    article.deleted_at = datetime.now(timezone.utc)
    await db.flush()

    # Note: We don't delete files on soft delete
    # A cleanup job can remove files for hard-deleted articles later


@router.get(
    "/{article_id}/content",
    response_model=ArticleContentResponse,
    summary="Get extracted content",
    description="Get the extracted text content of an article. Only available when extraction is completed.",
    responses={
        404: {"description": "Article not found"},
        400: {"description": "Content not available (extraction not completed)"},
    },
)
async def get_article_content(
    session_id: int,
    article_id: int,
    db: AsyncSession = Depends(get_db),
) -> ArticleContentResponse:
    """Get extracted content for an article."""
    article = await _get_article_or_404(session_id, article_id, db)

    if article.extraction_status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Content not available. Extraction status: {article.extraction_status}",
        )

    if not article.content_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No content available for this article",
        )

    return ArticleContentResponse(
        id=article.id,
        title=article.title,
        content_text=article.content_text,
        extraction_status=article.extraction_status,
    )
```

### Update Routers __init__.py

Update `src/article_mind_service/routers/__init__.py`:

```python
"""FastAPI routers."""

from .articles import router as articles_router
from .health import router as health_router
from .sessions import router as sessions_router

__all__ = [
    "articles_router",
    "health_router",
    "sessions_router",
]
```

### Update Main App

Update `src/article_mind_service/main.py` to include new routers:

```python
"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .config import settings
from .database import engine
from .routers import articles_router, health_router, sessions_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager."""
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")

    # Test database connection
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        print("Database connection verified")
    except Exception as e:
        print(f"Database connection failed: {e}")
        print("Service will start but /health will report degraded status")

    yield

    # Shutdown
    print("Shutting down application")
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# Health check router (no prefix - as per API contract)
app.include_router(health_router)

# API routers
app.include_router(sessions_router)
app.include_router(articles_router)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Article Mind Service API"}
```

### Update Config

Update `src/article_mind_service/config.py` to include upload settings:

```python
"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = "Article Mind Service"
    app_version: str = "0.1.0"
    debug: bool = False

    # Database
    database_url: str = "postgresql://article_mind:article_mind@localhost:5432/article_mind"

    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:5173"]

    # File Upload
    upload_base_path: str = "data/uploads"
    max_upload_size_mb: int = 50

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
```

---

## 5. File Storage Strategy

### Directory Structure

```
data/
└── uploads/
    └── {session_id}/
        └── {article_id}/
            ├── original.pdf       # Original uploaded file
            └── extracted.txt      # Extracted text (future)
```

### Environment Configuration

Add to `.env.example`:

```env
# File Upload Configuration
UPLOAD_BASE_PATH=data/uploads
MAX_UPLOAD_SIZE_MB=50
```

### .gitignore Addition

Add to `.gitignore`:

```gitignore
# Uploaded files
data/uploads/
```

---

## 6. UI Components (Svelte 5 Runes)

### API Client Extension

Create `src/lib/api/sessions.ts`:

```typescript
/**
 * Research sessions API client
 */

import { apiClient } from './client';
import type { components } from './generated';

// Type aliases
type SessionResponse = components['schemas']['SessionResponse'];
type SessionListResponse = components['schemas']['SessionListResponse'];
type SessionCreate = components['schemas']['SessionCreate'];
type SessionUpdate = components['schemas']['SessionUpdate'];

/**
 * List all research sessions
 */
export async function listSessions(): Promise<SessionListResponse> {
  return apiClient.get<SessionListResponse>('/api/v1/sessions');
}

/**
 * Get a specific session
 */
export async function getSession(sessionId: number): Promise<SessionResponse> {
  return apiClient.get<SessionResponse>(`/api/v1/sessions/${sessionId}`);
}

/**
 * Create a new session
 */
export async function createSession(data: SessionCreate): Promise<SessionResponse> {
  return apiClient.post<SessionResponse>('/api/v1/sessions', data);
}

/**
 * Update a session
 */
export async function updateSession(
  sessionId: number,
  data: SessionUpdate
): Promise<SessionResponse> {
  return apiClient.patch<SessionResponse>(`/api/v1/sessions/${sessionId}`, data);
}

/**
 * Delete a session (soft delete)
 */
export async function deleteSession(sessionId: number): Promise<void> {
  return apiClient.delete(`/api/v1/sessions/${sessionId}`);
}
```

Create `src/lib/api/articles.ts`:

```typescript
/**
 * Articles API client
 */

import { apiClient } from './client';
import type { components } from './generated';

// Type aliases
type ArticleResponse = components['schemas']['ArticleResponse'];
type ArticleListResponse = components['schemas']['ArticleListResponse'];
type ArticleContentResponse = components['schemas']['ArticleContentResponse'];
type AddUrlRequest = components['schemas']['AddUrlRequest'];
type UploadFileResponse = components['schemas']['UploadFileResponse'];

/**
 * List articles in a session
 */
export async function listArticles(sessionId: number): Promise<ArticleListResponse> {
  return apiClient.get<ArticleListResponse>(`/api/v1/sessions/${sessionId}/articles`);
}

/**
 * Get a specific article
 */
export async function getArticle(
  sessionId: number,
  articleId: number
): Promise<ArticleResponse> {
  return apiClient.get<ArticleResponse>(
    `/api/v1/sessions/${sessionId}/articles/${articleId}`
  );
}

/**
 * Add article from URL
 */
export async function addUrlArticle(
  sessionId: number,
  data: AddUrlRequest
): Promise<ArticleResponse> {
  return apiClient.post<ArticleResponse>(
    `/api/v1/sessions/${sessionId}/articles/url`,
    data
  );
}

/**
 * Upload article file
 */
export async function uploadArticleFile(
  sessionId: number,
  file: File
): Promise<UploadFileResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(
    `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/api/v1/sessions/${sessionId}/articles/upload`,
    {
      method: 'POST',
      body: formData,
    }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Upload failed');
  }

  return response.json();
}

/**
 * Delete article (soft delete)
 */
export async function deleteArticle(
  sessionId: number,
  articleId: number
): Promise<void> {
  return apiClient.delete(`/api/v1/sessions/${sessionId}/articles/${articleId}`);
}

/**
 * Get article extracted content
 */
export async function getArticleContent(
  sessionId: number,
  articleId: number
): Promise<ArticleContentResponse> {
  return apiClient.get<ArticleContentResponse>(
    `/api/v1/sessions/${sessionId}/articles/${articleId}/content`
  );
}
```

### ArticleList Component

Create `src/lib/components/ArticleList.svelte`:

```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  import { listArticles, deleteArticle } from '$lib/api/articles';
  import type { components } from '$lib/api/generated';
  import ArticleCard from './ArticleCard.svelte';

  type ArticleResponse = components['schemas']['ArticleResponse'];
  type ArticleListResponse = components['schemas']['ArticleListResponse'];

  // Props
  interface Props {
    sessionId: number;
    onArticleDeleted?: () => void;
  }
  let { sessionId, onArticleDeleted }: Props = $props();

  // State
  let articles = $state<ArticleResponse[]>([]);
  let loading = $state(true);
  let error = $state<string | null>(null);

  // Fetch articles
  async function fetchArticles() {
    try {
      loading = true;
      error = null;
      const response = await listArticles(sessionId);
      articles = response.items;
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load articles';
      console.error('Error fetching articles:', err);
    } finally {
      loading = false;
    }
  }

  // Handle delete
  async function handleDelete(articleId: number) {
    if (!confirm('Are you sure you want to delete this article?')) {
      return;
    }

    try {
      await deleteArticle(sessionId, articleId);
      articles = articles.filter(a => a.id !== articleId);
      onArticleDeleted?.();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete article');
    }
  }

  // Expose refresh method
  export function refresh() {
    fetchArticles();
  }

  onMount(() => {
    fetchArticles();
  });
</script>

<div class="article-list">
  {#if loading}
    <div class="loading">Loading articles...</div>
  {:else if error}
    <div class="error">
      <p>{error}</p>
      <button onclick={fetchArticles}>Retry</button>
    </div>
  {:else if articles.length === 0}
    <div class="empty">
      <p>No articles yet. Add one using the form above.</p>
    </div>
  {:else}
    <div class="articles-grid">
      {#each articles as article (article.id)}
        <ArticleCard
          {article}
          onDelete={() => handleDelete(article.id)}
        />
      {/each}
    </div>
  {/if}
</div>

<style>
  .article-list {
    width: 100%;
  }

  .loading,
  .error,
  .empty {
    text-align: center;
    padding: 2rem;
    color: #666;
  }

  .error {
    color: #ef4444;
  }

  .error button {
    margin-top: 1rem;
    padding: 0.5rem 1rem;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  }

  .articles-grid {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  }
</style>
```

### ArticleCard Component

Create `src/lib/components/ArticleCard.svelte`:

```svelte
<script lang="ts">
  import type { components } from '$lib/api/generated';

  type ArticleResponse = components['schemas']['ArticleResponse'];
  type ExtractionStatus = ArticleResponse['extraction_status'];

  // Props
  interface Props {
    article: ArticleResponse;
    onDelete: () => void;
  }
  let { article, onDelete }: Props = $props();

  // Derived state
  let displayName = $derived(
    article.title ||
    (article.type === 'url' ? article.original_url?.slice(0, 60) : article.original_filename) ||
    `Article #${article.id}`
  );

  let statusColor = $derived(getStatusColor(article.extraction_status));
  let statusText = $derived(getStatusText(article.extraction_status));

  function getStatusColor(status: ExtractionStatus): string {
    switch (status) {
      case 'completed': return '#22c55e';
      case 'processing': return '#3b82f6';
      case 'failed': return '#ef4444';
      default: return '#9ca3af';
    }
  }

  function getStatusText(status: ExtractionStatus): string {
    switch (status) {
      case 'completed': return 'Ready';
      case 'processing': return 'Processing';
      case 'failed': return 'Failed';
      default: return 'Pending';
    }
  }

  function formatDate(date: string): string {
    return new Date(date).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  }
</script>

<div class="article-card">
  <div class="card-header">
    <span class="type-badge" class:url={article.type === 'url'} class:file={article.type === 'file'}>
      {article.type === 'url' ? 'URL' : 'File'}
    </span>
    <button
      class="delete-btn"
      onclick={onDelete}
      title="Delete article"
      aria-label="Delete article"
    >
      &times;
    </button>
  </div>

  <h3 class="title">{displayName}</h3>

  {#if article.type === 'url' && article.original_url}
    <a href={article.original_url} target="_blank" rel="noopener noreferrer" class="source-link">
      View source
    </a>
  {/if}

  <div class="card-footer">
    <span class="status" style="--status-color: {statusColor}">
      {statusText}
    </span>
    <span class="date">{formatDate(article.created_at)}</span>
  </div>
</div>

<style>
  .article-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .type-badge {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    text-transform: uppercase;
  }

  .type-badge.url {
    background: #dbeafe;
    color: #1d4ed8;
  }

  .type-badge.file {
    background: #f3e8ff;
    color: #7c3aed;
  }

  .delete-btn {
    background: none;
    border: none;
    font-size: 1.25rem;
    color: #9ca3af;
    cursor: pointer;
    padding: 0.25rem;
    line-height: 1;
  }

  .delete-btn:hover {
    color: #ef4444;
  }

  .title {
    font-size: 1rem;
    font-weight: 600;
    margin: 0;
    color: #1f2937;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .source-link {
    font-size: 0.875rem;
    color: #3b82f6;
    text-decoration: none;
  }

  .source-link:hover {
    text-decoration: underline;
  }

  .card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: auto;
    padding-top: 0.5rem;
    border-top: 1px solid #f3f4f6;
  }

  .status {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--status-color);
  }

  .date {
    font-size: 0.75rem;
    color: #9ca3af;
  }
</style>
```

### AddUrlForm Component

Create `src/lib/components/AddUrlForm.svelte`:

```svelte
<script lang="ts">
  import { addUrlArticle } from '$lib/api/articles';

  // Props
  interface Props {
    sessionId: number;
    onArticleAdded?: () => void;
  }
  let { sessionId, onArticleAdded }: Props = $props();

  // State
  let url = $state('');
  let title = $state('');
  let loading = $state(false);
  let error = $state<string | null>(null);

  async function handleSubmit(event: Event) {
    event.preventDefault();

    if (!url.trim()) {
      error = 'URL is required';
      return;
    }

    try {
      loading = true;
      error = null;

      await addUrlArticle(sessionId, {
        url: url.trim(),
        title: title.trim() || undefined,
      });

      // Reset form
      url = '';
      title = '';

      onArticleAdded?.();
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to add article';
    } finally {
      loading = false;
    }
  }
</script>

<form class="add-url-form" onsubmit={handleSubmit}>
  <h3>Add Article from URL</h3>

  {#if error}
    <div class="error-message">{error}</div>
  {/if}

  <div class="form-group">
    <label for="url">URL *</label>
    <input
      type="url"
      id="url"
      bind:value={url}
      placeholder="https://example.com/article"
      required
      disabled={loading}
    />
  </div>

  <div class="form-group">
    <label for="title">Title (optional)</label>
    <input
      type="text"
      id="title"
      bind:value={title}
      placeholder="Article title"
      disabled={loading}
    />
  </div>

  <button type="submit" disabled={loading}>
    {loading ? 'Adding...' : 'Add URL'}
  </button>
</form>

<style>
  .add-url-form {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1.5rem;
  }

  h3 {
    margin: 0 0 1rem 0;
    font-size: 1rem;
    color: #374151;
  }

  .error-message {
    background: #fef2f2;
    color: #dc2626;
    padding: 0.75rem;
    border-radius: 4px;
    margin-bottom: 1rem;
    font-size: 0.875rem;
  }

  .form-group {
    margin-bottom: 1rem;
  }

  label {
    display: block;
    font-size: 0.875rem;
    font-weight: 500;
    color: #374151;
    margin-bottom: 0.25rem;
  }

  input {
    width: 100%;
    padding: 0.5rem 0.75rem;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    font-size: 0.875rem;
  }

  input:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }

  input:disabled {
    background: #f3f4f6;
    cursor: not-allowed;
  }

  button {
    width: 100%;
    padding: 0.75rem;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 4px;
    font-weight: 500;
    cursor: pointer;
  }

  button:hover:not(:disabled) {
    background: #2563eb;
  }

  button:disabled {
    background: #9ca3af;
    cursor: not-allowed;
  }
</style>
```

### FileUploadDropzone Component

Create `src/lib/components/FileUploadDropzone.svelte`:

```svelte
<script lang="ts">
  import { uploadArticleFile } from '$lib/api/articles';

  // Props
  interface Props {
    sessionId: number;
    onArticleAdded?: () => void;
  }
  let { sessionId, onArticleAdded }: Props = $props();

  // State
  let loading = $state(false);
  let error = $state<string | null>(null);
  let isDragging = $state(false);

  const allowedTypes = [
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/msword',
    'text/plain',
    'text/markdown',
    'text/html',
  ];

  const allowedExtensions = ['.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.htm'];

  function isValidFile(file: File): boolean {
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    return allowedExtensions.includes(extension) || allowedTypes.includes(file.type);
  }

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;

    const file = files[0];

    if (!isValidFile(file)) {
      error = `Invalid file type. Allowed: ${allowedExtensions.join(', ')}`;
      return;
    }

    try {
      loading = true;
      error = null;

      await uploadArticleFile(sessionId, file);
      onArticleAdded?.();
    } catch (err) {
      error = err instanceof Error ? err.message : 'Upload failed';
    } finally {
      loading = false;
    }
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    isDragging = false;
    handleFiles(event.dataTransfer?.files ?? null);
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    isDragging = true;
  }

  function handleDragLeave() {
    isDragging = false;
  }

  function handleFileInput(event: Event) {
    const input = event.target as HTMLInputElement;
    handleFiles(input.files);
    input.value = ''; // Reset input
  }
</script>

<div
  class="dropzone"
  class:dragging={isDragging}
  class:loading
  ondrop={handleDrop}
  ondragover={handleDragOver}
  ondragleave={handleDragLeave}
  role="region"
  aria-label="File upload dropzone"
>
  {#if loading}
    <div class="loading-content">
      <span class="spinner"></span>
      <p>Uploading...</p>
    </div>
  {:else}
    <div class="dropzone-content">
      <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
      <p>Drag and drop a file here, or</p>
      <label class="file-input-label">
        <span>Browse files</span>
        <input
          type="file"
          accept={allowedExtensions.join(',')}
          onchange={handleFileInput}
          class="visually-hidden"
        />
      </label>
      <p class="hint">Supported: PDF, DOCX, DOC, TXT, MD, HTML</p>
    </div>
  {/if}

  {#if error}
    <div class="error-message">{error}</div>
  {/if}
</div>

<style>
  .dropzone {
    border: 2px dashed #d1d5db;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    transition: all 0.2s;
    background: #fafafa;
  }

  .dropzone.dragging {
    border-color: #3b82f6;
    background: #eff6ff;
  }

  .dropzone.loading {
    opacity: 0.7;
    pointer-events: none;
  }

  .dropzone-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
  }

  .upload-icon {
    width: 48px;
    height: 48px;
    color: #9ca3af;
  }

  .dropzone-content p {
    margin: 0;
    color: #6b7280;
  }

  .file-input-label {
    display: inline-block;
    padding: 0.5rem 1rem;
    background: #3b82f6;
    color: white;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 500;
    margin-top: 0.5rem;
  }

  .file-input-label:hover {
    background: #2563eb;
  }

  .visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  .hint {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 0.5rem;
  }

  .loading-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid #e5e7eb;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .error-message {
    margin-top: 1rem;
    padding: 0.75rem;
    background: #fef2f2;
    color: #dc2626;
    border-radius: 4px;
    font-size: 0.875rem;
  }
</style>
```

### Session Detail Page

Create `src/routes/sessions/[id]/+page.svelte`:

```svelte
<script lang="ts">
  import { page } from '$app/stores';
  import { onMount } from 'svelte';
  import { getSession } from '$lib/api/sessions';
  import type { components } from '$lib/api/generated';
  import ArticleList from '$lib/components/ArticleList.svelte';
  import AddUrlForm from '$lib/components/AddUrlForm.svelte';
  import FileUploadDropzone from '$lib/components/FileUploadDropzone.svelte';

  type SessionResponse = components['schemas']['SessionResponse'];

  // Get session ID from URL
  let sessionId = $derived(parseInt($page.params.id, 10));

  // State
  let session = $state<SessionResponse | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let articleListRef = $state<ArticleList | null>(null);

  async function fetchSession() {
    try {
      loading = true;
      error = null;
      session = await getSession(sessionId);
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load session';
    } finally {
      loading = false;
    }
  }

  function handleArticleAdded() {
    articleListRef?.refresh();
    // Refresh session to update article count
    fetchSession();
  }

  onMount(() => {
    fetchSession();
  });
</script>

<svelte:head>
  <title>{session?.name || 'Session'} - Article Mind</title>
</svelte:head>

<div class="session-page">
  {#if loading}
    <div class="loading">Loading session...</div>
  {:else if error}
    <div class="error">
      <p>{error}</p>
      <a href="/sessions">Back to sessions</a>
    </div>
  {:else if session}
    <header class="session-header">
      <div>
        <h1>{session.name}</h1>
        {#if session.description}
          <p class="description">{session.description}</p>
        {/if}
        <p class="meta">{session.article_count} article{session.article_count !== 1 ? 's' : ''}</p>
      </div>
      <a href="/sessions" class="back-link">Back to Sessions</a>
    </header>

    <section class="add-article-section">
      <h2>Add Articles</h2>
      <div class="add-forms">
        <AddUrlForm {sessionId} onArticleAdded={handleArticleAdded} />
        <FileUploadDropzone {sessionId} onArticleAdded={handleArticleAdded} />
      </div>
    </section>

    <section class="articles-section">
      <h2>Articles</h2>
      <ArticleList
        bind:this={articleListRef}
        {sessionId}
        onArticleDeleted={handleArticleAdded}
      />
    </section>
  {/if}
</div>

<style>
  .session-page {
    max-width: 1000px;
    margin: 0 auto;
  }

  .loading,
  .error {
    text-align: center;
    padding: 2rem;
  }

  .error {
    color: #dc2626;
  }

  .error a {
    display: inline-block;
    margin-top: 1rem;
    color: #3b82f6;
  }

  .session-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e5e7eb;
  }

  h1 {
    margin: 0 0 0.5rem 0;
    color: #1f2937;
  }

  .description {
    margin: 0 0 0.5rem 0;
    color: #6b7280;
  }

  .meta {
    margin: 0;
    font-size: 0.875rem;
    color: #9ca3af;
  }

  .back-link {
    color: #3b82f6;
    text-decoration: none;
    font-size: 0.875rem;
  }

  .back-link:hover {
    text-decoration: underline;
  }

  section {
    margin-bottom: 2rem;
  }

  h2 {
    font-size: 1.125rem;
    color: #374151;
    margin: 0 0 1rem 0;
  }

  .add-forms {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  }

  .articles-section {
    margin-top: 2rem;
  }
</style>
```

---

## 7. Implementation Steps

### Phase 1: Database Setup (2-3 hours)

1. **Create SQLAlchemy models**
   - Create `models/research_session.py`
   - Create `models/article.py`
   - Update `models/__init__.py`

2. **Generate Alembic migration**
   ```bash
   cd article-mind-service
   make migrate-create MSG="add research sessions and articles tables"
   ```

3. **Review and apply migration**
   ```bash
   # Review the generated migration file
   # Then apply:
   make migrate
   ```

4. **Verify tables created**
   ```bash
   docker exec -it article-mind-postgres psql -U article_mind -d article_mind -c '\dt'
   ```

### Phase 2: API Implementation (3-4 hours)

1. **Create Pydantic schemas**
   - Create `schemas/article.py`
   - Update `schemas/__init__.py`

2. **Update config for file uploads**
   - Add `upload_base_path` and `max_upload_size_mb` to `config.py`
   - Update `.env.example`

3. **Create sessions router**
   - Create `routers/sessions.py`

4. **Create articles router**
   - Create `routers/articles.py`

5. **Update main app**
   - Register new routers in `main.py`
   - Update `routers/__init__.py`

6. **Create upload directory**
   ```bash
   mkdir -p data/uploads
   ```

### Phase 3: API Testing (1-2 hours)

1. **Write tests for sessions endpoints**
2. **Write tests for articles endpoints**
3. **Run full test suite**
   ```bash
   make test
   ```

### Phase 4: UI Implementation (2-3 hours)

1. **Generate TypeScript types**
   ```bash
   cd article-mind-ui
   npm run gen:api
   ```

2. **Create API client functions**
   - Create `lib/api/sessions.ts`
   - Create `lib/api/articles.ts`

3. **Create UI components**
   - Create `ArticleCard.svelte`
   - Create `ArticleList.svelte`
   - Create `AddUrlForm.svelte`
   - Create `FileUploadDropzone.svelte`

4. **Create session detail page**
   - Create `routes/sessions/[id]/+page.svelte`

### Phase 5: Integration Testing (1 hour)

1. **Start both services**
   ```bash
   # Terminal 1
   cd article-mind-service && make dev

   # Terminal 2
   cd article-mind-ui && make dev
   ```

2. **Manual testing**
   - Create session
   - Add URL article
   - Upload file article
   - View article list
   - Delete article

---

## 8. Testing

### API Tests

Create `tests/test_sessions.py`:

```python
"""Tests for research sessions API."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_session(async_client: AsyncClient) -> None:
    """Test creating a research session."""
    response = await async_client.post(
        "/api/v1/sessions",
        json={"name": "Test Session", "description": "A test session"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Session"
    assert data["description"] == "A test session"
    assert data["article_count"] == 0
    assert "id" in data


@pytest.mark.asyncio
async def test_list_sessions(async_client: AsyncClient) -> None:
    """Test listing research sessions."""
    # Create a session first
    await async_client.post(
        "/api/v1/sessions",
        json={"name": "Test Session"},
    )

    response = await async_client.get("/api/v1/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] >= 1


@pytest.mark.asyncio
async def test_get_session(async_client: AsyncClient) -> None:
    """Test getting a specific session."""
    # Create session
    create_response = await async_client.post(
        "/api/v1/sessions",
        json={"name": "Test Session"},
    )
    session_id = create_response.json()["id"]

    # Get session
    response = await async_client.get(f"/api/v1/sessions/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == session_id
    assert data["name"] == "Test Session"


@pytest.mark.asyncio
async def test_update_session(async_client: AsyncClient) -> None:
    """Test updating a session."""
    # Create session
    create_response = await async_client.post(
        "/api/v1/sessions",
        json={"name": "Original Name"},
    )
    session_id = create_response.json()["id"]

    # Update session
    response = await async_client.patch(
        f"/api/v1/sessions/{session_id}",
        json={"name": "Updated Name"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Name"


@pytest.mark.asyncio
async def test_delete_session(async_client: AsyncClient) -> None:
    """Test soft deleting a session."""
    # Create session
    create_response = await async_client.post(
        "/api/v1/sessions",
        json={"name": "To Delete"},
    )
    session_id = create_response.json()["id"]

    # Delete session
    response = await async_client.delete(f"/api/v1/sessions/{session_id}")
    assert response.status_code == 204

    # Verify session is not found
    get_response = await async_client.get(f"/api/v1/sessions/{session_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_get_nonexistent_session(async_client: AsyncClient) -> None:
    """Test getting a nonexistent session returns 404."""
    response = await async_client.get("/api/v1/sessions/99999")
    assert response.status_code == 404
```

Create `tests/test_articles.py`:

```python
"""Tests for articles API."""

import pytest
from httpx import AsyncClient


@pytest.fixture
async def session_id(async_client: AsyncClient) -> int:
    """Create a session and return its ID."""
    response = await async_client.post(
        "/api/v1/sessions",
        json={"name": "Test Session for Articles"},
    )
    return response.json()["id"]


@pytest.mark.asyncio
async def test_add_url_article(async_client: AsyncClient, session_id: int) -> None:
    """Test adding an article from URL."""
    response = await async_client.post(
        f"/api/v1/sessions/{session_id}/articles/url",
        json={
            "url": "https://example.com/article",
            "title": "Example Article",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["type"] == "url"
    assert data["original_url"] == "https://example.com/article"
    assert data["title"] == "Example Article"
    assert data["extraction_status"] == "pending"


@pytest.mark.asyncio
async def test_list_articles(async_client: AsyncClient, session_id: int) -> None:
    """Test listing articles in a session."""
    # Add an article
    await async_client.post(
        f"/api/v1/sessions/{session_id}/articles/url",
        json={"url": "https://example.com/article"},
    )

    # List articles
    response = await async_client.get(f"/api/v1/sessions/{session_id}/articles")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert len(data["items"]) >= 1


@pytest.mark.asyncio
async def test_get_article(async_client: AsyncClient, session_id: int) -> None:
    """Test getting a specific article."""
    # Create article
    create_response = await async_client.post(
        f"/api/v1/sessions/{session_id}/articles/url",
        json={"url": "https://example.com/article"},
    )
    article_id = create_response.json()["id"]

    # Get article
    response = await async_client.get(
        f"/api/v1/sessions/{session_id}/articles/{article_id}"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == article_id


@pytest.mark.asyncio
async def test_delete_article(async_client: AsyncClient, session_id: int) -> None:
    """Test soft deleting an article."""
    # Create article
    create_response = await async_client.post(
        f"/api/v1/sessions/{session_id}/articles/url",
        json={"url": "https://example.com/article"},
    )
    article_id = create_response.json()["id"]

    # Delete article
    response = await async_client.delete(
        f"/api/v1/sessions/{session_id}/articles/{article_id}"
    )
    assert response.status_code == 204

    # Verify article is not found
    get_response = await async_client.get(
        f"/api/v1/sessions/{session_id}/articles/{article_id}"
    )
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_add_article_to_nonexistent_session(async_client: AsyncClient) -> None:
    """Test adding article to nonexistent session returns 404."""
    response = await async_client.post(
        "/api/v1/sessions/99999/articles/url",
        json={"url": "https://example.com/article"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_article_content_not_ready(
    async_client: AsyncClient, session_id: int
) -> None:
    """Test getting content when extraction not completed."""
    # Create article (status will be pending)
    create_response = await async_client.post(
        f"/api/v1/sessions/{session_id}/articles/url",
        json={"url": "https://example.com/article"},
    )
    article_id = create_response.json()["id"]

    # Try to get content
    response = await async_client.get(
        f"/api/v1/sessions/{session_id}/articles/{article_id}/content"
    )
    assert response.status_code == 400
```

### UI Tests

Create `tests/unit/ArticleCard.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import ArticleCard from '$lib/components/ArticleCard.svelte';

describe('ArticleCard', () => {
  const mockArticle = {
    id: 1,
    session_id: 1,
    type: 'url' as const,
    original_url: 'https://example.com/article',
    original_filename: null,
    title: 'Test Article',
    extraction_status: 'completed' as const,
    has_content: true,
    created_at: '2026-01-19T10:00:00Z',
    updated_at: '2026-01-19T10:00:00Z',
  };

  it('renders article title', () => {
    render(ArticleCard, {
      props: {
        article: mockArticle,
        onDelete: () => {},
      },
    });

    expect(screen.getByText('Test Article')).toBeTruthy();
  });

  it('shows URL badge for URL articles', () => {
    render(ArticleCard, {
      props: {
        article: mockArticle,
        onDelete: () => {},
      },
    });

    expect(screen.getByText('URL')).toBeTruthy();
  });

  it('shows Ready status for completed extraction', () => {
    render(ArticleCard, {
      props: {
        article: mockArticle,
        onDelete: () => {},
      },
    });

    expect(screen.getByText('Ready')).toBeTruthy();
  });
});
```

---

## 9. Acceptance Criteria

### API Acceptance Criteria

- [ ] Sessions CRUD endpoints work correctly
  - [ ] POST /api/v1/sessions creates a session
  - [ ] GET /api/v1/sessions lists all active sessions
  - [ ] GET /api/v1/sessions/{id} returns session details
  - [ ] PATCH /api/v1/sessions/{id} updates session
  - [ ] DELETE /api/v1/sessions/{id} soft deletes session

- [ ] Articles CRUD endpoints work correctly
  - [ ] POST /api/v1/sessions/{id}/articles/url adds URL article
  - [ ] POST /api/v1/sessions/{id}/articles/upload handles file upload
  - [ ] GET /api/v1/sessions/{id}/articles lists articles
  - [ ] GET /api/v1/sessions/{id}/articles/{aid} returns article details
  - [ ] DELETE /api/v1/sessions/{id}/articles/{aid} soft deletes article
  - [ ] GET /api/v1/sessions/{id}/articles/{aid}/content returns extracted content

- [ ] File upload works correctly
  - [ ] Files are saved to configured upload directory
  - [ ] File size limits are enforced
  - [ ] Only allowed file types are accepted
  - [ ] Original filename is preserved in database

- [ ] Soft delete cascade works
  - [ ] Deleting a session soft deletes all its articles
  - [ ] Soft deleted items are excluded from list queries

- [ ] All API tests pass
  - [ ] Sessions tests pass
  - [ ] Articles tests pass
  - [ ] Integration tests pass

### UI Acceptance Criteria

- [ ] Session detail page shows session information
- [ ] Add URL form successfully adds articles
- [ ] File upload dropzone works with drag and drop
- [ ] File upload dropzone works with file browser
- [ ] Article list displays all articles
- [ ] Article cards show correct status indicators
- [ ] Delete button removes articles from list
- [ ] Error states are handled gracefully
- [ ] Loading states are displayed appropriately

### Integration Acceptance Criteria

- [ ] TypeScript types are generated from OpenAPI spec
- [ ] Frontend types match backend schemas
- [ ] CORS works correctly for all endpoints
- [ ] File upload works end-to-end
- [ ] Session deletion cascades to UI

---

## 10. Common Pitfalls

### Pitfall 1: Missing Foreign Key Cascade

**Problem:** Deleting session fails due to foreign key constraint

**Solution:** Ensure `ondelete="CASCADE"` is set on foreign key:
```python
session_id: Mapped[int] = mapped_column(
    ForeignKey("research_sessions.id", ondelete="CASCADE"),
    ...
)
```

### Pitfall 2: File Upload CORS

**Problem:** File uploads fail with CORS error

**Solution:** Ensure `CORSMiddleware` allows multipart/form-data content type.

### Pitfall 3: Async Session Not Committed

**Problem:** Data not persisted after create/update

**Solution:** Ensure `get_db` dependency commits on success:
```python
async with AsyncSessionLocal() as session:
    yield session
    await session.commit()
```

### Pitfall 4: Alembic Not Detecting Models

**Problem:** Migration is empty

**Solution:** Import models in `models/__init__.py` before running Alembic.

### Pitfall 5: File Path Issues

**Problem:** Files not found or written to wrong location

**Solution:** Always use `Path` objects and `settings.upload_base_path`.

---

## 11. Next Steps

After completing R3:

1. **R4: Content Extraction** - Implement article content extraction
2. **R5: Search** - Add full-text search across articles
3. **R6: Embeddings** - Generate and store article embeddings

---

**Plan Status:** Ready for implementation
**Last Updated:** 2026-01-19
