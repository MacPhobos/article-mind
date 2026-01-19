# Plan R2: Session CRUD API

**Plan ID:** R2-session-crud-api
**Created:** 2026-01-19
**Dependencies:** Plan 02 (scaffold-service), Plan 03 (database-setup), Plan 04 (health-api)
**Estimated Time:** 3-4 hours

---

## 1. Overview

### Purpose

Implement the RESTful API endpoints for managing research sessions in the Article Mind backend service, including full CRUD operations, status transitions, and soft delete support.

### Scope

- Database schema for `research_sessions` table
- SQLAlchemy 2.0 async model
- Pydantic schemas for request/response validation
- RESTful API endpoints for CRUD operations
- Soft delete support
- Status transitions: draft -> active -> completed -> archived
- Alembic migration for schema creation
- Comprehensive pytest tests

### Architecture Decisions Applied

- **Session Lifecycle:** Full lifecycle (draft -> active -> completed -> archived) with soft delete
- **Auth Model:** Single-user (no authentication required)
- **Database:** PostgreSQL with SQLAlchemy 2.0 async

### Dependencies

- **Plan 02:** Requires FastAPI scaffolding
- **Plan 03:** Requires PostgreSQL database setup
- **Plan 04:** Follows health check API patterns

### Outputs

- Working `/api/v1/sessions` endpoints
- SQLAlchemy model for `research_sessions`
- Pydantic schemas for all request/response types
- Alembic migration
- pytest tests with 90%+ coverage
- Updated OpenAPI spec for frontend type generation

---

## 2. Technology Stack

- **Framework:** FastAPI 0.115+
- **ORM:** SQLAlchemy 2.0+ (async)
- **Validation:** Pydantic 2.x
- **Database:** PostgreSQL 16.x
- **Migrations:** Alembic 1.13+
- **Testing:** pytest 8.x + httpx

---

## 3. Directory Structure

```
article-mind-service/
├── src/article_mind_service/
│   ├── models/
│   │   ├── __init__.py           # UPDATED: Export ResearchSession
│   │   └── session.py            # NEW: ResearchSession model
│   ├── schemas/
│   │   ├── __init__.py           # UPDATED: Export session schemas
│   │   └── session.py            # NEW: Session Pydantic schemas
│   ├── routers/
│   │   ├── __init__.py           # UPDATED: Export sessions router
│   │   └── sessions.py           # NEW: Session CRUD router
│   └── main.py                   # UPDATED: Include sessions router
├── alembic/
│   └── versions/
│       └── xxxx_add_research_sessions.py  # NEW: Migration
└── tests/
    ├── test_sessions.py          # NEW: Session API tests
    └── unit/
        └── test_session_model.py # NEW: Model unit tests
```

---

## 4. Database Schema

### 4.1 SQLAlchemy Model

**File:** `src/article_mind_service/models/session.py`

```python
"""Research session database model."""

from datetime import datetime
from typing import Literal

from sqlalchemy import DateTime, Enum, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from article_mind_service.database import Base


# Session status as a Python type
SessionStatus = Literal["draft", "active", "completed", "archived"]


class ResearchSession(Base):
    """Research session model for organizing article collections.

    Design Decisions:

    1. Status Enum: Using PostgreSQL ENUM type for data integrity.
       - Ensures only valid statuses can be stored
       - Better performance than string comparison
       - Clear documentation of allowed values

    2. Soft Delete: Using deleted_at timestamp instead of hard delete.
       - Allows recovery of accidentally deleted sessions
       - Maintains referential integrity with articles
       - Enables audit trail

    3. Timestamps: Using server_default=func.now() for database-level defaults.
       - Ensures consistent timestamps across application instances
       - Works correctly even if app server clock is wrong

    4. Article Count: Not stored, computed via relationship.
       - Avoids denormalization issues
       - Always accurate (no sync problems)
       - Performance acceptable for expected data volumes

    Status Lifecycle:
        draft -> active -> completed -> archived
                  |          |
                  v          v
               archived   archived
    """

    __tablename__ = "research_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Session display name",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Optional session description",
    )
    status: Mapped[str] = mapped_column(
        Enum(
            "draft",
            "active",
            "completed",
            "archived",
            name="session_status",
            create_constraint=True,
        ),
        nullable=False,
        default="draft",
        server_default="draft",
        index=True,
        comment="Session lifecycle status",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="When session was created",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="When session was last updated",
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        index=True,
        comment="Soft delete timestamp (null = not deleted)",
    )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"<ResearchSession(id={self.id}, name='{self.name}', status='{self.status}')>"

    @property
    def is_deleted(self) -> bool:
        """Check if session is soft-deleted."""
        return self.deleted_at is not None

    def can_transition_to(self, new_status: str) -> bool:
        """Check if status transition is valid.

        Valid transitions:
        - draft -> active, archived
        - active -> completed, archived
        - completed -> archived
        - archived -> (none)

        Args:
            new_status: The target status to transition to

        Returns:
            True if transition is valid, False otherwise
        """
        valid_transitions: dict[str, set[str]] = {
            "draft": {"active", "archived"},
            "active": {"completed", "archived"},
            "completed": {"archived"},
            "archived": set(),  # No transitions from archived
        }

        return new_status in valid_transitions.get(self.status, set())
```

### 4.2 Update Models __init__.py

**File:** `src/article_mind_service/models/__init__.py`

```python
"""SQLAlchemy database models.

All models must be imported here for Alembic autogenerate to detect them.
"""

from .session import ResearchSession, SessionStatus

__all__ = [
    "ResearchSession",
    "SessionStatus",
]
```

---

## 5. Pydantic Schemas

**File:** `src/article_mind_service/schemas/session.py`

```python
"""Session request/response schemas for API contract."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# Session status type for type safety
SessionStatus = Literal["draft", "active", "completed", "archived"]


class CreateSessionRequest(BaseModel):
    """Request schema for creating a new session.

    All new sessions start in 'draft' status.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Session display name",
        examples=["My Research Project"],
    )
    description: str | None = Field(
        default=None,
        max_length=5000,
        description="Optional session description",
        examples=["Research on machine learning algorithms"],
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Strip whitespace from name."""
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        """Strip whitespace from description."""
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None


class UpdateSessionRequest(BaseModel):
    """Request schema for updating a session.

    All fields are optional - only provided fields are updated.
    """

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="New session name",
        examples=["Updated Project Name"],
    )
    description: str | None = Field(
        default=None,
        max_length=5000,
        description="New session description (empty string to clear)",
        examples=["Updated description"],
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        """Strip whitespace from name."""
        if v is None:
            return None
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        """Strip whitespace from description."""
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None


class ChangeStatusRequest(BaseModel):
    """Request schema for changing session status.

    Valid transitions:
    - draft -> active, archived
    - active -> completed, archived
    - completed -> archived
    - archived -> (none - cannot be changed)
    """

    status: SessionStatus = Field(
        ...,
        description="New status for the session",
        examples=["active"],
    )


class SessionResponse(BaseModel):
    """Response schema for a single session.

    This is the canonical session representation in the API.
    Used for all endpoints that return session data.
    """

    id: int = Field(
        ...,
        description="Unique session identifier",
        examples=[1],
    )
    name: str = Field(
        ...,
        description="Session display name",
        examples=["My Research Project"],
    )
    description: str | None = Field(
        default=None,
        description="Session description",
        examples=["Research on machine learning algorithms"],
    )
    status: SessionStatus = Field(
        ...,
        description="Current session status",
        examples=["active"],
    )
    article_count: int = Field(
        default=0,
        ge=0,
        description="Number of articles in this session",
        examples=[12],
    )
    created_at: datetime = Field(
        ...,
        description="When the session was created",
        examples=["2026-01-15T10:30:00Z"],
    )
    updated_at: datetime = Field(
        ...,
        description="When the session was last updated",
        examples=["2026-01-19T14:45:00Z"],
    )

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "name": "Machine Learning Research",
                    "description": "Collecting papers on deep learning",
                    "status": "active",
                    "article_count": 15,
                    "created_at": "2026-01-15T10:30:00Z",
                    "updated_at": "2026-01-19T14:45:00Z",
                }
            ]
        },
    }


class SessionListResponse(BaseModel):
    """Response schema for listing sessions.

    Includes pagination metadata for future expansion.
    """

    sessions: list[SessionResponse] = Field(
        ...,
        description="List of sessions",
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of sessions (excluding deleted)",
        examples=[25],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sessions": [
                        {
                            "id": 1,
                            "name": "Research Project A",
                            "description": "First project",
                            "status": "active",
                            "article_count": 10,
                            "created_at": "2026-01-15T10:30:00Z",
                            "updated_at": "2026-01-19T14:45:00Z",
                        }
                    ],
                    "total": 1,
                }
            ]
        },
    }
```

### 5.1 Update Schemas __init__.py

**File:** `src/article_mind_service/schemas/__init__.py`

```python
"""Pydantic schemas for API request/response models."""

from .health import HealthResponse
from .session import (
    ChangeStatusRequest,
    CreateSessionRequest,
    SessionListResponse,
    SessionResponse,
    SessionStatus,
    UpdateSessionRequest,
)

__all__ = [
    # Health
    "HealthResponse",
    # Session
    "ChangeStatusRequest",
    "CreateSessionRequest",
    "SessionListResponse",
    "SessionResponse",
    "SessionStatus",
    "UpdateSessionRequest",
]
```

---

## 6. API Endpoints

**File:** `src/article_mind_service/routers/sessions.py`

```python
"""Session CRUD API endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.database import get_db
from article_mind_service.models.session import ResearchSession
from article_mind_service.schemas.session import (
    ChangeStatusRequest,
    CreateSessionRequest,
    SessionListResponse,
    SessionResponse,
    SessionStatus,
    UpdateSessionRequest,
)

router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["sessions"],
)


def session_to_response(session: ResearchSession) -> SessionResponse:
    """Convert SQLAlchemy model to Pydantic response.

    Args:
        session: Database model instance

    Returns:
        Pydantic response schema

    Note:
        article_count is set to 0 for now. When articles are implemented,
        this should compute the actual count from the relationship.
    """
    return SessionResponse(
        id=session.id,
        name=session.name,
        description=session.description,
        status=session.status,  # type: ignore[arg-type]
        article_count=0,  # TODO: Compute from articles relationship
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


async def get_session_or_404(
    session_id: int,
    db: AsyncSession,
) -> ResearchSession:
    """Fetch a session by ID or raise 404.

    Args:
        session_id: Session ID to fetch
        db: Database session

    Returns:
        ResearchSession model

    Raises:
        HTTPException: 404 if session not found or deleted
    """
    result = await db.execute(
        select(ResearchSession).where(
            ResearchSession.id == session_id,
            ResearchSession.deleted_at.is_(None),  # Exclude soft-deleted
        )
    )
    session = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id {session_id} not found",
        )

    return session


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new session",
    description="Create a new research session. Sessions start in 'draft' status.",
    responses={
        201: {"description": "Session created successfully"},
        400: {"description": "Validation error"},
    },
)
async def create_session(
    data: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Create a new research session.

    Args:
        data: Session creation data (name, description)
        db: Database session

    Returns:
        Created session data
    """
    session = ResearchSession(
        name=data.name,
        description=data.description,
        status="draft",
    )

    db.add(session)
    await db.flush()  # Get the ID without committing
    await db.refresh(session)  # Refresh to get server defaults

    return session_to_response(session)


@router.get(
    "",
    response_model=SessionListResponse,
    summary="List all sessions",
    description="Get a list of all non-deleted research sessions with optional filtering.",
    responses={
        200: {"description": "List of sessions"},
    },
)
async def list_sessions(
    status_filter: SessionStatus | None = Query(
        default=None,
        alias="status",
        description="Filter by session status",
    ),
    db: AsyncSession = Depends(get_db),
) -> SessionListResponse:
    """List all research sessions.

    Args:
        status_filter: Optional status to filter by
        db: Database session

    Returns:
        List of sessions with total count
    """
    # Base query - exclude soft-deleted sessions
    query = select(ResearchSession).where(
        ResearchSession.deleted_at.is_(None)
    )

    # Apply status filter if provided
    if status_filter:
        query = query.where(ResearchSession.status == status_filter)

    # Order by updated_at descending (most recent first)
    query = query.order_by(ResearchSession.updated_at.desc())

    # Execute query
    result = await db.execute(query)
    sessions = result.scalars().all()

    # Get total count (for pagination metadata)
    count_query = select(func.count(ResearchSession.id)).where(
        ResearchSession.deleted_at.is_(None)
    )
    if status_filter:
        count_query = count_query.where(ResearchSession.status == status_filter)

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    return SessionListResponse(
        sessions=[session_to_response(s) for s in sessions],
        total=total,
    )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get a session by ID",
    description="Retrieve a single research session by its unique identifier.",
    responses={
        200: {"description": "Session found"},
        404: {"description": "Session not found"},
    },
)
async def get_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Get a single research session.

    Args:
        session_id: Unique session identifier
        db: Database session

    Returns:
        Session data

    Raises:
        HTTPException: 404 if not found
    """
    session = await get_session_or_404(session_id, db)
    return session_to_response(session)


@router.patch(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Update a session",
    description="Update session name and/or description. Only provided fields are updated.",
    responses={
        200: {"description": "Session updated"},
        404: {"description": "Session not found"},
        400: {"description": "Validation error"},
    },
)
async def update_session(
    session_id: int,
    data: UpdateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Update a research session.

    Args:
        session_id: Session to update
        data: Fields to update (name and/or description)
        db: Database session

    Returns:
        Updated session data

    Raises:
        HTTPException: 404 if not found
    """
    session = await get_session_or_404(session_id, db)

    # Update only provided fields
    if data.name is not None:
        session.name = data.name

    if data.description is not None:
        # Empty string clears the description
        session.description = data.description if data.description else None

    await db.flush()
    await db.refresh(session)

    return session_to_response(session)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session (soft delete)",
    description="Soft delete a session. The session can be recovered by an administrator.",
    responses={
        204: {"description": "Session deleted"},
        404: {"description": "Session not found"},
    },
)
async def delete_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Soft delete a research session.

    Args:
        session_id: Session to delete
        db: Database session

    Raises:
        HTTPException: 404 if not found
    """
    session = await get_session_or_404(session_id, db)

    # Soft delete by setting deleted_at
    session.deleted_at = datetime.now(timezone.utc)

    await db.flush()


@router.post(
    "/{session_id}/status",
    response_model=SessionResponse,
    summary="Change session status",
    description=(
        "Transition session to a new status. "
        "Valid transitions: draft->active, draft->archived, "
        "active->completed, active->archived, completed->archived."
    ),
    responses={
        200: {"description": "Status changed"},
        400: {"description": "Invalid status transition"},
        404: {"description": "Session not found"},
    },
)
async def change_session_status(
    session_id: int,
    data: ChangeStatusRequest,
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Change the status of a research session.

    Args:
        session_id: Session to update
        data: New status
        db: Database session

    Returns:
        Updated session data

    Raises:
        HTTPException: 404 if not found, 400 if invalid transition
    """
    session = await get_session_or_404(session_id, db)

    # Validate transition
    if not session.can_transition_to(data.status):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot transition from '{session.status}' to '{data.status}'",
        )

    session.status = data.status

    await db.flush()
    await db.refresh(session)

    return session_to_response(session)
```

### 6.1 Update Routers __init__.py

**File:** `src/article_mind_service/routers/__init__.py`

```python
"""FastAPI routers."""

from .health import router as health_router
from .sessions import router as sessions_router

__all__ = [
    "health_router",
    "sessions_router",
]
```

### 6.2 Update main.py

**File:** `src/article_mind_service/main.py` (additions only)

```python
# Add to imports:
from .routers import health_router, sessions_router

# Add after health_router include:
# Sessions CRUD API
app.include_router(sessions_router)
```

---

## 7. Alembic Migration

**File:** `alembic/versions/xxxx_add_research_sessions.py`

Create migration using:

```bash
make migrate-create MSG="add research sessions table"
```

Expected migration content:

```python
"""add research sessions table

Revision ID: <auto-generated>
Revises: 608494810fe8
Create Date: 2026-01-19 XX:XX:XX

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "<auto-generated>"
down_revision: str | Sequence[str] | None = "608494810fe8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create research_sessions table."""
    # Create session_status enum type
    session_status = sa.Enum(
        "draft",
        "active",
        "completed",
        "archived",
        name="session_status",
    )
    session_status.create(op.get_bind(), checkfirst=True)

    # Create research_sessions table
    op.create_table(
        "research_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "name",
            sa.String(length=255),
            nullable=False,
            comment="Session display name",
        ),
        sa.Column(
            "description",
            sa.Text(),
            nullable=True,
            comment="Optional session description",
        ),
        sa.Column(
            "status",
            session_status,
            nullable=False,
            server_default="draft",
            comment="Session lifecycle status",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            comment="When session was created",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            comment="When session was last updated",
        ),
        sa.Column(
            "deleted_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Soft delete timestamp (null = not deleted)",
        ),
    )

    # Create indexes
    op.create_index("ix_research_sessions_name", "research_sessions", ["name"])
    op.create_index("ix_research_sessions_status", "research_sessions", ["status"])
    op.create_index("ix_research_sessions_deleted_at", "research_sessions", ["deleted_at"])


def downgrade() -> None:
    """Drop research_sessions table."""
    # Drop indexes
    op.drop_index("ix_research_sessions_deleted_at", table_name="research_sessions")
    op.drop_index("ix_research_sessions_status", table_name="research_sessions")
    op.drop_index("ix_research_sessions_name", table_name="research_sessions")

    # Drop table
    op.drop_table("research_sessions")

    # Drop enum type
    sa.Enum(name="session_status").drop(op.get_bind(), checkfirst=True)
```

---

## 8. Implementation Steps

### Phase 1: Database Layer (1-1.5 hours)

1. **Create SQLAlchemy model**
   - File: `src/article_mind_service/models/session.py`
   - Include all fields and methods

2. **Update models/__init__.py**
   - Export ResearchSession and SessionStatus

3. **Create Alembic migration**
   ```bash
   cd article-mind-service
   make migrate-create MSG="add research sessions table"
   ```

4. **Review and apply migration**
   - Check generated migration for correctness
   - Apply: `make migrate`

5. **Verify table creation**
   ```bash
   docker exec -it article-mind-postgres psql -U article_mind -d article_mind -c "\d research_sessions"
   ```

### Phase 2: Schema Layer (30 minutes)

6. **Create Pydantic schemas**
   - File: `src/article_mind_service/schemas/session.py`
   - Create all request/response schemas

7. **Update schemas/__init__.py**
   - Export all session schemas

### Phase 3: API Layer (1-1.5 hours)

8. **Create sessions router**
   - File: `src/article_mind_service/routers/sessions.py`
   - Implement all endpoints

9. **Update routers/__init__.py**
   - Export sessions_router

10. **Update main.py**
    - Include sessions_router

11. **Start dev server and test manually**
    ```bash
    make dev
    # Test with curl or use Swagger UI at /docs
    ```

### Phase 4: Testing (1 hour)

12. **Write integration tests**
    - File: `tests/test_sessions.py`
    - Test all endpoints

13. **Write unit tests**
    - File: `tests/unit/test_session_model.py`
    - Test model methods

14. **Run all tests**
    ```bash
    make test
    ```

---

## 9. Testing

### 9.1 Integration Tests

**File:** `tests/test_sessions.py`

```python
"""Integration tests for session CRUD API."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestCreateSession:
    """Tests for POST /api/v1/sessions."""

    async def test_create_session_success(self, async_client: AsyncClient) -> None:
        """Test successful session creation."""
        response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Test Session", "description": "A test description"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Session"
        assert data["description"] == "A test description"
        assert data["status"] == "draft"
        assert data["article_count"] == 0
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    async def test_create_session_minimal(self, async_client: AsyncClient) -> None:
        """Test session creation with only required fields."""
        response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Minimal Session"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Minimal Session"
        assert data["description"] is None

    async def test_create_session_empty_name_fails(self, async_client: AsyncClient) -> None:
        """Test that empty name fails validation."""
        response = await async_client.post(
            "/api/v1/sessions",
            json={"name": ""},
        )

        assert response.status_code == 422  # Validation error

    async def test_create_session_whitespace_name_trimmed(
        self, async_client: AsyncClient
    ) -> None:
        """Test that whitespace is trimmed from name."""
        response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "  Trimmed Name  "},
        )

        assert response.status_code == 201
        assert response.json()["name"] == "Trimmed Name"


@pytest.mark.asyncio
class TestListSessions:
    """Tests for GET /api/v1/sessions."""

    async def test_list_sessions_empty(self, async_client: AsyncClient) -> None:
        """Test listing when no sessions exist."""
        response = await async_client.get("/api/v1/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data
        assert isinstance(data["sessions"], list)

    async def test_list_sessions_with_data(self, async_client: AsyncClient) -> None:
        """Test listing with existing sessions."""
        # Create a session first
        await async_client.post(
            "/api/v1/sessions",
            json={"name": "Session 1"},
        )

        response = await async_client.get("/api/v1/sessions")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert any(s["name"] == "Session 1" for s in data["sessions"])

    async def test_list_sessions_filter_by_status(
        self, async_client: AsyncClient
    ) -> None:
        """Test filtering by status."""
        # Create sessions with different statuses
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Draft Session"},
        )
        session_id = create_response.json()["id"]

        # Change status to active
        await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "active"},
        )

        # Filter by active status
        response = await async_client.get("/api/v1/sessions?status=active")

        assert response.status_code == 200
        data = response.json()
        assert all(s["status"] == "active" for s in data["sessions"])


@pytest.mark.asyncio
class TestGetSession:
    """Tests for GET /api/v1/sessions/{id}."""

    async def test_get_session_success(self, async_client: AsyncClient) -> None:
        """Test getting an existing session."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Get Test", "description": "Test description"},
        )
        session_id = create_response.json()["id"]

        # Get the session
        response = await async_client.get(f"/api/v1/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id
        assert data["name"] == "Get Test"
        assert data["description"] == "Test description"

    async def test_get_session_not_found(self, async_client: AsyncClient) -> None:
        """Test getting a non-existent session."""
        response = await async_client.get("/api/v1/sessions/99999")

        assert response.status_code == 404


@pytest.mark.asyncio
class TestUpdateSession:
    """Tests for PATCH /api/v1/sessions/{id}."""

    async def test_update_session_name(self, async_client: AsyncClient) -> None:
        """Test updating session name."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Original Name"},
        )
        session_id = create_response.json()["id"]

        # Update the name
        response = await async_client.patch(
            f"/api/v1/sessions/{session_id}",
            json={"name": "Updated Name"},
        )

        assert response.status_code == 200
        assert response.json()["name"] == "Updated Name"

    async def test_update_session_description(self, async_client: AsyncClient) -> None:
        """Test updating session description."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Test", "description": "Original"},
        )
        session_id = create_response.json()["id"]

        # Update the description
        response = await async_client.patch(
            f"/api/v1/sessions/{session_id}",
            json={"description": "Updated description"},
        )

        assert response.status_code == 200
        assert response.json()["description"] == "Updated description"

    async def test_update_session_clear_description(
        self, async_client: AsyncClient
    ) -> None:
        """Test clearing session description with empty string."""
        # Create a session with description
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Test", "description": "Has description"},
        )
        session_id = create_response.json()["id"]

        # Clear the description
        response = await async_client.patch(
            f"/api/v1/sessions/{session_id}",
            json={"description": ""},
        )

        assert response.status_code == 200
        assert response.json()["description"] is None

    async def test_update_session_not_found(self, async_client: AsyncClient) -> None:
        """Test updating non-existent session."""
        response = await async_client.patch(
            "/api/v1/sessions/99999",
            json={"name": "Updated"},
        )

        assert response.status_code == 404


@pytest.mark.asyncio
class TestDeleteSession:
    """Tests for DELETE /api/v1/sessions/{id}."""

    async def test_delete_session_success(self, async_client: AsyncClient) -> None:
        """Test soft deleting a session."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "To Delete"},
        )
        session_id = create_response.json()["id"]

        # Delete the session
        response = await async_client.delete(f"/api/v1/sessions/{session_id}")

        assert response.status_code == 204

        # Verify it's not accessible
        get_response = await async_client.get(f"/api/v1/sessions/{session_id}")
        assert get_response.status_code == 404

    async def test_delete_session_not_found(self, async_client: AsyncClient) -> None:
        """Test deleting non-existent session."""
        response = await async_client.delete("/api/v1/sessions/99999")

        assert response.status_code == 404


@pytest.mark.asyncio
class TestChangeSessionStatus:
    """Tests for POST /api/v1/sessions/{id}/status."""

    async def test_change_status_draft_to_active(
        self, async_client: AsyncClient
    ) -> None:
        """Test transitioning from draft to active."""
        # Create a session (starts as draft)
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Status Test"},
        )
        session_id = create_response.json()["id"]
        assert create_response.json()["status"] == "draft"

        # Change to active
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "active"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "active"

    async def test_change_status_active_to_completed(
        self, async_client: AsyncClient
    ) -> None:
        """Test transitioning from active to completed."""
        # Create and activate session
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Status Test"},
        )
        session_id = create_response.json()["id"]

        await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "active"},
        )

        # Change to completed
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "completed"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "completed"

    async def test_change_status_to_archived(self, async_client: AsyncClient) -> None:
        """Test transitioning to archived from any non-archived status."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Archive Test"},
        )
        session_id = create_response.json()["id"]

        # Archive directly from draft
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "archived"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "archived"

    async def test_change_status_invalid_transition(
        self, async_client: AsyncClient
    ) -> None:
        """Test invalid status transition is rejected."""
        # Create a session (draft)
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Invalid Test"},
        )
        session_id = create_response.json()["id"]

        # Try to go directly to completed (invalid: draft -> completed)
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "completed"},
        )

        assert response.status_code == 400
        assert "Cannot transition" in response.json()["detail"]

    async def test_change_status_from_archived_fails(
        self, async_client: AsyncClient
    ) -> None:
        """Test that archived sessions cannot change status."""
        # Create and archive a session
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Archived Test"},
        )
        session_id = create_response.json()["id"]

        await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "archived"},
        )

        # Try to change from archived
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "active"},
        )

        assert response.status_code == 400

    async def test_change_status_not_found(self, async_client: AsyncClient) -> None:
        """Test changing status of non-existent session."""
        response = await async_client.post(
            "/api/v1/sessions/99999/status",
            json={"status": "active"},
        )

        assert response.status_code == 404


@pytest.mark.asyncio
class TestSessionOpenAPI:
    """Tests for OpenAPI documentation."""

    async def test_sessions_in_openapi(self, async_client: AsyncClient) -> None:
        """Test that session endpoints are in OpenAPI spec."""
        response = await async_client.get("/openapi.json")

        assert response.status_code == 200
        openapi = response.json()

        # Verify endpoints exist
        assert "/api/v1/sessions" in openapi["paths"]
        assert "/api/v1/sessions/{session_id}" in openapi["paths"]
        assert "/api/v1/sessions/{session_id}/status" in openapi["paths"]

    async def test_session_schemas_in_openapi(self, async_client: AsyncClient) -> None:
        """Test that session schemas are in OpenAPI spec."""
        response = await async_client.get("/openapi.json")

        assert response.status_code == 200
        openapi = response.json()

        # Verify schemas exist
        schemas = openapi["components"]["schemas"]
        assert "SessionResponse" in schemas
        assert "SessionListResponse" in schemas
        assert "CreateSessionRequest" in schemas
        assert "UpdateSessionRequest" in schemas
        assert "ChangeStatusRequest" in schemas
```

### 9.2 Unit Tests

**File:** `tests/unit/test_session_model.py`

```python
"""Unit tests for ResearchSession model."""

import pytest

from article_mind_service.models.session import ResearchSession


class TestResearchSessionModel:
    """Tests for ResearchSession model methods."""

    def test_repr(self) -> None:
        """Test string representation."""
        session = ResearchSession(id=1, name="Test", status="draft")
        repr_str = repr(session)

        assert "ResearchSession" in repr_str
        assert "id=1" in repr_str
        assert "name='Test'" in repr_str
        assert "status='draft'" in repr_str

    def test_is_deleted_false(self) -> None:
        """Test is_deleted returns False when not deleted."""
        session = ResearchSession(id=1, name="Test", status="draft", deleted_at=None)

        assert session.is_deleted is False

    def test_is_deleted_true(self) -> None:
        """Test is_deleted returns True when deleted."""
        from datetime import datetime, timezone

        session = ResearchSession(
            id=1,
            name="Test",
            status="draft",
            deleted_at=datetime.now(timezone.utc),
        )

        assert session.is_deleted is True


class TestStatusTransitions:
    """Tests for status transition validation."""

    @pytest.mark.parametrize(
        "current_status,new_status,expected",
        [
            # Valid transitions from draft
            ("draft", "active", True),
            ("draft", "archived", True),
            ("draft", "completed", False),  # Invalid: must go through active
            ("draft", "draft", False),  # Same status
            # Valid transitions from active
            ("active", "completed", True),
            ("active", "archived", True),
            ("active", "draft", False),  # Can't go back
            ("active", "active", False),  # Same status
            # Valid transitions from completed
            ("completed", "archived", True),
            ("completed", "draft", False),  # Can't go back
            ("completed", "active", False),  # Can't go back
            ("completed", "completed", False),  # Same status
            # No transitions from archived
            ("archived", "draft", False),
            ("archived", "active", False),
            ("archived", "completed", False),
            ("archived", "archived", False),
        ],
    )
    def test_can_transition_to(
        self, current_status: str, new_status: str, expected: bool
    ) -> None:
        """Test status transition validation."""
        session = ResearchSession(id=1, name="Test", status=current_status)

        assert session.can_transition_to(new_status) == expected
```

---

## 10. Verification Steps

### 1. Run Tests

```bash
cd article-mind-service
make test
```

**Expected:** All tests pass, including new session tests.

### 2. Start Dev Server

```bash
make dev
```

**Expected:** Server starts without errors.

### 3. Test Create Session

```bash
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"name": "My Research", "description": "Test description"}'
```

**Expected Response (201 Created):**
```json
{
  "id": 1,
  "name": "My Research",
  "description": "Test description",
  "status": "draft",
  "article_count": 0,
  "created_at": "2026-01-19T...",
  "updated_at": "2026-01-19T..."
}
```

### 4. Test List Sessions

```bash
curl http://localhost:8000/api/v1/sessions
```

**Expected Response (200 OK):**
```json
{
  "sessions": [...],
  "total": 1
}
```

### 5. Test Status Change

```bash
curl -X POST http://localhost:8000/api/v1/sessions/1/status \
  -H "Content-Type: application/json" \
  -d '{"status": "active"}'
```

**Expected Response (200 OK):**
```json
{
  "id": 1,
  "name": "My Research",
  "status": "active",
  ...
}
```

### 6. Test Invalid Status Transition

```bash
# First, archive the session
curl -X POST http://localhost:8000/api/v1/sessions/1/status \
  -H "Content-Type: application/json" \
  -d '{"status": "archived"}'

# Then try to change from archived
curl -X POST http://localhost:8000/api/v1/sessions/1/status \
  -H "Content-Type: application/json" \
  -d '{"status": "active"}'
```

**Expected Response (400 Bad Request):**
```json
{
  "detail": "Cannot transition from 'archived' to 'active'"
}
```

### 7. Verify OpenAPI Documentation

Visit: http://localhost:8000/docs

**Expected:**
- `/api/v1/sessions` endpoints listed
- Schemas for SessionResponse, CreateSessionRequest, etc.
- Try it out works for all endpoints

### 8. Generate Frontend Types

```bash
cd article-mind-ui
make gen-api
```

**Expected:** `src/lib/api/generated.ts` contains session types.

---

## 11. Acceptance Criteria

### Functional Requirements

- [ ] **AC1:** POST /api/v1/sessions creates a new session in draft status
- [ ] **AC2:** GET /api/v1/sessions returns list of non-deleted sessions
- [ ] **AC3:** GET /api/v1/sessions?status=X filters by status
- [ ] **AC4:** GET /api/v1/sessions/{id} returns single session
- [ ] **AC5:** PATCH /api/v1/sessions/{id} updates name and/or description
- [ ] **AC6:** DELETE /api/v1/sessions/{id} soft-deletes (sets deleted_at)
- [ ] **AC7:** POST /api/v1/sessions/{id}/status changes status
- [ ] **AC8:** Invalid status transitions return 400 error
- [ ] **AC9:** Non-existent session IDs return 404 error
- [ ] **AC10:** Soft-deleted sessions are not returned in list/get

### Non-Functional Requirements

- [ ] **AC11:** All endpoints use Pydantic response_model
- [ ] **AC12:** OpenAPI spec includes all session schemas
- [ ] **AC13:** Database migration creates table with indexes
- [ ] **AC14:** All tests pass (90%+ coverage for session code)
- [ ] **AC15:** Code passes mypy type checking
- [ ] **AC16:** Code passes ruff linting
- [ ] **AC17:** Frontend type generation succeeds

---

## 12. Common Pitfalls

### Pitfall 1: Alembic Not Detecting Model

**Symptom:** `make migrate-create` creates empty migration

**Solution:** Ensure model is imported in `models/__init__.py`:
```python
from .session import ResearchSession
```

### Pitfall 2: Enum Type Already Exists

**Symptom:** Migration fails with "type already exists"

**Solution:** Add `checkfirst=True` to enum creation:
```python
session_status.create(op.get_bind(), checkfirst=True)
```

### Pitfall 3: Async Session Not Committed

**Symptom:** Data not persisted after endpoint returns

**Solution:** Use `db.flush()` and ensure `get_db` commits on success (already in database.py).

### Pitfall 4: Missing Timezone on Datetime

**Symptom:** Datetime comparisons fail or behave unexpectedly

**Solution:** Always use `datetime.now(timezone.utc)` instead of `datetime.utcnow()`.

### Pitfall 5: Test Database State Pollution

**Symptom:** Tests pass individually but fail when run together

**Solution:** Use test database with transaction rollback per test (configure in conftest.py).

---

## 13. Next Steps

After completing this plan:

1. **Implement Plan R1** (Session Management UI) to build frontend
2. **Run frontend type generation** with `make gen-api`
3. **Test full integration** between frontend and backend
4. **Add article management** endpoints (future plan)
5. **Add pagination** to list endpoint (future enhancement)

---

**Plan Status:** Ready for implementation
**Last Updated:** 2026-01-19
