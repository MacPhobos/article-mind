# Plan 04: Service Health Check API

**Plan ID:** 04-service-health-api
**Created:** 2026-01-18
**Dependencies:** Plan 02 (scaffold-service), Plan 03 (database-setup)
**Estimated Time:** 1-2 hours

---

## 1. Overview

### Purpose
Implement a production-ready health check endpoint following the API contract specification, including database connectivity verification and proper OpenAPI documentation.

### Scope
- Create Pydantic schema for health check response
- Implement `/health` endpoint (no API prefix)
- Add database connectivity check
- Write comprehensive tests
- Update API contract documentation
- Ensure type-safe OpenAPI generation

### Dependencies
- **Plan 02:** Requires FastAPI scaffolding
- **Plan 03:** Requires database connection for health check

### Outputs
- Working `/health` endpoint at http://localhost:8000/health
- Pydantic schema in `schemas/health.py`
- Unit and integration tests
- OpenAPI spec includes health endpoint
- API contract documentation

---

## 2. Technology Stack

Same as Plan 02:
- **FastAPI:** 0.115+ (route handling, OpenAPI)
- **Pydantic:** 2.x (schema validation)
- **SQLAlchemy:** 2.0+ (database health check)
- **pytest:** 8.x (testing)

---

## 3. Directory Structure

```
article-mind-service/
└── src/article_mind_service/
    ├── schemas/
    │   ├── __init__.py
    │   └── health.py          # NEW: Health check schema
    ├── routers/
    │   ├── __init__.py
    │   └── health.py          # NEW: Health check router
    └── main.py                # UPDATED: Include health router
```

---

## 4. Implementation Steps

### Step 1: Create Health Check Schema

Create `src/article_mind_service/schemas/health.py`:

```python
"""Health check response schemas."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response following API contract specification.

    See: docs/auxiliary_info/initial_api_contract_instructions.md
    """

    status: str = Field(
        ...,
        description="Service health status",
        examples=["ok", "degraded", "error"],
    )
    version: str = Field(
        ...,
        description="API version",
        examples=["1.0.0"],
    )
    database: str | None = Field(
        default=None,
        description="Database connection status",
        examples=["connected", "disconnected"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ok",
                    "version": "1.0.0",
                    "database": "connected",
                }
            ]
        }
    }
```

Update `src/article_mind_service/schemas/__init__.py`:

```python
"""Pydantic schemas for API request/response models."""

from .health import HealthResponse

__all__ = [
    "HealthResponse",
]
```

### Step 2: Create Health Check Router

Create `src/article_mind_service/routers/health.py`:

```python
"""Health check endpoints."""

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.database import get_db
from article_mind_service.schemas.health import HealthResponse
from article_mind_service.config import settings

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="Returns service health status including database connectivity. No authentication required.",
)
async def health_check(
    db: AsyncSession = Depends(get_db),
) -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse with status, version, and database connectivity.

    Note:
        This endpoint does NOT use the /api/v1 prefix as specified in the API contract.
    """
    # Check database connectivity
    database_status = "disconnected"
    try:
        await db.execute(text("SELECT 1"))
        database_status = "connected"
    except Exception as e:
        print(f"Database health check failed: {e}")
        # Don't raise - return degraded status instead

    # Determine overall status
    overall_status = "ok" if database_status == "connected" else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        database=database_status,
    )
```

Update `src/article_mind_service/routers/__init__.py`:

```python
"""FastAPI routers."""

from .health import router as health_router

__all__ = [
    "health_router",
]
```

### Step 3: Register Health Router in Main App

Update `src/article_mind_service/main.py`:

```python
"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .config import settings
from .database import engine
from .routers import health_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager."""
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")

    # Test database connection
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        print("✅ Database connection verified")
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        print("⚠️  Service will start but /health will report degraded status")

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


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Article Mind Service API"}
```

### Step 4: Write Tests

Update `tests/test_main.py`:

```python
"""Tests for main application and health check."""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import text

from article_mind_service.main import app
from article_mind_service.database import get_db


def test_root(client: TestClient) -> None:
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Article Mind Service API"}


@pytest.mark.asyncio
async def test_root_async(async_client: AsyncClient) -> None:
    """Test root endpoint with async client."""
    response = await async_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Article Mind Service API"}


def test_openapi_docs(client: TestClient) -> None:
    """Test OpenAPI documentation is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200
    openapi = response.json()
    assert openapi["info"]["title"] == "Article Mind Service"


def test_health_check_endpoint_exists(client: TestClient) -> None:
    """Test health check endpoint is accessible."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_check_response_structure(client: TestClient) -> None:
    """Test health check response follows schema."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "version" in data
    assert data["status"] in ["ok", "degraded", "error"]
    assert isinstance(data["version"], str)


@pytest.mark.asyncio
async def test_health_check_with_database(async_client: AsyncClient) -> None:
    """Test health check includes database status."""
    response = await async_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "database" in data
    # Database status depends on if DB is running
    # Don't assert specific value, just that it exists


@pytest.mark.asyncio
async def test_health_check_database_connected(async_client: AsyncClient) -> None:
    """Test health check reports database as connected when DB is healthy."""
    response = await async_client.get("/health")
    data = response.json()

    # If this test runs with database available:
    if data["database"] == "connected":
        assert data["status"] == "ok"


def test_health_check_in_openapi(client: TestClient) -> None:
    """Test health check endpoint is documented in OpenAPI spec."""
    response = client.get("/openapi.json")
    openapi = response.json()

    # Verify /health endpoint exists in OpenAPI spec
    assert "/health" in openapi["paths"]

    health_spec = openapi["paths"]["/health"]
    assert "get" in health_spec

    # Verify response schema
    get_spec = health_spec["get"]
    assert "200" in get_spec["responses"]
    assert "application/json" in get_spec["responses"]["200"]["content"]


@pytest.mark.asyncio
async def test_health_check_no_authentication_required(async_client: AsyncClient) -> None:
    """Test health check does not require authentication."""
    # No auth headers provided
    response = await async_client.get("/health")
    assert response.status_code == 200  # Should succeed without auth
```

Create `tests/test_health.py` for specific health check tests:

```python
"""Detailed tests for health check endpoint."""

import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock


@pytest.mark.asyncio
async def test_health_check_ok_status(async_client: AsyncClient) -> None:
    """Test health check returns 'ok' when database is healthy."""
    response = await async_client.get("/health")
    data = response.json()

    assert response.status_code == 200
    assert data["version"] == "0.1.0"

    # If DB is available, should be ok
    if data["database"] == "connected":
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_health_check_degraded_on_db_failure(async_client: AsyncClient) -> None:
    """Test health check returns 'degraded' when database is down."""
    # Mock database failure
    with patch("article_mind_service.routers.health.get_db") as mock_get_db:
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("Database connection failed")

        async def mock_db_generator():
            yield mock_session

        mock_get_db.return_value = mock_db_generator()

        response = await async_client.get("/health")
        data = response.json()

        assert response.status_code == 200  # Still returns 200
        assert data["status"] == "degraded"
        assert data["database"] == "disconnected"


@pytest.mark.asyncio
async def test_health_check_response_schema_validation(async_client: AsyncClient) -> None:
    """Test health check response validates against Pydantic schema."""
    response = await async_client.get("/health")
    data = response.json()

    # Required fields
    assert "status" in data
    assert "version" in data

    # Optional fields
    assert "database" in data or data.get("database") is None

    # Type validation
    assert isinstance(data["status"], str)
    assert isinstance(data["version"], str)
    if data.get("database"):
        assert isinstance(data["database"], str)


@pytest.mark.asyncio
async def test_health_check_performance(async_client: AsyncClient) -> None:
    """Test health check responds quickly."""
    import time

    start = time.time()
    response = await async_client.get("/health")
    duration = time.time() - start

    assert response.status_code == 200
    assert duration < 1.0  # Should respond in less than 1 second
```

### Step 5: Create API Contract Documentation

Create `docs/api-contract.md` (both in service and UI):

```markdown
# Article Mind API Contract

> **Version**: 1.0.0
> **Last Updated**: 2026-01-18
> **Status**: FROZEN - Changes require version bump and UI sync

This document defines the API contract between `article-mind-service` (backend) and `article-mind-ui` (frontend).

---

## Base Configuration

| Environment | Base URL |
|-------------|----------|
| Development | `http://localhost:8000` |
| Production  | `https://api.article-mind.com` (TBD) |

All endpoints are prefixed with `/api/v1` except `/health` and `/openapi.json`.

---

## Type Generation Strategy

### Backend (Python FastAPI)

- OpenAPI spec auto-generated at `/openapi.json`
- All response models are Pydantic schemas
- Schemas located in `src/article_mind_service/schemas/`

### Frontend (SvelteKit TypeScript)

- Generated types at `src/lib/api/generated.ts`
- Generation script: `npm run gen:api`
- Command: `openapi-typescript http://localhost:8000/openapi.json -o src/lib/api/generated.ts`

**Workflow**: Backend changes models → Backend deploys → UI runs type generation → UI updates

---

## Common Types

### ErrorResponse

All errors follow this shape.

```typescript
interface ErrorResponse {
  error: {
    code: string;       // Machine-readable code
    message: string;    // Human-readable message
    details?: unknown;  // Optional additional context
  };
}
```

---

## Endpoints

### Health

#### `GET /health`

Health check endpoint. No authentication required. No `/api/v1` prefix.

**Response** `200 OK`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "database": "connected"
}
```

**Schema:**

```typescript
interface HealthResponse {
  status: "ok" | "degraded" | "error";
  version: string;
  database?: "connected" | "disconnected";
}
```

**Status Values:**
- `ok`: All systems operational
- `degraded`: Service running but database unavailable
- `error`: Critical failure

---

## Status Codes

| Code | Usage |
|------|-------|
| `200 OK` | Successful GET, PATCH, POST (actions) |
| `201 Created` | Successful POST (resource creation) |
| `204 No Content` | Successful DELETE |
| `400 Bad Request` | Validation error |
| `404 Not Found` | Resource not found |
| `500 Internal Server Error` | Unexpected error |

---

## CORS Configuration

### Development

```
Origins: http://localhost:5173
Credentials: Allowed
Methods: All
Headers: All
```

### Production

Configure via environment variable: `CORS_ORIGINS`

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-18 | Initial contract with /health endpoint |

---

_This contract is the source of truth. UI and service implementations must conform to these definitions._
```

Copy to both locations:

```bash
# In article-mind-service
cp docs/api-contract.md article-mind-service/docs/

# In article-mind-ui (create after Plan 01)
cp docs/api-contract.md article-mind-ui/docs/
```

### Step 6: Update Root CLAUDE.md

Update `/export/workspace/article-mind/CLAUDE.md` with API contract section:

```markdown
## The Golden Rule: API Contract is FROZEN

**Contract Location**: `docs/api-contract.md` exists in BOTH subprojects (must be identical)

### Contract Synchronization Workflow (ONE WAY ONLY)

1. **Update backend contract**: Edit `article-mind-service/docs/api-contract.md`
2. **Update backend models**: Update Pydantic schemas
3. **Run backend tests**: Verify nothing breaks
4. **Copy contract to frontend**: `cp article-mind-service/docs/api-contract.md article-mind-ui/docs/`
5. **Regenerate frontend types**: `cd article-mind-ui && npm run gen:api`
6. **Update frontend code**: Use new generated types
7. **Run frontend tests**: Verify type safety

**NEVER**:
- Manually edit generated type files in the frontend
- Change API without updating contract in BOTH repos
- Deploy frontend without regenerating types after backend changes
- Break backward compatibility without version bump

### Contract Change Checklist

- [ ] Update `article-mind-service/docs/api-contract.md`
- [ ] Update backend Pydantic schemas
- [ ] Backend tests pass
- [ ] Copy contract to `article-mind-ui/docs/api-contract.md`
- [ ] Regenerate types (`npm run gen:api`)
- [ ] Update frontend code using new types
- [ ] Frontend tests pass
```

---

## 5. CLAUDE.md Requirements

Updated in Plan 02's CLAUDE.md (article-mind-service):

- ✅ Health check endpoint implementation
- ✅ Pydantic schema as source of truth for OpenAPI
- ✅ API contract workflow
- ✅ Testing strategy for API endpoints

---

## 6. Verification Steps

### 1. Run Tests

```bash
cd article-mind-service
make test
```

**Expected:** All tests pass, including new health check tests.

### 2. Start Dev Server

```bash
make dev
```

**Expected:** Server starts with database health check in logs.

### 3. Test Health Endpoint

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "database": "connected"
}
```

### 4. Verify OpenAPI Documentation

Visit: http://localhost:8000/docs

**Expected:**
- `/health` endpoint listed
- Schema shows `HealthResponse` model
- Try it out works without authentication

### 5. Verify OpenAPI JSON

```bash
curl http://localhost:8000/openapi.json | jq '.paths["/health"]'
```

**Expected:** JSON schema for health endpoint.

### 6. Test Database Failure Scenario

Stop database:

```bash
docker stop article-mind-postgres
```

Check health endpoint:

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "degraded",
  "version": "0.1.0",
  "database": "disconnected"
}
```

Restart database:

```bash
docker start article-mind-postgres
```

### 7. Verify Type Generation (After Plan 01)

```bash
cd article-mind-ui
npm run gen:api
```

**Expected:** `src/lib/api/generated.ts` contains `HealthResponse` type.

---

## 7. Common Pitfalls

### Issue 1: Health Check Returns 500 Error

**Symptom:** `/health` returns 500 Internal Server Error

**Solution:**
- Check `get_db` dependency is working
- Verify database connection in logs
- Ensure `AsyncSession` is properly configured

### Issue 2: Database Status Always "disconnected"

**Symptom:** Health check always reports database as disconnected

**Solution:**
- Verify PostgreSQL is running: `docker ps`
- Check `.env` has correct `DATABASE_URL`
- Test connection with `scripts/test_db.py`
- Check for async/await errors

### Issue 3: Health Endpoint Not in OpenAPI Spec

**Symptom:** `/health` not visible in Swagger UI

**Solution:**
- Ensure router is included in main app: `app.include_router(health_router)`
- Verify `response_model=HealthResponse` is set
- Restart dev server (hot reload may not catch router changes)

### Issue 4: Frontend Type Generation Fails

**Symptom:** `npm run gen:api` errors

**Solution:**
- Ensure backend is running on http://localhost:8000
- Verify `/openapi.json` endpoint is accessible
- Check `openapi-typescript` is installed

### Issue 5: CORS Errors in Frontend

**Symptom:** Browser console shows CORS errors when calling `/health`

**Solution:**
- Verify CORS middleware is configured in `main.py`
- Check `settings.cors_origins` includes frontend URL
- Ensure `.env` has `CORS_ORIGINS=http://localhost:5173`

---

## 8. Next Steps

After completing this plan:

1. **Proceed to Plan 05** (ui-health-display) to implement frontend health indicator
2. **Generate TypeScript types** in UI project with `npm run gen:api`
3. **Add more API endpoints** following same pattern (Pydantic schema + router + tests)
4. **Maintain API contract** documentation as new endpoints are added

---

## 9. Success Criteria

- ✅ `/health` endpoint returns proper JSON response
- ✅ Health check includes database connectivity status
- ✅ All tests pass (8+ tests for health check)
- ✅ OpenAPI spec includes health endpoint with proper schema
- ✅ Health check responds in <1 second
- ✅ Health check works without authentication
- ✅ Service handles database failures gracefully (returns "degraded")
- ✅ API contract documentation complete
- ✅ Pydantic schema follows API contract specification
- ✅ Type-safe response model validated

---

**Plan Status:** Ready for implementation
**Last Updated:** 2026-01-18
