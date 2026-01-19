# Plan 02: FastAPI Service Scaffolding

**Plan ID:** 02-scaffold-service
**Created:** 2026-01-18
**Dependencies:** None (can run in parallel with Plan 01)
**Estimated Time:** 3-5 hours

---

## 1. Overview

### Purpose
Scaffold the article-mind-service backend application using FastAPI, Python 3.12, and modern Python tooling (uv, ruff, mypy, bandit) with a complete development and database infrastructure.

### Scope
- Initialize Python project with uv package manager
- Configure FastAPI with automatic OpenAPI generation
- Set up database tooling (Alembic for migrations)
- Configure code quality tools (ruff, mypy, bandit, pytest)
- Create Makefile for common commands
- Configure ASDF for version management
- Establish project-specific CLAUDE.md

### Dependencies
None - this is a foundational plan. However, Plan 03 (database-setup) should follow immediately after.

### Outputs
- Working FastAPI application skeleton
- Development server running on http://localhost:8000
- OpenAPI documentation at http://localhost:8000/docs
- Complete toolchain ready for feature development

---

## 2. Technology Stack

### Core Framework
- **Python:** 3.12.12 (via ASDF)
- **FastAPI:** 0.115+ (latest stable)
- **Uvicorn:** 0.34+ (ASGI server with hot reload)
- **Pydantic:** 2.x (data validation and OpenAPI generation)

### Package Management
- **uv:** 0.5+ (fast Python package manager, 10-100x faster than pip)

### Database
- **PostgreSQL:** 16.x (client library: psycopg[binary])
- **SQLAlchemy:** 2.0+ (async ORM)
- **Alembic:** 1.13+ (database migrations)

### Code Quality
- **ruff:** 0.8+ (fast linter and formatter, replaces black, flake8, isort)
- **mypy:** 1.14+ (static type checker)
- **bandit:** 1.7+ (security linter)

### Testing
- **pytest:** 8.x (test framework)
- **pytest-asyncio:** 0.24+ (async test support)
- **httpx:** 0.28+ (async HTTP client for testing)

---

## 3. Directory Structure

```
article-mind/
└── article-mind-service/
    ├── .tool-versions              # ASDF version pinning
    ├── Makefile                    # Common commands
    ├── CLAUDE.md                   # Backend-specific instructions
    ├── pyproject.toml              # Python project metadata and dependencies
    ├── .python-version             # Python version for uv
    ├── alembic.ini                 # Alembic configuration
    ├── .env.example                # Environment variable template
    ├── .env                        # Environment variables (gitignored)
    ├── src/
    │   └── article_mind_service/
    │       ├── __init__.py
    │       ├── main.py             # FastAPI application entry point
    │       ├── config.py           # Configuration management
    │       ├── database.py         # Database session management
    │       ├── models/             # SQLAlchemy models
    │       │   └── __init__.py
    │       ├── schemas/            # Pydantic schemas (API contracts)
    │       │   └── __init__.py
    │       ├── routers/            # FastAPI routers
    │       │   └── __init__.py
    │       └── dependencies.py     # FastAPI dependencies
    ├── alembic/                    # Database migrations
    │   ├── versions/               # Migration files
    │   ├── env.py                  # Alembic environment
    │   └── script.py.mako          # Migration template
    ├── tests/
    │   ├── __init__.py
    │   ├── conftest.py             # Pytest fixtures
    │   └── test_main.py            # Initial tests
    └── scripts/
        └── init_db.py              # Database initialization
```

---

## 4. Implementation Steps

### Step 1: Install uv Package Manager

```bash
# Install uv globally
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

**Expected:** `uv 0.5.x` or later.

### Step 2: Configure ASDF for Python

```bash
cd /export/workspace/article-mind
mkdir -p article-mind-service
cd article-mind-service

# Create ASDF version file
cat > .tool-versions << 'EOF'
python 3.12.12
EOF

# Install Python via ASDF
asdf install python 3.12.12
asdf reshim python
```

### Step 3: Initialize Python Project with uv

```bash
# Create pyproject.toml and project structure
uv init --lib article-mind-service

# Create Python version file for uv
echo "3.12.12" > .python-version

# Create src-layout structure
mkdir -p src/article_mind_service
touch src/article_mind_service/__init__.py
```

### Step 4: Configure pyproject.toml

Create comprehensive `pyproject.toml`:

```toml
[project]
name = "article-mind-service"
version = "0.1.0"
description = "Backend service for Article Mind knowledge management system"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
    "sqlalchemy>=2.0.36",
    "alembic>=1.13.0",
    "psycopg[binary]>=3.2.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.28.0",
    "ruff>=0.8.0",
    "mypy>=1.14.0",
    "bandit>=1.7.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by formatter)
    "B008",  # do not perform function calls in argument defaults (FastAPI Depends)
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports in __init__.py

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = "alembic.*"
ignore_missing_imports = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["src"]

[tool.bandit]
exclude_dirs = ["tests", "alembic/versions"]
```

### Step 5: Install Dependencies

```bash
# Install all dependencies including dev
uv pip install -e ".[dev]"
```

### Step 6: Create Makefile

Create `Makefile`:

```makefile
.PHONY: help install dev test lint format migrate clean

PYTHON := python
UV := uv

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	$(UV) pip install -e ".[dev]"

dev: ## Start development server with hot reload
	$(UV) run uvicorn article_mind_service.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run tests
	$(UV) run pytest

test-cov: ## Run tests with coverage
	$(UV) run pytest --cov=article_mind_service --cov-report=html --cov-report=term

lint: ## Run all linters (ruff, mypy, bandit)
	$(UV) run ruff check src tests
	$(UV) run mypy src
	$(UV) run bandit -r src

lint-fix: ## Auto-fix linting issues
	$(UV) run ruff check --fix src tests

format: ## Format code with ruff
	$(UV) run ruff format src tests

format-check: ## Check code formatting
	$(UV) run ruff format --check src tests

migrate: ## Run database migrations
	$(UV) run alembic upgrade head

migrate-down: ## Rollback one migration
	$(UV) run alembic downgrade -1

migrate-create: ## Create new migration (use: make migrate-create MSG="description")
	$(UV) run alembic revision --autogenerate -m "$(MSG)"

clean: ## Remove build artifacts and cache
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

shell: ## Start Python shell with app context
	$(UV) run python -i -c "from article_mind_service.main import app; print('App loaded. Use: app')"
```

### Step 7: Create Environment Configuration

Create `.env.example`:

```env
# Database Configuration
DATABASE_URL=postgresql://article_mind:article_mind@localhost:5432/article_mind

# API Configuration
API_V1_PREFIX=/api/v1
CORS_ORIGINS=http://localhost:5173

# Development Settings
DEBUG=true
LOG_LEVEL=INFO
```

Create `.env` (copy from example):

```bash
cp .env.example .env
```

Add to `.gitignore`:

```
.env
.env.local
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
htmlcov/
.coverage
dist/
build/
*.egg-info/
```

### Step 8: Create Configuration Management

Create `src/article_mind_service/config.py`:

```python
"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "postgresql://article_mind:article_mind@localhost:5432/article_mind"

    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:5173"]

    # App
    debug: bool = False
    log_level: str = "INFO"
    app_name: str = "Article Mind Service"
    app_version: str = "0.1.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
```

### Step 9: Create Database Session Management

Create `src/article_mind_service/database.py`:

```python
"""Database session management with async SQLAlchemy."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from .config import settings

# Convert postgresql:// to postgresql+asyncpg://
database_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(
    database_url,
    echo=settings.debug,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

### Step 10: Create FastAPI Application

Create `src/article_mind_service/main.py`:

```python
"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager."""
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")
    yield
    # Shutdown
    print("Shutting down application")


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


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Article Mind Service API"}
```

### Step 11: Initialize Alembic

```bash
# Initialize Alembic
uv run alembic init alembic

# This creates:
# - alembic.ini
# - alembic/ directory with env.py
```

### Step 12: Configure Alembic

Update `alembic.ini`:

```ini
# Change sqlalchemy.url to use environment variable
# sqlalchemy.url = driver://user:pass@localhost/dbname
# Comment out or remove the above line, we'll use env.py instead
```

Update `alembic/env.py`:

```python
"""Alembic environment configuration."""

from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Import your models Base
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from article_mind_service.database import Base
from article_mind_service.config import settings

# this is the Alembic Config object
config = context.config

# Set database URL from settings
config.set_main_option("sqlalchemy.url", settings.database_url)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with given connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = settings.database_url.replace(
        "postgresql://", "postgresql+asyncpg://"
    )

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    import asyncio

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Step 13: Create Initial Tests

Create `tests/conftest.py`:

```python
"""Pytest configuration and fixtures."""

from collections.abc import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from article_mind_service.main import app


@pytest.fixture
def client() -> TestClient:
    """Synchronous test client."""
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
```

Create `tests/test_main.py`:

```python
"""Tests for main application."""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


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
```

### Step 14: Create CLAUDE.md

Create `CLAUDE.md`:

```markdown
# Article Mind Service - Backend Development Guide

## Project Overview

FastAPI-based backend service for the Article Mind knowledge management system.

## Technology Stack

- **Language:** Python 3.12.12
- **Framework:** FastAPI 0.115+
- **ASGI Server:** Uvicorn with hot reload
- **Package Manager:** uv (10-100x faster than pip)
- **Database:** PostgreSQL 16.x
- **ORM:** SQLAlchemy 2.0 (async)
- **Migrations:** Alembic 1.13+
- **Code Quality:** ruff (linter + formatter), mypy (type checker), bandit (security)
- **Testing:** pytest with async support

## Development Commands

### Standard Workflow

```bash
make install    # Install dependencies with uv
make dev        # Start dev server (http://localhost:8000)
make test       # Run test suite
make lint       # Run linters (ruff, mypy, bandit)
make format     # Format code with ruff
make migrate    # Apply database migrations
```

### Database Migrations

```bash
make migrate-create MSG="add users table"  # Create new migration
make migrate                                # Apply migrations
make migrate-down                           # Rollback one migration
```

### Advanced Commands

```bash
make test-cov       # Tests with coverage report
make lint-fix       # Auto-fix linting issues
make format-check   # Check formatting without changes
make shell          # Python shell with app loaded
make clean          # Remove build artifacts
```

## Project Structure

```
src/article_mind_service/
├── main.py              # FastAPI app entry point
├── config.py            # Settings (Pydantic)
├── database.py          # SQLAlchemy async setup
├── models/              # SQLAlchemy models (database schema)
├── schemas/             # Pydantic schemas (API contracts)
├── routers/             # FastAPI routers (endpoints)
└── dependencies.py      # FastAPI dependencies
```

## API Contract

### OpenAPI Generation

FastAPI automatically generates OpenAPI spec at:
- **Interactive Docs:** http://localhost:8000/docs (Swagger UI)
- **Alternate Docs:** http://localhost:8000/redoc (ReDoc)
- **OpenAPI JSON:** http://localhost:8000/openapi.json

### Pydantic Schemas Define API Contract

**All request and response models MUST be Pydantic schemas.**

Example:

```python
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    version: str

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", version="1.0.0")
```

### API Contract Workflow

1. Define Pydantic schema in `schemas/`
2. Use `response_model` in route decorator
3. OpenAPI spec auto-updates at `/openapi.json`
4. Frontend runs `npm run gen:api` to generate TypeScript types
5. Type-safe communication guaranteed

## Database Management

### SQLAlchemy Models

Define database tables in `models/`:

```python
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from article_mind_service.database import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
```

### Migrations with Alembic

```bash
# After changing models, create migration
make migrate-create MSG="add users table"

# Review generated migration in alembic/versions/

# Apply migration
make migrate
```

### Database Sessions

Use FastAPI dependency injection:

```python
from sqlalchemy.ext.asyncio import AsyncSession
from article_mind_service.database import get_db

@app.get("/users")
async def get_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User))
    return result.scalars().all()
```

## Code Quality Standards

### Ruff (Linter + Formatter)

Replaces black, flake8, isort with single tool:

```bash
make lint       # Check for issues
make lint-fix   # Auto-fix issues
make format     # Format code
```

Configuration in `pyproject.toml`:
- Line length: 100 characters
- Python 3.12 target
- Enforces: pycodestyle, pyflakes, isort, bugbear, comprehensions, pyupgrade

### Mypy (Type Checker)

Strict type checking enabled:

```bash
make lint  # Includes mypy
```

**All functions MUST have type annotations:**

```python
# ✅ CORRECT
async def get_user(user_id: int, db: AsyncSession) -> User | None:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()

# ❌ WRONG (missing types)
async def get_user(user_id, db):
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
```

### Bandit (Security Linter)

Scans for security issues:

```bash
make lint  # Includes bandit
```

Common issues:
- SQL injection risks
- Hardcoded passwords
- Insecure random usage
- Shell injection vulnerabilities

## Testing Strategy

### Pytest with Async Support

```bash
make test       # Run all tests
make test-cov   # With coverage report
```

### Test Structure

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_user(async_client: AsyncClient) -> None:
    response = await async_client.post(
        "/api/v1/users",
        json={"email": "test@example.com"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
```

### Fixtures

Use `conftest.py` for shared fixtures:
- `client`: Synchronous TestClient
- `async_client`: Async HTTP client
- `db`: Test database session

## Environment Variables

Configure via `.env` (never commit `.env`, use `.env.example`):

```env
DATABASE_URL=postgresql://article_mind:article_mind@localhost:5432/article_mind
API_V1_PREFIX=/api/v1
CORS_ORIGINS=http://localhost:5173
DEBUG=true
LOG_LEVEL=INFO
```

Access in code via Settings:

```python
from article_mind_service.config import settings

print(settings.database_url)
```

## Common Pitfalls

### 1. Forgetting Async/Await

❌ **WRONG:**
```python
@app.get("/users")
def get_users(db: AsyncSession = Depends(get_db)):
    result = db.execute(select(User))  # Missing await!
    return result.scalars().all()
```

✅ **CORRECT:**
```python
@app.get("/users")
async def get_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User))
    return result.scalars().all()
```

### 2. Missing Response Model

❌ **WRONG:**
```python
@app.get("/health")
async def health_check():
    return {"status": "ok"}  # No type safety, no OpenAPI schema
```

✅ **CORRECT:**
```python
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", version="1.0.0")
```

### 3. Hardcoding Configuration

❌ **WRONG:**
```python
DATABASE_URL = "postgresql://localhost/mydb"
```

✅ **CORRECT:**
```python
from article_mind_service.config import settings
database_url = settings.database_url
```

### 4. Not Running Migrations

After changing SQLAlchemy models:

```bash
make migrate-create MSG="description"
make migrate
```

### 5. Forgetting to Import Models in Alembic

Alembic won't detect models unless imported. Ensure all models are imported in `models/__init__.py`:

```python
from .user import User
from .document import Document

__all__ = ["User", "Document"]
```

## Development Workflow

1. **Start dev server:** `make dev`
2. **Make changes** to routes/models/schemas
3. **Hot reload** happens automatically
4. **Run tests:** `make test`
5. **Check quality:** `make lint`
6. **Format code:** `make format`
7. **Create migration** (if models changed): `make migrate-create MSG="description"`
8. **Apply migration:** `make migrate`

## Package Management with uv

uv is 10-100x faster than pip:

```bash
# Install package
uv pip install httpx

# Update pyproject.toml
# Then sync environment
uv pip install -e ".[dev]"

# Or use uv add (upcoming feature)
uv add httpx
```

## ASDF Version Management

Python version is pinned in `.tool-versions`:

```
python 3.12.12
```

To install:

```bash
asdf install
```

## Guard Rails

- ✅ **DO:** Use Pydantic schemas for all API requests/responses
- ✅ **DO:** Use async/await for all database operations
- ✅ **DO:** Add type hints to all functions
- ✅ **DO:** Create migrations after model changes
- ✅ **DO:** Use dependency injection (FastAPI Depends)
- ✅ **DO:** Run tests before committing
- ❌ **DON'T:** Hardcode configuration values
- ❌ **DON'T:** Use synchronous database operations
- ❌ **DON'T:** Skip type annotations
- ❌ **DON'T:** Modify database schema without migrations
- ❌ **DON'T:** Return raw dicts from endpoints (use Pydantic)

## API Versioning

All API routes should use version prefix:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1")

@router.get("/users")
async def get_users():
    pass

# Full path: /api/v1/users
```

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [uv Documentation](https://github.com/astral-sh/uv)
```

---

## 5. CLAUDE.md Requirements

The project-specific CLAUDE.md documents:

- ✅ Technology stack (Python 3.12, FastAPI, uv, PostgreSQL)
- ✅ Package manager (uv) and why it's faster
- ✅ Development commands (Makefile targets)
- ✅ Database management (SQLAlchemy, Alembic)
- ✅ API contract generation (Pydantic → OpenAPI)
- ✅ Code quality tools (ruff, mypy, bandit)
- ✅ Testing strategy (pytest async)
- ✅ Environment variable management
- ✅ Common pitfalls and anti-patterns
- ✅ Guard rails and best practices

---

## 6. Verification Steps

### 1. Verify uv Installation

```bash
uv --version
```

**Expected:** `uv 0.5.x` or later.

### 2. Verify Python Version

```bash
python --version
```

**Expected:** `Python 3.12.12`

### 3. Verify Dependencies Installation

```bash
make install
```

**Expected:** All packages installed successfully with uv.

### 4. Verify Dev Server

```bash
make dev
```

**Expected:**
- Server starts on http://localhost:8000
- Visit http://localhost:8000 → `{"message": "Article Mind Service API"}`
- Visit http://localhost:8000/docs → Swagger UI loads
- Visit http://localhost:8000/openapi.json → OpenAPI spec

### 5. Verify Tests

```bash
make test
```

**Expected:** All tests pass (3 tests: root, root_async, openapi_docs).

### 6. Verify Linting

```bash
make lint
```

**Expected:** No linting errors from ruff, mypy, or bandit.

### 7. Verify Formatting

```bash
make format-check
```

**Expected:** All files properly formatted.

### 8. Verify Alembic Setup

```bash
uv run alembic current
```

**Expected:** Shows current migration status (empty for now).

### 9. Verify Environment Variables

```bash
cat .env
```

**Expected:** Contains `DATABASE_URL` and other config variables.

### 10. Verify ASDF

```bash
cat .tool-versions
```

**Expected:** `python 3.12.12`

---

## 7. Common Pitfalls

### Issue 1: uv Not Found

**Symptom:** `uv: command not found`

**Solution:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Add to PATH (usually automatic)
export PATH="$HOME/.cargo/bin:$PATH"
```

### Issue 2: Python Version Mismatch

**Symptom:** Wrong Python version active.

**Solution:**
```bash
asdf install python 3.12.12
asdf local python 3.12.12
asdf reshim python
```

### Issue 3: Port 8000 Already in Use

**Symptom:** Dev server fails to start.

**Solution:**
```bash
lsof -ti:8000 | xargs kill -9
# Or use different port
uvicorn article_mind_service.main:app --reload --port 8001
```

### Issue 4: Alembic Can't Find Models

**Symptom:** `alembic revision --autogenerate` doesn't detect changes.

**Solution:**
- Ensure models are imported in `alembic/env.py`
- Import `Base` from `database.py`
- Set `target_metadata = Base.metadata`

### Issue 5: Async Database Errors

**Symptom:** `RuntimeError: Event loop is closed`

**Solution:**
- Use `async def` for all route handlers
- Use `await` for all database operations
- Ensure `AsyncSession` from `get_db` dependency

### Issue 6: OpenAPI Spec Not Updating

**Symptom:** Changes to Pydantic schemas don't appear in `/openapi.json`.

**Solution:**
- Restart dev server (hot reload may not catch schema changes)
- Ensure `response_model` is set on route decorator
- Check for syntax errors in schema definitions

### Issue 7: Import Errors in Tests

**Symptom:** `ModuleNotFoundError: No module named 'article_mind_service'`

**Solution:**
- Ensure `pytest.ini_options.pythonpath = ["src"]` in `pyproject.toml`
- Or run: `export PYTHONPATH=src`
- Install package in editable mode: `uv pip install -e .`

---

## 8. Next Steps

After completing this plan:

1. **Proceed to Plan 03** (database-setup) to configure PostgreSQL
2. **Proceed to Plan 04** (service-health-api) to implement health check
3. **Set up API contract** following API contract instructions
4. **Create first migration** once database models are defined

---

## 9. Success Criteria

- ✅ FastAPI application runs on http://localhost:8000
- ✅ OpenAPI documentation accessible at /docs and /openapi.json
- ✅ All tests pass with pytest
- ✅ Linters (ruff, mypy, bandit) pass with no errors
- ✅ Makefile provides all standard commands
- ✅ ASDF pins Python version correctly
- ✅ uv package manager configured and working
- ✅ Alembic initialized for database migrations
- ✅ Environment variables configured in .env
- ✅ CLAUDE.md documents all workflows and guard rails
- ✅ Hot reload works for development

---

**Plan Status:** Ready for implementation
**Last Updated:** 2026-01-18
