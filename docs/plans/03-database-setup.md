# Plan 03: PostgreSQL Database Configuration

**Plan ID:** 03-database-setup
**Created:** 2026-01-18
**Dependencies:** Plan 02 (scaffold-service) - requires Alembic configuration
**Estimated Time:** 1-2 hours

---

## 1. Overview

### Purpose
Set up a local PostgreSQL database for development, configure connection pooling, create database user and schema, and verify database connectivity from the FastAPI service.

### Scope
- Install PostgreSQL 16.x locally
- Create database and user with appropriate permissions
- Configure SQLAlchemy connection with async support
- Create initial database migration
- Verify end-to-end database connectivity

### Dependencies
- **Plan 02:** Requires Alembic configuration and database.py setup

### Outputs
- Running PostgreSQL instance on localhost:5432
- Database `article_mind` with user `article_mind`
- SQLAlchemy async connection verified
- Initial Alembic migration applied

---

## 2. Technology Stack

### Database
- **PostgreSQL:** 16.x (latest stable)
- **Client Library:** psycopg 3.x (with binary drivers for async)
- **Connection Pool:** SQLAlchemy AsyncEngine with asyncpg

### Tools
- **PostgreSQL CLI:** `psql` for administration
- **ASDF Plugin:** `postgres` for version management (optional)
- **Docker Alternative:** PostgreSQL via Docker (recommended for isolation)

---

## 3. Directory Structure

No new directories required. Database configuration files:

```
article-mind-service/
├── .env                        # Updated with DB credentials
├── alembic/
│   └── versions/
│       └── 001_initial.py      # Initial migration (created in this plan)
└── scripts/
    └── init_db.py              # Database initialization helper
```

---

## 4. Implementation Steps

### Step 1: Install PostgreSQL

**Option A: Using ASDF (Recommended for consistent versioning)**

```bash
# Install PostgreSQL plugin
asdf plugin add postgres

# Install PostgreSQL 16.6
asdf install postgres 16.6
asdf local postgres 16.6

# Initialize PostgreSQL data directory
export PGDATA="$HOME/.asdf/installs/postgres/16.6/data"
pg_ctl init -D "$PGDATA"

# Start PostgreSQL
pg_ctl start -D "$PGDATA" -l "$PGDATA/postgres.log"
```

**Option B: Using Docker (Recommended for isolation)**

```bash
# Create Docker network for article-mind services
docker network create article-mind-network

# Run PostgreSQL container
docker run -d \
  --name article-mind-postgres \
  --network article-mind-network \
  -e POSTGRES_USER=article_mind \
  -e POSTGRES_PASSWORD=article_mind \
  -e POSTGRES_DB=article_mind \
  -p 5432:5432 \
  -v article-mind-pgdata:/var/lib/postgresql/data \
  postgres:16-alpine

# Verify container is running
docker ps | grep article-mind-postgres
```

**Option C: System Package Manager (Ubuntu/Debian)**

```bash
# Add PostgreSQL APT repository
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget -qO- https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo tee /etc/apt/trusted.gpg.d/pgdg.asc

# Install PostgreSQL 16
sudo apt update
sudo apt install -y postgresql-16 postgresql-contrib-16

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### Step 2: Create Database and User

**If using Docker:**

```bash
# Already created via environment variables
# Verify:
docker exec -it article-mind-postgres psql -U article_mind -d article_mind -c '\l'
```

**If using ASDF or system PostgreSQL:**

```bash
# Create user and database
psql postgres << 'EOF'
CREATE USER article_mind WITH PASSWORD 'article_mind';
CREATE DATABASE article_mind OWNER article_mind;
GRANT ALL PRIVILEGES ON DATABASE article_mind TO article_mind;
EOF

# Verify
psql -U article_mind -d article_mind -c '\conninfo'
```

### Step 3: Update Environment Configuration

Update `article-mind-service/.env`:

```env
# Database Configuration
DATABASE_URL=postgresql://article_mind:article_mind@localhost:5432/article_mind

# For Docker (if using custom network):
# DATABASE_URL=postgresql://article_mind:article_mind@article-mind-postgres:5432/article_mind

# API Configuration
API_V1_PREFIX=/api/v1
CORS_ORIGINS=http://localhost:5173

# Development Settings
DEBUG=true
LOG_LEVEL=INFO
```

### Step 4: Install Async PostgreSQL Driver

Ensure `psycopg[binary]` is in dependencies (should be from Plan 02):

```bash
cd article-mind-service
uv pip install "psycopg[binary]>=3.2.0"
```

**Note:** The `[binary]` extra provides pre-compiled C extensions for better performance.

### Step 5: Verify Database Connection

Create `scripts/test_db.py`:

```python
"""Test database connection."""

import asyncio
from sqlalchemy import text

from article_mind_service.database import engine


async def test_connection() -> None:
    """Test async database connection."""
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✅ Database connection successful!")
            print(f"PostgreSQL version: {version}")

            # Test basic query
            result = await conn.execute(text("SELECT 1 + 1 AS result"))
            value = result.scalar()
            print(f"Test query result: 1 + 1 = {value}")

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_connection())
```

Run test:

```bash
uv run python scripts/test_db.py
```

**Expected Output:**
```
✅ Database connection successful!
PostgreSQL version: PostgreSQL 16.6 on ...
Test query result: 1 + 1 = 2
```

### Step 6: Create Initial Migration

Create `scripts/init_db.py` for database schema initialization:

```python
"""Initialize database with base schema."""

import asyncio

from article_mind_service.database import Base, engine


async def init_db() -> None:
    """Create all database tables."""
    print("Creating database tables...")

    async with engine.begin() as conn:
        # Drop all tables (use with caution!)
        # await conn.run_sync(Base.metadata.drop_all)

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

    print("✅ Database tables created successfully!")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(init_db())
```

### Step 7: Create Alembic Initial Migration

```bash
cd article-mind-service

# Create initial migration
make migrate-create MSG="initial schema"

# This generates: alembic/versions/XXXX_initial_schema.py
```

Review the generated migration file and ensure it looks correct.

### Step 8: Apply Initial Migration

```bash
make migrate
```

**Expected Output:**
```
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade  -> XXXX, initial schema
```

### Step 9: Verify Migration Status

```bash
uv run alembic current
```

**Expected:** Shows current migration version.

```bash
uv run alembic history
```

**Expected:** Shows migration history.

### Step 10: Add Database Health Check

Update `article_mind_service/main.py` to include database health check:

```python
"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .config import settings
from .database import engine


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
        print(f"❌ Database connection failed: {e}")
        raise

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


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Article Mind Service API"}
```

### Step 11: Update CLAUDE.md with Database Instructions

Add to `article-mind-service/CLAUDE.md`:

```markdown
## Database Setup

### Local Development (Docker)

Start PostgreSQL container:

```bash
docker run -d \
  --name article-mind-postgres \
  -e POSTGRES_USER=article_mind \
  -e POSTGRES_PASSWORD=article_mind \
  -e POSTGRES_DB=article_mind \
  -p 5432:5432 \
  -v article-mind-pgdata:/var/lib/postgresql/data \
  postgres:16-alpine
```

Stop container:

```bash
docker stop article-mind-postgres
docker rm article-mind-postgres
```

### Database Connection

Connection string format:

```
postgresql://[user]:[password]@[host]:[port]/[database]
```

For async SQLAlchemy (automatically converted in `database.py`):

```
postgresql+asyncpg://[user]:[password]@[host]:[port]/[database]
```

### Common Database Operations

```bash
# Connect to database
docker exec -it article-mind-postgres psql -U article_mind -d article_mind

# Or with system psql
psql -U article_mind -d article_mind -h localhost

# Common psql commands
\l          # List databases
\dt         # List tables
\d table    # Describe table
\q          # Quit
```

### Troubleshooting

**Connection refused:**
- Check PostgreSQL is running: `docker ps` or `pg_ctl status`
- Verify port 5432 is not blocked
- Check firewall settings

**Authentication failed:**
- Verify credentials in `.env` match database
- Check `pg_hba.conf` for authentication method

**Migration conflicts:**
- Check current status: `uv run alembic current`
- Rollback if needed: `make migrate-down`
- Recreate migration: `make migrate-create MSG="description"`
```

---

## 5. CLAUDE.md Requirements

Database-related documentation added to `article-mind-service/CLAUDE.md`:

- ✅ PostgreSQL installation options (Docker, ASDF, system)
- ✅ Database connection string format
- ✅ Common database operations (psql commands)
- ✅ Migration workflow (create, apply, rollback)
- ✅ Troubleshooting common issues
- ✅ Docker commands for starting/stopping database

---

## 6. Verification Steps

### 1. Verify PostgreSQL is Running

**Docker:**
```bash
docker ps | grep article-mind-postgres
```

**ASDF/System:**
```bash
pg_ctl status -D "$PGDATA"
# Or
sudo systemctl status postgresql
```

**Expected:** Service running on port 5432.

### 2. Verify Database Exists

```bash
# Docker
docker exec -it article-mind-postgres psql -U article_mind -d article_mind -c '\l'

# ASDF/System
psql -U article_mind -d article_mind -c '\l'
```

**Expected:** Database `article_mind` listed with owner `article_mind`.

### 3. Verify Database Connection from Python

```bash
cd article-mind-service
uv run python scripts/test_db.py
```

**Expected:**
```
✅ Database connection successful!
PostgreSQL version: PostgreSQL 16.6 on ...
Test query result: 1 + 1 = 2
```

### 4. Verify FastAPI Startup with Database

```bash
make dev
```

**Expected Console Output:**
```
Starting Article Mind Service v0.1.0
✅ Database connection verified
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 5. Verify Alembic Migration Status

```bash
uv run alembic current
```

**Expected:** Shows current migration head (not empty).

### 6. Verify Tables Created

```bash
# Docker
docker exec -it article-mind-postgres psql -U article_mind -d article_mind -c '\dt'

# ASDF/System
psql -U article_mind -d article_mind -c '\dt'
```

**Expected:** Lists `alembic_version` table (at minimum).

### 7. Test Database Query from FastAPI

Add temporary test endpoint to verify database access:

```python
# In article_mind_service/main.py
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_db
from sqlalchemy import text
from fastapi import Depends

@app.get("/test-db")
async def test_db(db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    """Test database query."""
    result = await db.execute(text("SELECT version()"))
    version = result.scalar()
    return {"database": "connected", "version": version}
```

Visit: http://localhost:8000/test-db

**Expected:** JSON response with database version.

---

## 7. Common Pitfalls

### Issue 1: PostgreSQL Port Already in Use

**Symptom:** `port 5432 is already allocated`

**Solution:**
```bash
# Find process using port 5432
lsof -i :5432

# Kill process (if safe)
kill -9 <PID>

# Or use different port
docker run -p 5433:5432 ...
# Update DATABASE_URL to use port 5433
```

### Issue 2: Connection Refused

**Symptom:** `psycopg.OperationalError: connection refused`

**Solution:**
```bash
# Verify PostgreSQL is running
docker ps | grep postgres

# Check logs
docker logs article-mind-postgres

# Restart container
docker restart article-mind-postgres
```

### Issue 3: Authentication Failed

**Symptom:** `psycopg.OperationalError: password authentication failed`

**Solution:**
- Verify credentials in `.env` match Docker environment variables
- Recreate container with correct credentials:
  ```bash
  docker rm -f article-mind-postgres
  docker run -d \
    --name article-mind-postgres \
    -e POSTGRES_USER=article_mind \
    -e POSTGRES_PASSWORD=article_mind \
    -e POSTGRES_DB=article_mind \
    -p 5432:5432 \
    postgres:16-alpine
  ```

### Issue 4: Alembic Can't Connect to Database

**Symptom:** `alembic upgrade head` fails with connection error

**Solution:**
- Ensure `.env` file exists and `DATABASE_URL` is set
- Verify `alembic/env.py` loads settings from `config.py`
- Check PostgreSQL is accessible

### Issue 5: Migration File Empty

**Symptom:** `alembic revision --autogenerate` creates empty migration

**Solution:**
- Ensure models are imported in `alembic/env.py`
- Set `target_metadata = Base.metadata`
- Import all model classes before running Alembic

### Issue 6: Async/Await Errors with Database

**Symptom:** `RuntimeError: no running event loop`

**Solution:**
- Use `async def` for route handlers
- Use `await` for all database operations
- Use `AsyncSession` from `get_db` dependency

### Issue 7: Docker Volume Permissions

**Symptom:** PostgreSQL container fails to start with permission errors

**Solution:**
```bash
# Remove volume and recreate
docker volume rm article-mind-pgdata
docker run -d ... postgres:16-alpine
```

---

## 8. Next Steps

After completing this plan:

1. **Proceed to Plan 04** (service-health-api) to implement health check endpoint
2. **Create database models** in `models/` as needed for features
3. **Generate migrations** whenever models change
4. **Set up database backups** for production (future)

---

## 9. Success Criteria

- ✅ PostgreSQL 16.x running on localhost:5432 (or Docker container)
- ✅ Database `article_mind` created with owner `article_mind`
- ✅ Database connection successful from Python (test script passes)
- ✅ FastAPI starts successfully with database health check
- ✅ Alembic migrations configured and initial migration applied
- ✅ `make migrate` command works without errors
- ✅ SQLAlchemy async connection pool working
- ✅ CLAUDE.md updated with database setup instructions
- ✅ No connection errors in FastAPI startup logs

---

## 10. Production Considerations (Future)

While not required for initial development, document these for future reference:

### Connection Pooling
```python
# In database.py
engine = create_async_engine(
    database_url,
    echo=settings.debug,
    pool_size=20,          # Max connections in pool
    max_overflow=10,       # Connections beyond pool_size
    pool_timeout=30,       # Seconds to wait for connection
    pool_recycle=3600,     # Recycle connections after 1 hour
)
```

### Read Replicas
- Configure separate read-only connection for queries
- Use write connection only for INSERT/UPDATE/DELETE
- Implement connection routing based on operation type

### Database Backups
```bash
# Automated daily backups
docker exec article-mind-postgres pg_dump -U article_mind article_mind > backup.sql

# Restore from backup
docker exec -i article-mind-postgres psql -U article_mind article_mind < backup.sql
```

### Monitoring
- Enable PostgreSQL query logging
- Monitor slow queries with pg_stat_statements
- Set up alerting for connection pool exhaustion

---

**Plan Status:** Ready for implementation
**Last Updated:** 2026-01-18
