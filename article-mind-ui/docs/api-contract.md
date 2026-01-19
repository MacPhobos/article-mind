# Article Mind API Contract

> **Version**: 1.0.0
> **Last Updated**: 2026-01-18
> **Status**: FROZEN - Changes require version bump and UI sync

This document defines the REST API contract between `article-mind-ui` (frontend) and `article-mind-service` (backend).

---

## Table of Contents

1. [Contract Philosophy](#contract-philosophy)
2. [Type Generation Workflow](#type-generation-workflow)
3. [Base Configuration](#base-configuration)
4. [Common Types](#common-types)
5. [API Endpoints](#api-endpoints)
6. [Error Handling](#error-handling)
7. [Status Codes](#status-codes)
8. [Frontend Guard Rails](#frontend-guard-rails)
9. [Testing API Integration](#testing-api-integration)
10. [Common Pitfalls](#common-pitfalls)
11. [Version Compatibility](#version-compatibility)
12. [Reference](#reference)

---

## Contract Philosophy

**IMPORTANT**: This is a **frozen contract** - both frontend and backend MUST adhere to these exact specifications.

### Core Principles

- **Backend defines Pydantic schemas** → Generates OpenAPI spec
- **Frontend generates TypeScript types** from OpenAPI spec
- **Type-safe communication** guaranteed
- **No breaking changes** without coordination

### Change Protocol

1. Backend proposes API change in `article-mind-service/docs/api-contract.md`
2. Both teams review and approve
3. Backend implements change with version bump if breaking
4. Backend deploys to development
5. Frontend runs `make gen-api` to update types
6. Frontend implements changes
7. Both teams test integration

---

## Type Generation Workflow

### Backend to Frontend

```
┌─────────────────────┐
│ Backend Pydantic    │
│ Schema Definition   │
│ (schemas/*.py)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ FastAPI Auto-       │
│ Generates OpenAPI   │
│ (/openapi.json)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Frontend Runs       │
│ make gen-api        │
│ (openapi-typescript)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ TypeScript Types    │
│ Generated           │
│ (lib/api/generated) │
└─────────────────────┘
```

### Example Flow

**Backend (article-mind-service):**

```python
# schemas/health.py
from pydantic import BaseModel
from typing import Literal

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    version: str
    database: Literal["connected", "disconnected"]

# routers/health.py
@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="1.0.0",
        database="connected"
    )
```

**Frontend (article-mind-ui):**

```bash
# Generate TypeScript types
make gen-api
```

```typescript
// Auto-generated in src/lib/api/generated.ts
export interface HealthResponse {
  status: "ok" | "degraded" | "error";
  version: string;
  database: "connected" | "disconnected";
}

// Usage in components
import type { HealthResponse } from '$lib/api/generated';
import { apiClient } from '$lib/api/client';

const health = await apiClient.get<HealthResponse>('/health');
console.log(health.status); // Type-safe!
```

---

## Base Configuration

| Environment | Base URL |
|-------------|----------|
| Development | `http://localhost:8000` |
| Production  | `https://api.article-mind.com` (TBD) |

All endpoints are prefixed with `/api/v1` **except** `/health` and `/openapi.json`.

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

**Example:**

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid email format",
    "details": {
      "field": "email",
      "value": "not-an-email"
    }
  }
}
```

---

## API Endpoints

### Health Check

#### `GET /health`

Health check endpoint. No authentication required. No `/api/v1` prefix.

**Response** `200 OK`

```typescript
interface HealthResponse {
  status: "ok" | "degraded" | "error";
  version: string;
  database: "connected" | "disconnected";
}
```

**Status Values:**
- `ok`: All systems operational (database connected)
- `degraded`: Service running but database unavailable
- `error`: Critical failure (not currently used, reserved for future)

**Examples:**

Healthy system:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "database": "connected"
}
```

Degraded system (database down):
```json
{
  "status": "degraded",
  "version": "1.0.0",
  "database": "disconnected"
}
```

**Design Decision: Graceful Degradation**

The health endpoint ALWAYS returns HTTP 200, even when the database is down. This allows monitoring systems to distinguish between:
- **Service completely dead** (connection refused, timeout)
- **Service alive but degraded** (returns 200 with status: "degraded")

This pattern enables smarter load balancing and alerts.

---

## Error Handling

### Error Response Shape

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description",
    "details": { ... }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `NOT_FOUND` | 404 | Resource not found |
| `INTERNAL_ERROR` | 500 | Server error |

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

## Frontend Guard Rails

### Type Safety (CRITICAL)

- ✅ **ALWAYS** use generated TypeScript types from `src/lib/api/generated.ts`
- ❌ **NEVER** manually type API responses
- ❌ **NEVER** use `any` or `@ts-ignore` on API calls
- ✅ **ALWAYS** run `make gen-api` after backend schema changes

**Why?**
- Compile-time safety prevents runtime errors
- IDE autocomplete shows exact API shape
- Breaking changes caught during development

### API Client Usage

- ✅ **ALWAYS** use centralized API client in `src/lib/api/client.ts`
- ❌ **NEVER** use raw `fetch()` for API calls
- ✅ **ALWAYS** handle errors using client's error handling
- ❌ **NEVER** suppress API errors silently

**Why?**
- Consistent error handling across app
- Centralized request/response logging
- Easy to add authentication headers
- Request/response interceptors in one place

### Environment Configuration

- ✅ **ALWAYS** use `VITE_API_BASE_URL` environment variable
- ❌ **NEVER** hardcode API URLs (http://localhost:8000, production URLs, etc.)
- ✅ Configure in `.env` for development
- ✅ Configure in `.env.production` for production

**Why?**
- Easy environment switching
- No accidental production API calls in development
- Deployment flexibility

---

## Testing API Integration

### Mock with Generated Types

**Always use generated types for test mocks to ensure type safety:**

```typescript
import type { HealthResponse } from '$lib/api/generated';
import { describe, it, expect } from 'vitest';

describe('Health Check', () => {
  it('should match API contract', () => {
    const mockHealth: HealthResponse = {
      status: "ok",
      version: "1.0.0",
      database: "connected"
    };

    // TypeScript enforces contract
    expect(mockHealth.status).toBe("ok");
  });
});
```

### Test Contract Compliance

**Verify responses match the contract:**

```typescript
import { apiClient } from '$lib/api/client';
import type { HealthResponse } from '$lib/api/generated';
import { describe, it, expect } from 'vitest';

describe('Health API', () => {
  it('returns valid health response', async () => {
    const health = await apiClient.get<HealthResponse>('/health');

    // Status must be one of allowed values
    expect(['ok', 'degraded', 'error']).toContain(health.status);

    // Database must be one of allowed values
    expect(['connected', 'disconnected']).toContain(health.database);

    // Version must be semantic versioning
    expect(health.version).toMatch(/^\d+\.\d+\.\d+$/);
  });

  it('handles degraded state', async () => {
    // Mock degraded response
    const health = await apiClient.get<HealthResponse>('/health');

    if (health.status === 'degraded') {
      expect(health.database).toBe('disconnected');
    }
  });
});
```

### Integration Test Pattern

```typescript
import { apiClient } from '$lib/api/client';
import { describe, it, expect, beforeAll } from 'vitest';

describe('API Integration Tests', () => {
  beforeAll(async () => {
    // Ensure backend is running
    const health = await apiClient.get('/health');
    expect(health.status).toBeDefined();
  });

  it('follows API contract for all endpoints', async () => {
    // Test each endpoint matches generated types
  });
});
```

---

## Common Pitfalls

### ❌ Pitfall 1: Manual Type Definitions

**WRONG:**
```typescript
// DON'T DO THIS
interface HealthResponse {
  status: string;  // Too loose! Should be union type
  version: string;
  database: string;  // Missing constraints!
}

const health = await fetch('/health').then(r => r.json()) as HealthResponse;
```

**CORRECT:**
```typescript
// DO THIS
import type { HealthResponse } from '$lib/api/generated';
import { apiClient } from '$lib/api/client';

const health = await apiClient.get<HealthResponse>('/health');
```

**Why?**
- Generated types include exact literal types (`"ok" | "degraded" | "error"`)
- Manual types are too loose and miss validation
- Changes to backend automatically update types

### ❌ Pitfall 2: Ignoring Type Errors

**WRONG:**
```typescript
// DON'T DO THIS
const health = await fetch('/health').then(r => r.json()) as any;
console.log(health.status); // No type safety!

// OR THIS
const health = await apiClient.get('/health');
// @ts-ignore
health.nonexistent_field; // Runtime error!
```

**CORRECT:**
```typescript
// DO THIS
import type { HealthResponse } from '$lib/api/generated';

const health = await apiClient.get<HealthResponse>('/health');

// TypeScript enforces correct field access
console.log(health.status); // ✅ Type-safe
// console.log(health.nonexistent); // ❌ Compile error
```

### ❌ Pitfall 3: Hardcoded API URLs

**WRONG:**
```typescript
// DON'T DO THIS
const response = await fetch('http://localhost:8000/health');

// OR THIS
const API_URL = 'https://api.article-mind.com';
const response = await fetch(`${API_URL}/health`);
```

**CORRECT:**
```typescript
// DO THIS
import { apiClient } from '$lib/api/client';

const health = await apiClient.get('/health');
```

**Why?**
- Environment-specific URLs handled by client
- Easy to switch between dev/staging/production
- Centralized configuration

### ❌ Pitfall 4: Silent Error Handling

**WRONG:**
```typescript
// DON'T DO THIS
try {
  const data = await apiClient.get('/api/v1/articles');
} catch (error) {
  // Silent failure - user has no idea what happened
}

// OR THIS
const data = await apiClient.get('/api/v1/articles').catch(() => null);
```

**CORRECT:**
```typescript
// DO THIS
try {
  const data = await apiClient.get('/api/v1/articles');
  // Handle success
} catch (error) {
  console.error('Failed to fetch articles:', error);
  // Show user-friendly error message
  showErrorToast('Could not load articles. Please try again.');
}
```

### ❌ Pitfall 5: Not Regenerating Types

**WRONG:**
```typescript
// Backend adds new field "uptime" to HealthResponse
// Frontend doesn't run make gen-api
// Code still compiles but new field is missing!

const health = await apiClient.get<HealthResponse>('/health');
console.log(health.uptime); // undefined! Field doesn't exist in old types
```

**CORRECT:**
```bash
# Backend adds new field to HealthResponse
# Frontend runs:
make gen-api

# Now TypeScript knows about new field
const health = await apiClient.get<HealthResponse>('/health');
console.log(health.uptime); // ✅ Type-safe access to new field
```

### ❌ Pitfall 6: Incorrect Response Type

**WRONG:**
```typescript
// DON'T DO THIS
import type { HealthResponse } from '$lib/api/generated';

// Wrong endpoint but claiming it returns HealthResponse
const articles = await apiClient.get<HealthResponse>('/api/v1/articles');
```

**CORRECT:**
```typescript
// DO THIS
import type { HealthResponse, ArticleListResponse } from '$lib/api/generated';

const health = await apiClient.get<HealthResponse>('/health');
const articles = await apiClient.get<ArticleListResponse>('/api/v1/articles');
```

---

## Version Compatibility

### Semantic Versioning

Backend follows semantic versioning (e.g., 1.0.0):

- **Major version** (1.x.x): Breaking changes - frontend MUST update
- **Minor version** (x.1.x): New features (backwards compatible)
- **Patch version** (x.x.1): Bug fixes (no API changes)

### Version Checking

Frontend should check `/health` endpoint for version compatibility:

```typescript
import { apiClient } from '$lib/api/client';
import type { HealthResponse } from '$lib/api/generated';

async function checkApiVersion() {
  const health = await apiClient.get<HealthResponse>('/health');
  const [major, minor, patch] = health.version.split('.').map(Number);

  const EXPECTED_MAJOR = 1;

  if (major !== EXPECTED_MAJOR) {
    throw new Error(
      `API version mismatch! Expected ${EXPECTED_MAJOR}.x.x, got ${health.version}`
    );
  }

  console.log(`API version ${health.version} is compatible`);
}
```

### Breaking Change Protocol

When backend releases a breaking change:

1. Backend bumps major version (1.0.0 → 2.0.0)
2. Backend updates `docs/api-contract.md` with changelog
3. Frontend runs `make gen-api` to get new types
4. TypeScript compilation errors highlight breaking changes
5. Frontend updates code to match new contract
6. Both teams test integration before production deploy

---

## Reference

### Documentation

- **Backend API Contract**: `article-mind-service/docs/api-contract.md`
- **Frontend CLAUDE.md**: `article-mind-ui/CLAUDE.md` (API Contract Integration section)

### Development Endpoints

- **Backend Health**: http://localhost:8000/health
- **Backend OpenAPI JSON**: http://localhost:8000/openapi.json
- **Backend Swagger UI**: http://localhost:8000/docs
- **Backend ReDoc**: http://localhost:8000/redoc

### Commands

```bash
# Frontend type generation
cd article-mind-ui
make gen-api

# Or directly with npm
npm run gen:api

# Backend dev server
cd article-mind-service
make dev

# Frontend dev server
cd article-mind-ui
make dev
```

### Type Generation Details

**Command:**
```bash
openapi-typescript http://localhost:8000/openapi.json -o src/lib/api/generated.ts
```

**Requirements:**
- Backend must be running (http://localhost:8000)
- `/openapi.json` endpoint must be accessible
- `openapi-typescript` package installed

**Output:**
- Generated types in `src/lib/api/generated.ts`
- **DO NOT EDIT** this file manually
- Regenerate after every backend schema change

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-18 | Initial frontend contract with /health endpoint documentation |

---

_This contract is the source of truth for frontend-backend communication. Both implementations must conform to these definitions._
