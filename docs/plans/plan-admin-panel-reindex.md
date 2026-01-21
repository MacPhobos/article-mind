# Implementation Plan: Admin Panel with Reindexing

**Date:** 2026-01-21
**Status:** For Review
**Estimated Total Effort:** 20-28 hours

---

## Overview

This plan outlines a staged approach to implementing an Admin Panel with a reindexing feature that includes real-time progress tracking. The implementation is designed to be:

1. **Flexible** - Easy to add future admin tasks
2. **Well-abstracted** - Reusable progress tracking infrastructure
3. **User-friendly** - Real-time progress feedback via SSE

---

## Architecture Summary

### Backend (article-mind-service)
```
Admin Router (/api/v1/admin)
    ↓
[Start Task] → [Task Registry (in-memory)]
    ↓                ↓
Background Task  [SSE Progress Stream] → UI
    ↓
For each article:
    - Process
    - Update TaskRegistry
    - Emit SSE event
```

### Frontend (article-mind-ui)
```
TopBar → Admin Link
    ↓
/admin route → AdminPage
    ↓
[Task Cards Grid]
    ↓
[ReindexModal] → [SSE EventSource] → [ProgressBar]
```

---

## Staged Implementation

### Stage 1: Backend Progress Infrastructure (4-6 hours)

**Goal:** Create reusable progress tracking system with SSE streaming

#### 1.1 Add Dependencies

**File:** `article-mind-service/pyproject.toml`
```toml
dependencies = [
    ...
    "sse-starlette>=2.1.0",
]
```

#### 1.2 Create Task Registry Module

**New File:** `src/article_mind_service/tasks/registry.py`

Purpose: Generic, reusable task progress tracking system

```python
# Core interfaces
class TaskProgress(BaseModel):
    task_id: str
    task_type: str  # "reindex", "export", "import", etc.
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    total_items: int
    processed_items: int = 0
    failed_items: int = 0
    current_item: str | None = None
    message: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    errors: list[dict] = []

class TaskRegistry:
    """In-memory registry for tracking background tasks with SSE streaming."""

    def create_task(self, task_type: str, total_items: int) -> str
    async def update_progress(self, task_id: str, **kwargs) -> None
    async def stream_progress(self, task_id: str) -> AsyncGenerator[TaskProgress, None]
    def get_task(self, task_id: str) -> TaskProgress | None
    def cancel_task(self, task_id: str) -> bool
```

**Design Decisions:**
- In-memory storage (acceptable for admin operations)
- Uses `asyncio.Queue` for SSE event distribution
- Generic `task_type` field for future extensibility
- Thread-safe singleton pattern

#### 1.3 Create Admin Schemas

**New File:** `src/article_mind_service/schemas/admin.py`

```python
# Request/Response schemas
class AdminReindexRequest(BaseModel):
    session_ids: list[int] | None = None  # None = all sessions
    force: bool = False  # Force reindex even if status is 'completed'

class AdminReindexResponse(BaseModel):
    task_id: str
    total_sessions: int
    total_articles: int
    progress_url: str

class TaskStatusResponse(BaseModel):
    task_id: str
    task_type: str
    status: str
    progress_percent: int
    message: str | None
    errors: list[dict]
```

#### 1.4 Create Reindex Task Module

**New File:** `src/article_mind_service/tasks/reindex.py`

Purpose: Shared reindex logic used by both session and admin endpoints

```python
async def reindex_article(
    article: Article,
    pipeline: EmbeddingPipeline,
    db: AsyncSession,
    progress_callback: Callable | None = None,
) -> bool:
    """
    Reindex a single article with optional progress callback.

    Extracted from sessions.py for reuse in admin reindex.
    """

async def reindex_all_articles(
    task_id: str,
    session_ids: list[int] | None,
    force: bool,
    task_registry: TaskRegistry,
    db_session_factory: Callable,
):
    """
    Background task to reindex all articles across sessions.

    Updates TaskRegistry with progress for SSE streaming.
    """
```

#### 1.5 Unit Tests

**New File:** `tests/unit/test_task_registry.py`

- Test task creation and ID generation
- Test progress updates and SSE queue
- Test concurrent task tracking
- Test task cancellation

**Acceptance Criteria:**
- [ ] TaskRegistry singleton works correctly
- [ ] Progress updates are queued for SSE
- [ ] Multiple concurrent tasks are tracked independently
- [ ] All unit tests pass

---

### Stage 2: Backend Admin Endpoints (3-4 hours)

**Goal:** Implement admin reindex API with SSE progress streaming

#### 2.1 Create Admin Router

**New File:** `src/article_mind_service/routers/admin.py`

```python
router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

@router.post("/reindex", response_model=AdminReindexResponse)
async def start_reindex(request: AdminReindexRequest, ...):
    """Start reindexing all or selected sessions."""

@router.get("/reindex/{task_id}/progress")
async def stream_reindex_progress(task_id: str):
    """Stream reindex progress via Server-Sent Events."""
    return EventSourceResponse(event_generator())

@router.get("/reindex/{task_id}")
async def get_reindex_status(task_id: str):
    """Get final status of a reindex task."""

@router.post("/reindex/{task_id}/cancel")
async def cancel_reindex(task_id: str):
    """Request cancellation of a running reindex task."""
```

#### 2.2 Register Admin Router

**Modify:** `src/article_mind_service/main.py`

```python
from article_mind_service.routers import admin
app.include_router(admin.router)
```

#### 2.3 Update API Contract

**Modify:** `article-mind-service/docs/api-contract.md`

Add admin endpoints section with:
- Request/response schemas
- SSE event format documentation
- Example usage

#### 2.4 Integration Tests

**New File:** `tests/integration/test_admin_reindex.py`

- Test starting reindex task
- Test SSE progress streaming
- Test task cancellation
- Test error handling

**Acceptance Criteria:**
- [ ] `POST /api/v1/admin/reindex` returns task_id
- [ ] `GET /api/v1/admin/reindex/{task_id}/progress` streams SSE events
- [ ] Progress updates are accurate (within 1 article)
- [ ] Errors are captured and reported
- [ ] All integration tests pass

---

### Stage 3: Frontend Admin Infrastructure (3-4 hours)

**Goal:** Create admin panel page and reusable components

#### 3.1 Add Admin Navigation

**Modify:** `src/lib/components/TopBar.svelte`

```svelte
<li class:active={isAdminActive}>
  <a href="/admin">Admin</a>
</li>
```

#### 3.2 Create Admin Page

**New File:** `src/routes/admin/+page.svelte`

```svelte
<script lang="ts">
  let showReindexModal = $state(false);
</script>

<main class="admin-page">
  <h1>Administration</h1>

  <section class="admin-tasks">
    <h2>System Tasks</h2>

    <div class="task-cards">
      <AdminTaskCard
        title="Reindex All Articles"
        description="Regenerate embeddings for all articles across all sessions"
        icon="refresh"
        onclick={() => showReindexModal = true}
      />

      <!-- Future task cards go here -->
    </div>
  </section>
</main>

<ReindexModal
  isOpen={showReindexModal}
  onClose={() => showReindexModal = false}
/>
```

#### 3.3 Create Reusable Components

**New File:** `src/lib/components/admin/AdminTaskCard.svelte`

Props: `title`, `description`, `icon`, `onclick`

**New File:** `src/lib/components/ProgressBar.svelte`

```svelte
<script lang="ts">
  interface Props {
    value: number;      // 0-100
    label?: string;
    variant?: 'default' | 'success' | 'error';
    indeterminate?: boolean;
  }
</script>

<div class="progress-bar" class:indeterminate>
  <div class="progress-fill" style="width: {value}%"></div>
  {#if label}
    <span class="progress-label">{label}</span>
  {/if}
</div>
```

**Acceptance Criteria:**
- [ ] Admin link appears in TopBar
- [ ] /admin route renders admin page
- [ ] Task cards are clickable
- [ ] ProgressBar component works with all variants

---

### Stage 4: Frontend SSE Integration (4-6 hours)

**Goal:** Implement ReindexModal with SSE progress streaming

#### 4.1 Create SSE Utility

**New File:** `src/lib/api/sse.ts`

```typescript
export interface ProgressEvent {
  task_id: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  processed_items: number;
  total_items: number;
  message: string | null;
  errors: Array<{ article_id: number; error: string }>;
}

export function subscribeToProgress(
  taskId: string,
  onProgress: (event: ProgressEvent) => void,
  onError?: (error: Event) => void,
): () => void {
  const url = `${API_BASE_URL}/api/v1/admin/reindex/${taskId}/progress`;
  const eventSource = new EventSource(url);

  eventSource.addEventListener('progress', (e) => {
    onProgress(JSON.parse(e.data));
  });

  eventSource.addEventListener('complete', (e) => {
    const data = JSON.parse(e.data);
    onProgress({ ...data, status: 'completed' });
    eventSource.close();
  });

  eventSource.onerror = (e) => {
    onError?.(e);
    eventSource.close();
  };

  return () => eventSource.close();
}
```

#### 4.2 Add Admin API Functions

**New File:** `src/lib/api/admin.ts`

```typescript
export interface ReindexOptions {
  sessionIds?: number[];
  force?: boolean;
}

export interface ReindexResult {
  task_id: string;
  total_sessions: number;
  total_articles: number;
  progress_url: string;
}

export async function startReindex(options: ReindexOptions = {}): Promise<ReindexResult> {
  return apiClient.post('/api/v1/admin/reindex', {
    session_ids: options.sessionIds ?? null,
    force: options.force ?? false,
  });
}

export async function cancelReindex(taskId: string): Promise<void> {
  await apiClient.post(`/api/v1/admin/reindex/${taskId}/cancel`, {});
}
```

#### 4.3 Create ReindexModal Component

**New File:** `src/lib/components/admin/ReindexModal.svelte`

```svelte
<script lang="ts">
  import ProgressBar from '../ProgressBar.svelte';
  import { startReindex, cancelReindex } from '$lib/api/admin';
  import { subscribeToProgress, type ProgressEvent } from '$lib/api/sse';

  interface Props {
    isOpen: boolean;
    onClose: () => void;
  }

  let { isOpen, onClose }: Props = $props();

  // State
  let status = $state<'idle' | 'running' | 'completed' | 'failed'>('idle');
  let progress = $state(0);
  let message = $state('');
  let taskId = $state<string | null>(null);
  let errors = $state<Array<{ article_id: number; error: string }>>([]);
  let unsubscribe: (() => void) | null = null;

  // Lifecycle: cleanup SSE on close
  $effect(() => {
    if (!isOpen && unsubscribe) {
      unsubscribe();
      unsubscribe = null;
    }
  });

  async function handleStart() {
    status = 'running';
    message = 'Starting reindex...';
    progress = 0;
    errors = [];

    try {
      const result = await startReindex({ force: true });
      taskId = result.task_id;
      message = `Reindexing ${result.total_articles} articles...`;

      // Subscribe to progress
      unsubscribe = subscribeToProgress(
        result.task_id,
        handleProgress,
        handleSSEError
      );
    } catch (e) {
      status = 'failed';
      message = e instanceof Error ? e.message : 'Failed to start reindex';
    }
  }

  function handleProgress(event: ProgressEvent) {
    progress = Math.round((event.processed_items / event.total_items) * 100);
    message = event.message ?? `Processed ${event.processed_items} of ${event.total_items}`;
    errors = event.errors;

    if (event.status === 'completed') {
      status = 'completed';
      message = `Completed! ${event.processed_items} articles reindexed.`;
    } else if (event.status === 'failed') {
      status = 'failed';
    }
  }

  function handleSSEError(e: Event) {
    status = 'failed';
    message = 'Connection lost. Please try again.';
  }

  async function handleCancel() {
    if (taskId) {
      await cancelReindex(taskId);
      status = 'idle';
      message = 'Cancelled';
    }
  }

  function handleClose() {
    if (status === 'running') {
      // Confirm before closing during operation
      if (!confirm('Reindex is in progress. Close anyway?')) {
        return;
      }
    }
    // Reset state
    status = 'idle';
    progress = 0;
    message = '';
    taskId = null;
    errors = [];
    onClose();
  }
</script>

{#if isOpen}
  <div class="modal-overlay" role="dialog" aria-modal="true">
    <div class="modal-content reindex-modal">
      <header class="modal-header">
        <h2>Reindex All Articles</h2>
        <button class="close-btn" onclick={handleClose}>&times;</button>
      </header>

      <div class="modal-body">
        {#if status === 'idle'}
          <p>This will regenerate embeddings for all articles across all sessions.</p>
          <p class="warning">This operation may take several minutes depending on the number of articles.</p>
        {:else}
          <ProgressBar value={progress} label="{progress}%" />
          <p class="status-message">{message}</p>

          {#if errors.length > 0}
            <details class="error-details">
              <summary>{errors.length} error(s)</summary>
              <ul>
                {#each errors as error}
                  <li>Article {error.article_id}: {error.error}</li>
                {/each}
              </ul>
            </details>
          {/if}
        {/if}
      </div>

      <footer class="modal-footer">
        {#if status === 'idle'}
          <button class="btn-cancel" onclick={handleClose}>Cancel</button>
          <button class="btn-primary" onclick={handleStart}>Start Reindex</button>
        {:else if status === 'running'}
          <button class="btn-cancel" onclick={handleCancel}>Cancel Operation</button>
        {:else}
          <button class="btn-primary" onclick={handleClose}>Close</button>
        {/if}
      </footer>
    </div>
  </div>
{/if}
```

**Acceptance Criteria:**
- [ ] Modal opens when clicking task card
- [ ] Start button triggers reindex API
- [ ] Progress bar updates in real-time via SSE
- [ ] Errors are displayed if any occur
- [ ] Cancel button stops the operation
- [ ] Cleanup happens when modal closes

---

### Stage 5: End-to-End Testing & Polish (4-6 hours)

**Goal:** Comprehensive testing, error handling, and UX polish

#### 5.1 Backend E2E Tests

**File:** `tests/e2e/test_admin_reindex_flow.py`

- Complete reindex flow with mock articles
- SSE event sequence verification
- Cancellation mid-operation
- Error recovery scenarios

#### 5.2 Frontend E2E Tests (Playwright)

**File:** `article-mind-ui/tests/admin-reindex.spec.ts`

- Navigate to admin page
- Start reindex and verify progress
- Cancel operation
- Handle connection loss gracefully

#### 5.3 UX Polish

- Add loading skeleton for admin page
- Add toast notifications for completion/errors
- Add keyboard shortcuts (Escape to close, Enter to confirm)
- Add operation history (last 5 reindex operations)

#### 5.4 Documentation

- Update README with admin panel usage
- Add inline code documentation
- Create user guide for admin operations

**Acceptance Criteria:**
- [ ] All E2E tests pass
- [ ] UX is smooth and responsive
- [ ] Error states are handled gracefully
- [ ] Documentation is complete

---

## File Summary

### New Files (Backend - 8 files)

| File | Purpose |
|------|---------|
| `src/article_mind_service/tasks/registry.py` | Generic task progress tracking |
| `src/article_mind_service/tasks/reindex.py` | Shared reindex logic |
| `src/article_mind_service/routers/admin.py` | Admin API endpoints |
| `src/article_mind_service/schemas/admin.py` | Admin request/response schemas |
| `tests/unit/test_task_registry.py` | Registry unit tests |
| `tests/integration/test_admin_reindex.py` | Admin API integration tests |
| `tests/e2e/test_admin_reindex_flow.py` | End-to-end reindex tests |

### New Files (Frontend - 7 files)

| File | Purpose |
|------|---------|
| `src/routes/admin/+page.svelte` | Admin dashboard page |
| `src/lib/components/admin/AdminTaskCard.svelte` | Task card component |
| `src/lib/components/admin/ReindexModal.svelte` | Reindex progress modal |
| `src/lib/components/ProgressBar.svelte` | Reusable progress bar |
| `src/lib/api/sse.ts` | SSE utility functions |
| `src/lib/api/admin.ts` | Admin API client functions |
| `tests/admin-reindex.spec.ts` | Playwright E2E tests |

### Modified Files (5 files)

| File | Change |
|------|--------|
| `article-mind-service/pyproject.toml` | Add sse-starlette dependency |
| `article-mind-service/src/main.py` | Register admin router |
| `article-mind-service/docs/api-contract.md` | Add admin endpoints |
| `article-mind-ui/src/lib/components/TopBar.svelte` | Add Admin nav link |

---

## Estimated Timeline

| Stage | Effort | Dependencies |
|-------|--------|--------------|
| Stage 1: Backend Progress Infrastructure | 4-6 hours | None |
| Stage 2: Backend Admin Endpoints | 3-4 hours | Stage 1 |
| Stage 3: Frontend Admin Infrastructure | 3-4 hours | None (parallel with Stage 2) |
| Stage 4: Frontend SSE Integration | 4-6 hours | Stages 2 & 3 |
| Stage 5: E2E Testing & Polish | 4-6 hours | Stage 4 |
| **Total** | **20-28 hours** | |

**Parallel Work:** Stages 1-2 (backend) and Stage 3 (frontend) can proceed in parallel.

---

## Future Extensibility

The architecture supports adding future admin tasks with minimal effort:

### Adding a New Admin Task

1. **Backend:**
   - Add task type to `TaskRegistry.create_task(task_type="new_task")`
   - Create `tasks/new_task.py` with business logic
   - Add endpoints to `routers/admin.py`

2. **Frontend:**
   - Add new `AdminTaskCard` to admin page
   - Create `NewTaskModal.svelte` (copy ReindexModal pattern)
   - Reuse `ProgressBar` and `subscribeToProgress`

### Potential Future Tasks

- **Export Session Data** - Export articles/embeddings to file
- **Import Articles** - Bulk import from CSV/JSON
- **Clear Cache** - Clear various caches
- **Database Maintenance** - Vacuum, reindex DB indexes
- **User Management** - If authentication is added

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SSE connection drops | Medium | Low | Auto-reconnect, polling fallback |
| Large reindex overwhelms server | Low | Medium | Rate limiting, batch size tuning |
| Task state lost on restart | Medium | Low | Acceptable for admin ops; can add persistence later |
| Concurrent reindex conflicts | Low | Medium | Single-task enforcement or queuing |

---

## Open Questions

1. **Authentication:** Should admin endpoints require special auth?
   - Recommendation: Yes, add later when auth system is implemented

2. **Concurrent Tasks:** Allow multiple reindex tasks simultaneously?
   - Recommendation: No, enforce single task for simplicity

3. **Persistent History:** Store reindex history in database?
   - Recommendation: Not for MVP; add if needed later

4. **Selective Reindex:** Allow reindexing specific sessions from UI?
   - Recommendation: Yes, add session selector to modal in future iteration

---

## Approval Checklist

- [ ] Architecture approach approved
- [ ] API contract approved
- [ ] Stage breakdown approved
- [ ] Timeline estimate approved
- [ ] Risk mitigations accepted

---

**Next Steps:** Upon approval, proceed with Stage 1 (Backend Progress Infrastructure).
