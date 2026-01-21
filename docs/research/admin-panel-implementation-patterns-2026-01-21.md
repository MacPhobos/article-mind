# Admin Panel Implementation Patterns Analysis

**Research Date**: 2026-01-21
**Project**: article-mind-ui
**Purpose**: Understand existing patterns for implementing admin panel with progress modals
**Researcher**: Claude Code (Research Agent)

---

## Executive Summary

The article-mind-ui codebase uses **SvelteKit 2.x with Svelte 5 Runes API** as its foundation. The application has well-established patterns for:
- Modal management (demonstrated in CreateSessionModal, DeleteSessionModal)
- API integration via centralized client
- Loading states with visual feedback
- Top navigation structure

**Key Finding**: No existing SSE/streaming implementation found, but the architecture is ready for it. Modal patterns are clean and reusable. Navigation is straightforward to extend.

---

## Tech Stack Summary

### Framework & Core
- **Framework**: SvelteKit 2.x (file-based routing)
- **UI Library**: Svelte 5 with Runes API (`$state`, `$derived`, `$effect`, `$props`)
- **Language**: TypeScript 5.x (strict mode)
- **Build Tool**: Vite 7.x
- **Package Manager**: npm
- **Node Version**: 22.13.1 (ASDF managed)

### State Management
- **Local Component State**: Svelte 5 Runes (`$state`)
- **Derived State**: `$derived` for computed values
- **Side Effects**: `$effect` for lifecycle operations
- **No Global State Library**: Pure Svelte reactive system

### UI Component Patterns
- **No External UI Library**: Custom components with CSS
- **Modal Pattern**: Overlay-based, prop-driven visibility
- **Loading States**: Inline spinners and disabled states
- **Form Handling**: Controlled inputs with Svelte bindings

### API Integration
- **Client**: Custom `ApiClient` class (wraps `fetch`)
- **Type Generation**: `openapi-typescript` from backend OpenAPI spec
- **Base URL**: Environment variable (`VITE_API_BASE_URL`, defaults to `http://localhost:13010`)
- **Error Handling**: Centralized in `apiClient.fetch()`

### Real-time Capabilities
- **Current Status**: None implemented
- **No SSE**: No existing EventSource usage
- **No WebSockets**: No socket.io or native WebSocket usage
- **Potential**: Architecture supports adding SSE via native `EventSource` API

---

## Navigation Structure

### Current Implementation

**File**: `src/lib/components/TopBar.svelte`

```svelte
<nav class="top-bar">
  <div class="logo">
    <a href="/">Article Mind</a>
  </div>
  <ul class="nav-menu">
    <li class:active={isSessionsActive}>
      <a href="/">Sessions</a>
    </li>
  </ul>
</nav>
```

**Active State Logic**:
```typescript
let currentPath = $derived($page.url.pathname);
let isSessionsActive = $derived(currentPath === '/' || currentPath.startsWith('/sessions'));
```

### Adding Admin Navigation

**Recommended Approach**:

```svelte
<script lang="ts">
  import { page } from '$app/stores';
  import { resolve } from '$app/paths';

  let currentPath = $derived($page.url.pathname);
  let isSessionsActive = $derived(currentPath === '/' || currentPath.startsWith('/sessions'));
  let isAdminActive = $derived(currentPath.startsWith('/admin'));
</script>

<nav class="top-bar">
  <div class="logo">
    <a href={resolve('/')}>Article Mind</a>
  </div>
  <ul class="nav-menu">
    <li class:active={isSessionsActive}>
      <a href={resolve('/')}>Sessions</a>
    </li>
    <li class:active={isAdminActive}>
      <a href={resolve('/admin')}>Admin</a>
    </li>
  </ul>
</nav>
```

**Routing**:
- Create: `src/routes/admin/+page.svelte` for admin panel
- SvelteKit will automatically handle routing

---

## Modal Implementation Patterns

### Existing Modal Architecture

The codebase has two modal components demonstrating consistent patterns:

1. **CreateSessionModal.svelte** - Form-based modal
2. **DeleteSessionModal.svelte** - Confirmation modal

### Modal Component Anatomy

**Props Interface**:
```typescript
interface Props {
  isOpen: boolean;           // Visibility control
  onClose: () => void;       // Close callback
  onSubmit: (data: T) => Promise<void>; // Async action
  session?: SessionResponse | null; // Optional context data
}
```

**State Management**:
```typescript
let isSubmitting = $state(false); // Prevent double-submit
let error = $state<string | null>(null); // Error display
```

**Structure**:
```svelte
{#if isOpen}
  <div class="modal-overlay"
       role="dialog"
       aria-modal="true"
       onkeydown={handleKeydown}>
    <div class="modal-content">
      <header class="modal-header">
        <h2>{title}</h2>
        <button class="close-btn" onclick={onClose}>&times;</button>
      </header>

      <div class="modal-body">
        {#if error}
          <div class="error-message" role="alert">{error}</div>
        {/if}
        <!-- Modal content -->
      </div>

      <footer class="modal-footer">
        <button class="btn-cancel" onclick={onClose} disabled={isSubmitting}>
          Cancel
        </button>
        <button class="btn-submit" onclick={handleAction} disabled={isSubmitting}>
          {#if isSubmitting}
            Loading...
          {:else}
            Confirm
          {/if}
        </button>
      </footer>
    </div>
  </div>
{/if}
```

**Accessibility Features**:
- `role="dialog"` and `aria-modal="true"`
- Keyboard escape handler
- Focus management via `tabindex="-1"`
- Error announcements via `role="alert"`

**Styling Pattern**:
```css
.modal-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 8px;
  width: 100%;
  max-width: 500px;
  margin: 1rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}
```

### Modal Usage Example

From `src/routes/+page.svelte`:

```svelte
<script lang="ts">
  // State
  let showCreateModal = $state(false);
  let editingSession = $state<SessionResponse | null>(null);

  // Open/close handlers
  function openCreateModal() {
    editingSession = null;
    showCreateModal = true;
  }

  function closeCreateModal() {
    showCreateModal = false;
    editingSession = null;
  }

  // Action handler
  async function handleCreateSession(data: CreateSessionRequest) {
    if (editingSession) {
      await apiClient.patch(`/api/v1/sessions/${editingSession.id}`, data);
    } else {
      await apiClient.post('/api/v1/sessions', data);
    }
    await loadSessions();
  }
</script>

<!-- Modal component -->
<CreateSessionModal
  isOpen={showCreateModal}
  session={editingSession}
  onClose={closeCreateModal}
  onSubmit={handleCreateSession}
/>
```

---

## API Integration Approach

### API Client Architecture

**File**: `src/lib/api/client.ts`

```typescript
export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers
        }
      });

      if (!response.ok) {
        let errorMessage = 'API request failed';
        try {
          const errorData = await response.json();
          errorMessage = extractErrorMessage(errorData);
        } catch (parseError) {
          errorMessage = response.statusText || 'API request failed';
        }
        throw new Error(errorMessage);
      }

      return response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  async get<T>(endpoint: string): Promise<T>
  async post<T>(endpoint: string, data: unknown): Promise<T>
  async patch<T>(endpoint: string, data: unknown): Promise<T>
  async delete<T>(endpoint: string): Promise<T>
}

export const apiClient = new ApiClient();
```

**Key Features**:
- Centralized error handling
- Type-safe responses via generics
- Automatic JSON headers
- Environment-based base URL
- Error message extraction (handles FastAPI and custom formats)

### Type Safety Pattern

**Generated Types** (`src/lib/api/generated.ts`):
```typescript
// Auto-generated from backend OpenAPI spec via:
// npm run gen:api
export interface SessionResponse { ... }
export interface CreateSessionRequest { ... }
```

**Manual Types** (`src/lib/api/types.ts`):
```typescript
// Temporary manual types until OpenAPI generation is complete
export type SessionStatus = 'draft' | 'active' | 'completed' | 'archived';
export interface SessionResponse {
  id: number;
  name: string;
  status: SessionStatus;
  // ...
}
```

**Usage in Components**:
```typescript
import type { SessionResponse } from '$lib/api/types';
import { apiClient } from '$lib/api/client';

async function loadSessions() {
  const response = await apiClient.get<SessionListResponse>('/api/v1/sessions');
  sessions = response.sessions;
}
```

---

## Progress UI Patterns

### Existing Loading States

#### 1. Inline Spinner (FileUploadDropzone)

**Component**: `src/lib/components/FileUploadDropzone.svelte`

```svelte
<script lang="ts">
  let loading = $state(false);

  async function handleFiles(files: FileList | null) {
    try {
      loading = true;
      await uploadArticleFile(sessionId, file);
    } finally {
      loading = false;
    }
  }
</script>

{#if loading}
  <div class="loading-content">
    <span class="spinner"></span>
    <p>Uploading...</p>
  </div>
{:else}
  <!-- Normal content -->
{/if}

<style>
  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid #e5e7eb;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
```

#### 2. Button Disabled State (CreateSessionModal)

```svelte
<button type="submit" class="btn-submit" disabled={isSubmitting}>
  {#if isSubmitting}
    Saving...
  {:else}
    {submitLabel}
  {/if}
</button>

<style>
  .btn-submit:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
</style>
```

#### 3. Typing Indicator (ChatContainer)

```svelte
{#if isSending}
  <div class="typing-indicator">
    <span></span><span></span><span></span>
  </div>
{/if}

<style>
  .typing-indicator {
    display: flex;
    gap: 4px;
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

### No Existing Progress Bars

**Observation**: No progress bar components exist yet. Will need to create:
- Linear progress bar for percentage-based tasks
- Circular progress for determinate operations
- Indeterminate progress for unknown duration

---

## Real-time/Streaming Recommendations

### Current State
- **No SSE Implementation**: No `EventSource` usage found
- **No WebSocket Implementation**: No socket connections
- **Chat is Request/Response**: `ChatContainer.svelte` uses standard POST requests

### Recommended Approach for Progress Events

**Option 1: Server-Sent Events (SSE) - RECOMMENDED**

SSE is ideal for server-to-client progress updates:

```typescript
// src/lib/api/sse.ts
export interface ProgressEvent {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number; // 0-100
  message: string;
  metadata?: Record<string, unknown>;
}

export function subscribeToProgress(
  taskId: string,
  onProgress: (event: ProgressEvent) => void,
  onError?: (error: Error) => void
): () => void {
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:13010';
  const eventSource = new EventSource(`${API_BASE_URL}/api/v1/admin/tasks/${taskId}/progress`);

  eventSource.addEventListener('progress', (event) => {
    const data: ProgressEvent = JSON.parse(event.data);
    onProgress(data);
  });

  eventSource.addEventListener('error', (event) => {
    console.error('SSE error:', event);
    onError?.(new Error('Connection lost'));
    eventSource.close();
  });

  // Return cleanup function
  return () => {
    eventSource.close();
  };
}
```

**Usage in Component**:
```svelte
<script lang="ts">
  import { subscribeToProgress } from '$lib/api/sse';

  let progress = $state(0);
  let status = $state<string>('idle');
  let message = $state<string>('');

  async function startReindex() {
    const response = await apiClient.post<{ task_id: string }>('/api/v1/admin/reindex', {});

    const unsubscribe = subscribeToProgress(
      response.task_id,
      (event) => {
        progress = event.progress;
        status = event.status;
        message = event.message;

        if (event.status === 'completed' || event.status === 'failed') {
          unsubscribe();
        }
      },
      (error) => {
        console.error('Progress error:', error);
      }
    );
  }
</script>
```

**Option 2: Polling (Fallback)**

For environments without SSE support:

```typescript
export async function pollTaskStatus(
  taskId: string,
  interval: number = 1000
): Promise<ProgressEvent> {
  return new Promise((resolve, reject) => {
    const poll = setInterval(async () => {
      try {
        const status = await apiClient.get<ProgressEvent>(`/api/v1/admin/tasks/${taskId}`);

        if (status.status === 'completed') {
          clearInterval(poll);
          resolve(status);
        } else if (status.status === 'failed') {
          clearInterval(poll);
          reject(new Error(status.message));
        }
      } catch (error) {
        clearInterval(poll);
        reject(error);
      }
    }, interval);
  });
}
```

---

## Recommended Admin Panel Architecture

### File Structure

```
src/
├── routes/
│   └── admin/
│       ├── +page.svelte              # Admin dashboard
│       └── +layout.svelte            # Admin layout (optional)
├── lib/
│   ├── components/
│   │   ├── admin/
│   │   │   ├── ReindexModal.svelte   # Progress modal for reindex
│   │   │   ├── ProgressBar.svelte    # Reusable progress component
│   │   │   └── TaskCard.svelte       # Admin task card
│   │   └── TopBar.svelte             # (modified to add Admin link)
│   └── api/
│       ├── admin.ts                  # Admin API functions
│       └── sse.ts                    # SSE subscription utilities
```

### Component Implementation Plan

#### 1. ProgressBar Component (Reusable)

```svelte
<!-- src/lib/components/admin/ProgressBar.svelte -->
<script lang="ts">
  interface Props {
    progress: number; // 0-100
    variant?: 'blue' | 'green' | 'red';
    showPercentage?: boolean;
    indeterminate?: boolean;
  }

  let {
    progress,
    variant = 'blue',
    showPercentage = true,
    indeterminate = false
  }: Props = $props();

  let clampedProgress = $derived(Math.min(100, Math.max(0, progress)));
</script>

<div class="progress-container">
  <div class="progress-bar" class:indeterminate>
    <div
      class="progress-fill {variant}"
      style="width: {indeterminate ? '100%' : clampedProgress}%"
    ></div>
  </div>
  {#if showPercentage && !indeterminate}
    <span class="progress-text">{clampedProgress}%</span>
  {/if}
</div>

<style>
  .progress-container {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .progress-bar {
    flex: 1;
    height: 8px;
    background: #e5e7eb;
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    transition: width 0.3s ease;
    border-radius: 4px;
  }

  .progress-fill.blue { background: #3b82f6; }
  .progress-fill.green { background: #10b981; }
  .progress-fill.red { background: #ef4444; }

  .indeterminate .progress-fill {
    animation: slide 1.5s infinite;
  }

  @keyframes slide {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }

  .progress-text {
    font-size: 0.875rem;
    color: #6b7280;
    min-width: 3rem;
    text-align: right;
  }
</style>
```

#### 2. ReindexModal Component

```svelte
<!-- src/lib/components/admin/ReindexModal.svelte -->
<script lang="ts">
  import { subscribeToProgress, type ProgressEvent } from '$lib/api/sse';
  import { apiClient } from '$lib/api/client';
  import ProgressBar from './ProgressBar.svelte';

  interface Props {
    isOpen: boolean;
    onClose: () => void;
    onComplete?: () => void;
  }

  let { isOpen, onClose, onComplete }: Props = $props();

  let isStarting = $state(false);
  let isRunning = $state(false);
  let progress = $state(0);
  let status = $state<string>('idle');
  let message = $state<string>('');
  let error = $state<string | null>(null);
  let taskId = $state<string | null>(null);

  let unsubscribe: (() => void) | null = null;

  async function startReindex() {
    isStarting = true;
    error = null;

    try {
      const response = await apiClient.post<{ task_id: string }>('/api/v1/admin/reindex', {});
      taskId = response.task_id;
      isStarting = false;
      isRunning = true;

      unsubscribe = subscribeToProgress(
        response.task_id,
        handleProgressUpdate,
        handleProgressError
      );
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to start reindex';
      isStarting = false;
    }
  }

  function handleProgressUpdate(event: ProgressEvent) {
    progress = event.progress;
    status = event.status;
    message = event.message;

    if (event.status === 'completed') {
      isRunning = false;
      unsubscribe?.();
      onComplete?.();
    } else if (event.status === 'failed') {
      isRunning = false;
      error = event.message;
      unsubscribe?.();
    }
  }

  function handleProgressError(err: Error) {
    error = err.message;
    isRunning = false;
    unsubscribe?.();
  }

  function handleClose() {
    if (isRunning && !confirm('Reindex is in progress. Close anyway?')) {
      return;
    }
    unsubscribe?.();
    onClose();
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Escape' && !isRunning) {
      handleClose();
    }
  }
</script>

{#if isOpen}
  <div
    class="modal-overlay"
    role="dialog"
    aria-modal="true"
    onkeydown={handleKeydown}
    tabindex="-1"
  >
    <div class="modal-content">
      <header class="modal-header">
        <h2>Reindex All Articles</h2>
        <button
          class="close-btn"
          onclick={handleClose}
          disabled={isRunning}
          aria-label="Close modal"
        >
          &times;
        </button>
      </header>

      <div class="modal-body">
        {#if error}
          <div class="error-message" role="alert">{error}</div>
        {/if}

        {#if !isRunning && !isStarting}
          <p>This will rebuild the search index for all articles.</p>
          <p class="warning">This may take several minutes depending on the number of articles.</p>
        {:else}
          <div class="progress-section">
            <ProgressBar
              {progress}
              variant={status === 'failed' ? 'red' : 'blue'}
              indeterminate={isStarting}
            />
            <p class="status-message">{message || 'Initializing...'}</p>
          </div>
        {/if}
      </div>

      <footer class="modal-footer">
        <button
          class="btn-cancel"
          onclick={handleClose}
          disabled={isRunning || isStarting}
        >
          {isRunning ? 'Running...' : 'Cancel'}
        </button>
        {#if !isRunning}
          <button
            class="btn-submit"
            onclick={startReindex}
            disabled={isStarting}
          >
            {#if isStarting}
              Starting...
            {:else}
              Start Reindex
            {/if}
          </button>
        {/if}
      </footer>
    </div>
  </div>
{/if}

<style>
  /* Same modal styles as CreateSessionModal */
  .modal-overlay { /* ... */ }
  .modal-content { /* ... */ }
  .modal-header { /* ... */ }
  .modal-body { /* ... */ }
  .modal-footer { /* ... */ }

  .warning {
    font-size: 0.875rem;
    color: #f59e0b;
    font-weight: 500;
  }

  .progress-section {
    margin: 1.5rem 0;
  }

  .status-message {
    margin-top: 0.75rem;
    font-size: 0.875rem;
    color: #6b7280;
    text-align: center;
  }

  .error-message {
    background: #fef2f2;
    color: #dc2626;
    padding: 0.75rem 1rem;
    border-radius: 4px;
    margin-bottom: 1rem;
    font-size: 0.9rem;
  }
</style>
```

#### 3. Admin Page

```svelte
<!-- src/routes/admin/+page.svelte -->
<script lang="ts">
  import ReindexModal from '$lib/components/admin/ReindexModal.svelte';

  let showReindexModal = $state(false);

  function openReindexModal() {
    showReindexModal = true;
  }

  function closeReindexModal() {
    showReindexModal = false;
  }

  function handleReindexComplete() {
    // Optional: Show toast or refresh stats
    console.log('Reindex completed successfully');
  }
</script>

<div class="page-container">
  <header class="page-header">
    <h1>Admin Panel</h1>
  </header>

  <div class="admin-grid">
    <div class="admin-card">
      <h3>Search Index Management</h3>
      <p>Rebuild the search index for all articles.</p>
      <button class="btn-primary" onclick={openReindexModal}>
        Reindex All Articles
      </button>
    </div>

    <!-- Future: Add more admin tasks here -->
  </div>
</div>

<ReindexModal
  isOpen={showReindexModal}
  onClose={closeReindexModal}
  onComplete={handleReindexComplete}
/>

<style>
  .page-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
  }

  .page-header {
    margin-bottom: 2rem;
  }

  .page-header h1 {
    margin: 0;
    font-size: 2rem;
    color: #333;
  }

  .admin-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1.5rem;
  }

  .admin-card {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .admin-card h3 {
    margin: 0 0 0.5rem 0;
    font-size: 1.25rem;
    color: #333;
  }

  .admin-card p {
    margin: 0 0 1rem 0;
    color: #666;
    font-size: 0.9rem;
  }

  .btn-primary {
    padding: 0.75rem 1.5rem;
    background: #1976d2;
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s ease;
  }

  .btn-primary:hover {
    background: #1565c0;
  }
</style>
```

---

## Files Requiring Modification

### 1. Navigation (Required)
- **File**: `src/lib/components/TopBar.svelte`
- **Change**: Add "Admin" navigation link
- **Lines**: 20-24

### 2. New Files (Admin Panel)
- `src/routes/admin/+page.svelte` - Admin dashboard
- `src/lib/components/admin/ReindexModal.svelte` - Reindex progress modal
- `src/lib/components/admin/ProgressBar.svelte` - Reusable progress bar
- `src/lib/api/admin.ts` - Admin API functions
- `src/lib/api/sse.ts` - SSE utilities

### 3. Optional: Shared Modal Styles
Consider extracting modal styles to shared stylesheet:
- Create: `src/lib/styles/modal.css`
- Import in components: `@import '$lib/styles/modal.css';`

---

## Existing Components to Leverage

### 1. Modal Overlay Pattern
**Source**: `CreateSessionModal.svelte`, `DeleteSessionModal.svelte`
- Fixed overlay with backdrop
- Centered content box
- Keyboard escape handling
- Accessibility attributes

### 2. Loading Spinner
**Source**: `FileUploadDropzone.svelte` (lines 201-213)
- CSS-only spinner animation
- Can be extracted to reusable component

### 3. Error Message Display
**Pattern**: Consistent across all components
```svelte
{#if error}
  <div class="error-message" role="alert">{error}</div>
{/if}
```

### 4. Button States
**Source**: All modals
- Disabled state during async operations
- Loading text replacement
- Opacity and cursor styling

---

## Best Practices from Codebase

### 1. Component Prop Typing
Always use TypeScript interfaces for props:
```typescript
interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: T) => Promise<void>;
}
let { isOpen, onClose, onSubmit }: Props = $props();
```

### 2. Async Error Handling
Consistent try-catch pattern:
```typescript
async function handleAction() {
  isLoading = true;
  error = null;

  try {
    await apiClient.post('/endpoint', data);
    onClose();
  } catch (e) {
    error = e instanceof Error ? e.message : 'Operation failed';
  } finally {
    isLoading = false;
  }
}
```

### 3. State Cleanup
Use `$effect` for cleanup when modal state changes:
```typescript
$effect(() => {
  if (isOpen) {
    // Reset state when modal opens
    error = null;
    progress = 0;
  }
});
```

### 4. Accessibility
- Always include ARIA roles and labels
- Support keyboard navigation (Escape key)
- Use semantic HTML elements
- Provide alternative text for icons

---

## Implementation Checklist

### Phase 1: Navigation & Routing
- [ ] Add "Admin" link to `TopBar.svelte`
- [ ] Create `src/routes/admin/+page.svelte`
- [ ] Test navigation and routing

### Phase 2: Reusable Components
- [ ] Create `ProgressBar.svelte`
- [ ] Test progress bar with static values
- [ ] Add indeterminate mode support

### Phase 3: API Integration
- [ ] Create `src/lib/api/admin.ts` with reindex endpoint
- [ ] Create `src/lib/api/sse.ts` with SSE utilities
- [ ] Test API calls (unit tests if available)

### Phase 4: Progress Modal
- [ ] Create `ReindexModal.svelte`
- [ ] Integrate SSE progress updates
- [ ] Add error handling and edge cases
- [ ] Test modal lifecycle (open/close/cleanup)

### Phase 5: Integration
- [ ] Add modal to admin page
- [ ] Test end-to-end flow
- [ ] Add loading states and error feedback
- [ ] Verify accessibility

### Phase 6: Polish
- [ ] Extract shared modal styles (optional)
- [ ] Add toast notifications for completion (optional)
- [ ] Add confirmation before closing during operation
- [ ] Document usage in CLAUDE.md

---

## Open Questions

1. **Backend SSE Support**: Does the backend have SSE endpoints for task progress?
   - If not, fallback to polling approach
   - Backend needs: `/api/v1/admin/tasks/{task_id}/progress` endpoint

2. **Authentication**: Should admin panel require special permissions?
   - No auth system detected in frontend
   - Consider adding auth check in `+page.svelte` or `+layout.server.ts`

3. **Task Persistence**: Should we show historical reindex tasks?
   - Could add task list to admin dashboard
   - Requires backend endpoint for task history

4. **Notifications**: Should we add toast notifications for task completion?
   - No toast system exists yet
   - Could create simple toast component or use existing patterns

---

## Conclusion

**Summary**: The article-mind-ui codebase is well-structured for adding an admin panel with progress modals. Existing modal patterns are clean, reusable, and accessible. The API client is type-safe and centralized. No SSE implementation exists yet, but the architecture supports it cleanly.

**Recommended Path**:
1. Add navigation link to TopBar
2. Create admin route page
3. Build reusable ProgressBar component
4. Implement SSE utilities for real-time updates
5. Create ReindexModal with progress tracking
6. Integrate modal into admin page

**Estimated Complexity**: Medium
- Modal patterns are well-established (low complexity)
- SSE implementation is new (medium complexity)
- API integration follows existing patterns (low complexity)

**Dependencies**:
- Backend must provide SSE endpoint for task progress
- Backend must provide POST endpoint for triggering reindex
- Frontend needs no new npm packages (native EventSource API)

---

## References

- **SvelteKit Routing**: https://kit.svelte.dev/docs/routing
- **Svelte 5 Runes**: https://svelte.dev/docs/svelte/what-are-runes
- **Server-Sent Events**: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- **EventSource API**: https://developer.mozilla.org/en-US/docs/Web/API/EventSource
