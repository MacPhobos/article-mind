# Plan 05: UI Health Indicator in Header

**Plan ID:** 05-ui-health-display
**Created:** 2026-01-18
**Dependencies:** Plan 01 (scaffold-ui), Plan 04 (service-health-api)
**Estimated Time:** 2-3 hours

---

## 1. Overview

### Purpose
Implement a visual health status indicator in the UI header that calls the backend `/health` endpoint and displays service status with color-coded indicators.

### Scope
- Generate TypeScript types from backend OpenAPI spec
- Create health check API client function
- Build reusable HealthIndicator component
- Add health indicator to root layout header
- Implement automatic polling with configurable interval
- Handle loading, success, and error states
- Write component tests

### Dependencies
- **Plan 01:** Requires SvelteKit scaffolding and API client
- **Plan 04:** Requires `/health` endpoint to be implemented

### Outputs
- TypeScript types generated from OpenAPI spec
- Working health indicator in header
- Reusable HealthIndicator component
- Component tests
- Visual feedback for service status

---

## 2. Technology Stack

Same as Plan 01:
- **SvelteKit:** 2.x
- **Svelte:** 5.x (Runes API for state management)
- **TypeScript:** 5.x
- **openapi-typescript:** For type generation
- **Vitest:** For component testing

---

## 3. Directory Structure

```
article-mind-ui/
└── src/
    ├── lib/
    │   ├── api/
    │   │   ├── client.ts
    │   │   ├── generated.ts       # NEW: Generated from OpenAPI
    │   │   └── health.ts          # NEW: Health check client
    │   └── components/
    │       └── HealthIndicator.svelte  # NEW: Health indicator component
    └── routes/
        └── +layout.svelte         # UPDATED: Add header with health indicator
```

---

## 4. Implementation Steps

### Step 1: Ensure Backend is Running

Before generating types, verify backend is accessible:

```bash
# In terminal 1: Start backend
cd article-mind-service
make dev

# In terminal 2: Test health endpoint
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

### Step 2: Generate TypeScript Types from OpenAPI

```bash
cd article-mind-ui
npm run gen:api
```

This creates `src/lib/api/generated.ts` with:
```typescript
export interface HealthResponse {
  status: "ok" | "degraded" | "error";
  version: string;
  database?: "connected" | "disconnected";
}
```

**Verify the file:**

```bash
cat src/lib/api/generated.ts | grep -A 5 "HealthResponse"
```

### Step 3: Create Health Check API Client

Create `src/lib/api/health.ts`:

```typescript
/**
 * Health check API client
 */

import { apiClient } from './client';
import type { components } from './generated';

// Type alias for better readability
type HealthResponse = components['schemas']['HealthResponse'];

/**
 * Check service health
 * @returns Promise resolving to health status
 */
export async function checkHealth(): Promise<HealthResponse> {
  return apiClient.get<HealthResponse>('/health');
}

/**
 * Get health status string for display
 */
export function getHealthStatusText(status: HealthResponse['status']): string {
  switch (status) {
    case 'ok':
      return 'Healthy';
    case 'degraded':
      return 'Degraded';
    case 'error':
      return 'Error';
    default:
      return 'Unknown';
  }
}

/**
 * Get health status color
 */
export function getHealthStatusColor(status: HealthResponse['status']): string {
  switch (status) {
    case 'ok':
      return 'green';
    case 'degraded':
      return 'yellow';
    case 'error':
      return 'red';
    default:
      return 'gray';
  }
}
```

### Step 4: Create HealthIndicator Component

Create `src/lib/components/HealthIndicator.svelte`:

```svelte
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { checkHealth, getHealthStatusText, getHealthStatusColor } from '$lib/api/health';
  import type { components } from '$lib/api/generated';

  type HealthResponse = components['schemas']['HealthResponse'];

  // Props
  export let pollInterval = 30000; // 30 seconds default

  // State using Svelte 5 Runes
  let health = $state<HealthResponse | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let intervalId = $state<number | null>(null);

  // Derived state
  let statusText = $derived(
    health ? getHealthStatusText(health.status) : 'Checking...'
  );
  let statusColor = $derived(
    health ? getHealthStatusColor(health.status) : 'gray'
  );
  let showDetails = $state(false);

  /**
   * Fetch health status from backend
   */
  async function fetchHealth() {
    try {
      loading = true;
      error = null;
      health = await checkHealth();
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to check health';
      console.error('Health check failed:', err);
    } finally {
      loading = false;
    }
  }

  /**
   * Start polling for health status
   */
  function startPolling() {
    fetchHealth(); // Initial fetch
    intervalId = window.setInterval(fetchHealth, pollInterval);
  }

  /**
   * Stop polling
   */
  function stopPolling() {
    if (intervalId !== null) {
      clearInterval(intervalId);
      intervalId = null;
    }
  }

  /**
   * Toggle details panel
   */
  function toggleDetails() {
    showDetails = !showDetails;
  }

  // Lifecycle
  onMount(() => {
    startPolling();
  });

  onDestroy(() => {
    stopPolling();
  });
</script>

<div class="health-indicator">
  <button
    class="health-status"
    onclick={toggleDetails}
    title="Click for details"
    aria-label="Health status: {statusText}"
  >
    <span class="status-dot" class:ok={statusColor === 'green'}
                            class:degraded={statusColor === 'yellow'}
                            class:error={statusColor === 'red'}
                            class:loading={loading}>
    </span>
    <span class="status-text">{statusText}</span>
  </button>

  {#if showDetails}
    <div class="health-details">
      {#if loading && !health}
        <p>Loading health status...</p>
      {:else if error}
        <p class="error-message">❌ {error}</p>
      {:else if health}
        <dl>
          <dt>Status:</dt>
          <dd class="status-{statusColor}">{health.status}</dd>

          <dt>Version:</dt>
          <dd>{health.version}</dd>

          {#if health.database}
            <dt>Database:</dt>
            <dd class:db-connected={health.database === 'connected'}
                class:db-disconnected={health.database === 'disconnected'}>
              {health.database}
            </dd>
          {/if}
        </dl>
      {/if}
    </div>
  {/if}
</div>

<style>
  .health-indicator {
    position: relative;
  }

  .health-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: transparent;
    border: 1px solid #ddd;
    border-radius: 4px;
    cursor: pointer;
    font-family: inherit;
    font-size: 0.875rem;
    transition: background-color 0.2s;
  }

  .health-status:hover {
    background-color: #f5f5f5;
  }

  .status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background-color: gray;
  }

  .status-dot.ok {
    background-color: #22c55e;
  }

  .status-dot.degraded {
    background-color: #eab308;
  }

  .status-dot.error {
    background-color: #ef4444;
  }

  .status-dot.loading {
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
  }

  .status-text {
    font-weight: 500;
  }

  .health-details {
    position: absolute;
    top: calc(100% + 0.5rem);
    right: 0;
    background: white;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 1rem;
    min-width: 200px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    z-index: 10;
  }

  .health-details dl {
    margin: 0;
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.5rem;
  }

  .health-details dt {
    font-weight: 600;
    color: #666;
  }

  .health-details dd {
    margin: 0;
    color: #333;
  }

  .status-green {
    color: #22c55e;
    font-weight: 600;
  }

  .status-yellow {
    color: #eab308;
    font-weight: 600;
  }

  .status-red {
    color: #ef4444;
    font-weight: 600;
  }

  .db-connected {
    color: #22c55e;
  }

  .db-disconnected {
    color: #ef4444;
  }

  .error-message {
    color: #ef4444;
    margin: 0;
  }
</style>
```

### Step 5: Add Health Indicator to Root Layout

Update `src/routes/+layout.svelte`:

```svelte
<script lang="ts">
  import HealthIndicator from '$lib/components/HealthIndicator.svelte';
</script>

<div class="app">
  <header>
    <div class="container">
      <h1 class="logo">Article Mind</h1>
      <nav>
        <HealthIndicator pollInterval={30000} />
      </nav>
    </div>
  </header>

  <main>
    <slot />
  </main>
</div>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    font-family: system-ui, -apple-system, sans-serif;
  }

  .app {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  header {
    background: #fff;
    border-bottom: 1px solid #e5e7eb;
    padding: 1rem 0;
  }

  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .logo {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
    color: #1f2937;
  }

  nav {
    display: flex;
    align-items: center;
  }

  main {
    flex: 1;
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 1rem;
    width: 100%;
  }
</style>
```

### Step 6: Update Home Page for Testing

Update `src/routes/+page.svelte`:

```svelte
<script lang="ts">
  let appName = $state('Article Mind');
</script>

<div class="container">
  <h1>Welcome to {appName}</h1>
  <p>SvelteKit frontend is running!</p>

  <section class="info">
    <h2>Health Status</h2>
    <p>The health indicator in the header shows the current status of the backend service.</p>
    <ul>
      <li><strong>Green (Healthy):</strong> All systems operational</li>
      <li><strong>Yellow (Degraded):</strong> Service running but database unavailable</li>
      <li><strong>Red (Error):</strong> Critical failure</li>
    </ul>
    <p>Click the health indicator to see detailed information.</p>
  </section>
</div>

<style>
  .container {
    text-align: center;
  }

  h1 {
    color: #333;
    margin-bottom: 1rem;
  }

  .info {
    margin-top: 2rem;
    text-align: left;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
    padding: 1.5rem;
    background: #f9fafb;
    border-radius: 8px;
  }

  .info h2 {
    margin-top: 0;
    color: #1f2937;
  }

  .info ul {
    line-height: 1.8;
  }
</style>
```

### Step 7: Write Component Tests

Create `tests/unit/HealthIndicator.test.ts`:

```typescript
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import HealthIndicator from '$lib/components/HealthIndicator.svelte';
import { checkHealth } from '$lib/api/health';

// Mock the health API
vi.mock('$lib/api/health', () => ({
  checkHealth: vi.fn(),
  getHealthStatusText: (status: string) => {
    switch (status) {
      case 'ok': return 'Healthy';
      case 'degraded': return 'Degraded';
      case 'error': return 'Error';
      default: return 'Unknown';
    }
  },
  getHealthStatusColor: (status: string) => {
    switch (status) {
      case 'ok': return 'green';
      case 'degraded': return 'yellow';
      case 'error': return 'red';
      default: return 'gray';
    }
  },
}));

describe('HealthIndicator', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders loading state initially', () => {
    vi.mocked(checkHealth).mockResolvedValue({
      status: 'ok',
      version: '1.0.0',
      database: 'connected',
    });

    render(HealthIndicator);

    expect(screen.getByText('Checking...')).toBeDefined();
  });

  it('displays healthy status when service is ok', async () => {
    vi.mocked(checkHealth).mockResolvedValue({
      status: 'ok',
      version: '1.0.0',
      database: 'connected',
    });

    render(HealthIndicator);

    // Wait for async fetch
    await new Promise(resolve => setTimeout(resolve, 100));

    expect(screen.getByText('Healthy')).toBeDefined();
  });

  it('displays degraded status when database is down', async () => {
    vi.mocked(checkHealth).mockResolvedValue({
      status: 'degraded',
      version: '1.0.0',
      database: 'disconnected',
    });

    render(HealthIndicator);

    await new Promise(resolve => setTimeout(resolve, 100));

    expect(screen.getByText('Degraded')).toBeDefined();
  });

  it('handles API errors gracefully', async () => {
    vi.mocked(checkHealth).mockRejectedValue(new Error('Network error'));

    render(HealthIndicator);

    await new Promise(resolve => setTimeout(resolve, 100));

    // Should still render, possibly with error state
    expect(screen.getByRole('button')).toBeDefined();
  });
});
```

### Step 8: Update package.json Test Script

Ensure `package.json` has proper test configuration:

```json
{
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest",
    "test:ui": "vitest --ui"
  }
}
```

### Step 9: Add Environment Variable Documentation

Update `article-mind-ui/.env.example`:

```env
# API Configuration
VITE_API_BASE_URL=http://localhost:8000

# Health Check Configuration (optional, uses defaults if not set)
VITE_HEALTH_POLL_INTERVAL=30000
```

---

## 5. CLAUDE.md Requirements

Updated in Plan 01's CLAUDE.md (article-mind-ui):

- ✅ Type generation workflow (`npm run gen:api`)
- ✅ Component testing with Vitest
- ✅ Svelte 5 Runes API usage examples
- ✅ API client usage patterns

Add to existing CLAUDE.md:

```markdown
## Health Indicator Component

The `HealthIndicator` component in the header polls the backend `/health` endpoint every 30 seconds.

### Usage

```svelte
<script>
  import HealthIndicator from '$lib/components/HealthIndicator.svelte';
</script>

<HealthIndicator pollInterval={30000} />
```

### Props

- `pollInterval`: Milliseconds between health checks (default: 30000)

### Behavior

- **Green dot:** Service healthy, database connected
- **Yellow dot:** Service degraded (database disconnected)
- **Red dot:** Service error
- **Gray dot:** Loading state

Click indicator to see detailed status including version and database status.

### Testing

Component uses Svelte 5 Runes for state management:

```svelte
let health = $state<HealthResponse | null>(null);
let loading = $state(true);
let statusText = $derived(health ? getHealthStatusText(health.status) : 'Checking...');
```

Test with Vitest:

```bash
make test
```
```

---

## 6. Verification Steps

### 1. Verify Type Generation

```bash
cd article-mind-ui
npm run gen:api
cat src/lib/api/generated.ts | grep "HealthResponse"
```

**Expected:** TypeScript interface for HealthResponse exists.

### 2. Start Frontend Dev Server

```bash
make dev
```

**Expected:** Server starts on http://localhost:5173

### 3. Verify Health Indicator Renders

Visit: http://localhost:5173

**Expected:**
- Header shows "Article Mind" logo
- Health indicator on right side
- Status shows "Healthy" with green dot (if backend is running)

### 4. Test Health Indicator Click

Click health indicator button.

**Expected:**
- Dropdown panel appears
- Shows status, version, database status
- Clicking again closes panel

### 5. Test Backend Failure Scenario

Stop backend:

```bash
# In backend terminal
Ctrl+C
```

Wait 30 seconds (poll interval).

**Expected:**
- Health indicator changes to red
- Status shows "Error"
- Details panel shows error message

Restart backend:

```bash
cd article-mind-service
make dev
```

Wait 30 seconds.

**Expected:**
- Health indicator changes back to green
- Status shows "Healthy"

### 6. Test Database Failure Scenario

Stop database (keep backend running):

```bash
docker stop article-mind-postgres
```

Wait 30 seconds.

**Expected:**
- Health indicator changes to yellow
- Status shows "Degraded"
- Database status shows "disconnected"

Restart database:

```bash
docker start article-mind-postgres
```

### 7. Run Component Tests

```bash
make test
```

**Expected:** All tests pass, including HealthIndicator tests.

### 8. Verify CORS

Open browser console (F12) while viewing UI.

**Expected:** No CORS errors when health check runs.

---

## 7. Common Pitfalls

### Issue 1: Type Generation Fails

**Symptom:** `npm run gen:api` errors

**Solution:**
- Ensure backend is running: `curl http://localhost:8000/health`
- Verify OpenAPI endpoint: `curl http://localhost:8000/openapi.json`
- Check `openapi-typescript` is installed: `npm list openapi-typescript`

### Issue 2: CORS Errors in Browser

**Symptom:** Console shows CORS policy errors

**Solution:**
- Verify backend CORS middleware is configured
- Check `.env` in backend has `CORS_ORIGINS=http://localhost:5173`
- Restart backend after changing CORS config

### Issue 3: Health Indicator Stuck on "Checking..."

**Symptom:** Status never updates from loading state

**Solution:**
- Open browser console, check for errors
- Verify API base URL: `console.log(import.meta.env.VITE_API_BASE_URL)`
- Test health endpoint manually: `curl http://localhost:8000/health`
- Check network tab for failed requests

### Issue 4: Component Tests Fail

**Symptom:** `make test` shows errors in HealthIndicator tests

**Solution:**
- Ensure mocks are properly configured
- Check async behavior in tests (use proper awaits)
- Verify @testing-library/svelte is installed

### Issue 5: Details Panel Doesn't Close

**Symptom:** Clicking outside details panel doesn't close it

**Solution:**
This is expected behavior - only clicking the button toggles the panel. To add click-outside behavior:

```svelte
<script>
  function handleClickOutside(event: MouseEvent) {
    if (showDetails && !event.target.closest('.health-indicator')) {
      showDetails = false;
    }
  }

  onMount(() => {
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  });
</script>
```

### Issue 6: Polling Continues After Navigation

**Symptom:** Multiple health checks running simultaneously

**Solution:**
Component properly cleans up with `onDestroy` - verify lifecycle hooks:

```svelte
onDestroy(() => {
  stopPolling();
});
```

---

## 8. Next Steps

After completing this plan:

1. **Add more UI features** using the same pattern (API client + components + tests)
2. **Implement error boundary** for graceful error handling
3. **Add loading states** for better UX
4. **Create more reusable components** in `src/lib/components/`
5. **Set up E2E tests** with Playwright (deferred from Plan 01)

---

## 9. Success Criteria

- ✅ TypeScript types generated from backend OpenAPI spec
- ✅ Health indicator visible in header on all pages
- ✅ Status updates every 30 seconds automatically
- ✅ Color-coded status (green/yellow/red) displays correctly
- ✅ Details panel shows version and database status
- ✅ Component tests pass
- ✅ No CORS errors in browser console
- ✅ Service status reflects backend health accurately
- ✅ Handles backend unavailability gracefully
- ✅ Click interaction works (toggle details panel)

---

## 10. Optional Enhancements (Future)

While not required for this plan, consider these improvements:

### Visual Enhancements
- Add icons for different statuses
- Animate status transitions
- Add tooltips with last check timestamp
- Show historical status trend

### Functional Enhancements
- Manual refresh button
- Configurable poll interval via UI
- Notification when status changes
- Link to backend /docs from details panel
- Show response time/latency

### Accessibility
- Keyboard navigation for details panel
- Screen reader announcements on status change
- High contrast mode support
- Focus management

### Performance
- Debounce rapid status changes
- Reduce poll frequency when tab is not visible
- Cancel in-flight requests on unmount

---

**Plan Status:** Ready for implementation
**Last Updated:** 2026-01-18
