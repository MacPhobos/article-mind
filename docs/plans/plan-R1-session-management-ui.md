# Plan R1: Session Management UI

**Plan ID:** R1-session-management-ui
**Created:** 2026-01-19
**Dependencies:** Plan 01 (scaffold-ui), Plan 05 (ui-health-display), Plan R2 (session-crud-api)
**Estimated Time:** 4-6 hours

---

## 1. Overview

### Purpose

Implement the Session Management UI for the Article Mind frontend, providing users with a complete interface to create, view, update, delete, and archive research sessions.

### Scope

- Top navigation bar with "Sessions" menu item
- Default landing page showing list of research sessions
- CRUD operations (create, rename, delete, archive)
- Session cards displaying key information
- Click-to-open navigation to session detail page
- State management using Svelte 5 Runes API
- Integration with backend API via generated TypeScript types

### Architecture Decisions Applied

- **Session Lifecycle:** Full lifecycle (draft -> active -> completed -> archived) with soft delete
- **Auth Model:** Single-user (no authentication required)
- **File Storage:** PostgreSQL + filesystem (backend handles this)

### Dependencies

- **Plan 01:** Requires SvelteKit scaffolding
- **Plan 05:** Builds upon health check display patterns
- **Plan R2:** Requires Session CRUD API endpoints

### Outputs

- Working session list page at `/` (default landing)
- Session management components
- Navigation with Sessions menu
- Full CRUD functionality
- Generated TypeScript types from backend OpenAPI

---

## 2. Technology Stack

- **Framework:** SvelteKit 2.x with Svelte 5 (Runes API)
- **Language:** TypeScript 5.x (strict mode)
- **State Management:** Svelte 5 $state/$derived/$effect
- **API Integration:** Generated types from OpenAPI (`src/lib/api/generated.ts`)
- **Styling:** CSS (following existing patterns)
- **Testing:** Vitest + @testing-library/svelte

---

## 3. Directory Structure

```
article-mind-ui/
├── src/
│   ├── lib/
│   │   ├── api/
│   │   │   ├── client.ts           # Existing API client
│   │   │   └── generated.ts        # UPDATED: Add session types
│   │   ├── components/
│   │   │   ├── SessionCard.svelte     # NEW: Session card component
│   │   │   ├── CreateSessionModal.svelte  # NEW: Create/edit modal
│   │   │   ├── DeleteSessionModal.svelte  # NEW: Delete confirmation
│   │   │   └── TopBar.svelte          # NEW: Navigation bar
│   │   └── stores/
│   │       └── sessions.svelte.ts     # NEW: Session state management
│   └── routes/
│       ├── +layout.svelte          # UPDATED: Add TopBar
│       ├── +page.svelte            # UPDATED: Session list (landing)
│       └── sessions/
│           └── [id]/
│               └── +page.svelte    # NEW: Session detail page
└── tests/
    └── unit/
        ├── SessionCard.test.ts     # NEW
        ├── CreateSessionModal.test.ts  # NEW
        └── sessions.store.test.ts  # NEW
```

---

## 4. UI Components

### 4.1 TopBar Component

**File:** `src/lib/components/TopBar.svelte`

```svelte
<script lang="ts">
	import { page } from '$app/stores';

	interface Props {
		appName?: string;
	}

	let { appName = 'Article Mind' }: Props = $props();

	// Determine active nav item based on current route
	let currentPath = $derived($page.url.pathname);
	let isSessionsActive = $derived(currentPath === '/' || currentPath.startsWith('/sessions'));
</script>

<nav class="top-bar">
	<div class="logo">
		<a href="/">{appName}</a>
	</div>
	<ul class="nav-menu">
		<li class:active={isSessionsActive}>
			<a href="/">Sessions</a>
		</li>
		<!-- Future nav items: Articles, Search, Settings -->
	</ul>
</nav>

<style>
	.top-bar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0 2rem;
		height: 60px;
		background: #1a1a2e;
		color: white;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
	}

	.logo a {
		font-size: 1.5rem;
		font-weight: bold;
		color: white;
		text-decoration: none;
	}

	.nav-menu {
		display: flex;
		gap: 2rem;
		list-style: none;
		margin: 0;
		padding: 0;
	}

	.nav-menu li a {
		color: rgba(255, 255, 255, 0.8);
		text-decoration: none;
		padding: 0.5rem 1rem;
		border-radius: 4px;
		transition: all 0.2s ease;
	}

	.nav-menu li a:hover {
		color: white;
		background: rgba(255, 255, 255, 0.1);
	}

	.nav-menu li.active a {
		color: white;
		background: rgba(255, 255, 255, 0.15);
	}
</style>
```

### 4.2 SessionCard Component

**File:** `src/lib/components/SessionCard.svelte`

```svelte
<script lang="ts">
	import type { SessionResponse } from '$lib/api/generated';

	interface Props {
		session: SessionResponse;
		onEdit?: (session: SessionResponse) => void;
		onDelete?: (session: SessionResponse) => void;
		onArchive?: (session: SessionResponse) => void;
	}

	let { session, onEdit, onDelete, onArchive }: Props = $props();

	// Format dates for display
	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleDateString('en-US', {
			year: 'numeric',
			month: 'short',
			day: 'numeric'
		});
	}

	// Status badge color mapping
	let statusColor = $derived(() => {
		switch (session.status) {
			case 'draft':
				return 'badge-draft';
			case 'active':
				return 'badge-active';
			case 'completed':
				return 'badge-completed';
			case 'archived':
				return 'badge-archived';
			default:
				return 'badge-draft';
		}
	});
</script>

<article class="session-card">
	<a href="/sessions/{session.id}" class="card-link">
		<header class="card-header">
			<h3 class="session-name">{session.name}</h3>
			<span class="status-badge {statusColor()}">{session.status}</span>
		</header>

		{#if session.description}
			<p class="session-description">{session.description}</p>
		{/if}

		<footer class="card-footer">
			<div class="meta-info">
				<span class="article-count">{session.article_count ?? 0} articles</span>
				<span class="date-info">Created {formatDate(session.created_at)}</span>
				{#if session.updated_at !== session.created_at}
					<span class="date-info">Updated {formatDate(session.updated_at)}</span>
				{/if}
			</div>
		</footer>
	</a>

	<div class="card-actions">
		{#if onEdit && session.status !== 'archived'}
			<button class="btn-action" onclick={() => onEdit?.(session)} aria-label="Edit session">
				Edit
			</button>
		{/if}
		{#if onArchive && session.status !== 'archived'}
			<button
				class="btn-action btn-archive"
				onclick={() => onArchive?.(session)}
				aria-label="Archive session"
			>
				Archive
			</button>
		{/if}
		{#if onDelete}
			<button
				class="btn-action btn-delete"
				onclick={() => onDelete?.(session)}
				aria-label="Delete session"
			>
				Delete
			</button>
		{/if}
	</div>
</article>

<style>
	.session-card {
		background: white;
		border: 1px solid #e0e0e0;
		border-radius: 8px;
		padding: 1.5rem;
		transition:
			transform 0.2s ease,
			box-shadow 0.2s ease;
	}

	.session-card:hover {
		transform: translateY(-2px);
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
	}

	.card-link {
		text-decoration: none;
		color: inherit;
		display: block;
	}

	.card-header {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		margin-bottom: 0.75rem;
	}

	.session-name {
		margin: 0;
		font-size: 1.25rem;
		color: #333;
	}

	.status-badge {
		font-size: 0.75rem;
		padding: 0.25rem 0.75rem;
		border-radius: 12px;
		text-transform: uppercase;
		font-weight: 600;
	}

	.badge-draft {
		background: #f0f0f0;
		color: #666;
	}
	.badge-active {
		background: #e3f2fd;
		color: #1976d2;
	}
	.badge-completed {
		background: #e8f5e9;
		color: #388e3c;
	}
	.badge-archived {
		background: #fafafa;
		color: #9e9e9e;
	}

	.session-description {
		color: #666;
		font-size: 0.9rem;
		margin: 0 0 1rem 0;
		line-height: 1.5;
	}

	.card-footer {
		border-top: 1px solid #f0f0f0;
		padding-top: 0.75rem;
		margin-top: 0.75rem;
	}

	.meta-info {
		display: flex;
		flex-wrap: wrap;
		gap: 1rem;
		font-size: 0.8rem;
		color: #888;
	}

	.article-count {
		font-weight: 500;
		color: #666;
	}

	.card-actions {
		display: flex;
		gap: 0.5rem;
		margin-top: 1rem;
		padding-top: 0.75rem;
		border-top: 1px solid #f0f0f0;
	}

	.btn-action {
		padding: 0.375rem 0.75rem;
		font-size: 0.8rem;
		border: 1px solid #ddd;
		background: white;
		border-radius: 4px;
		cursor: pointer;
		transition: all 0.2s ease;
	}

	.btn-action:hover {
		background: #f5f5f5;
	}

	.btn-archive:hover {
		background: #fff3e0;
		border-color: #ffb74d;
		color: #f57c00;
	}

	.btn-delete:hover {
		background: #ffebee;
		border-color: #ef5350;
		color: #d32f2f;
	}
</style>
```

### 4.3 CreateSessionModal Component

**File:** `src/lib/components/CreateSessionModal.svelte`

```svelte
<script lang="ts">
	import type { SessionResponse, CreateSessionRequest } from '$lib/api/generated';

	interface Props {
		isOpen: boolean;
		session?: SessionResponse | null; // If provided, we're editing
		onClose: () => void;
		onSubmit: (data: CreateSessionRequest) => Promise<void>;
	}

	let { isOpen, session = null, onClose, onSubmit }: Props = $props();

	// Form state
	let name = $state(session?.name ?? '');
	let description = $state(session?.description ?? '');
	let isSubmitting = $state(false);
	let error = $state<string | null>(null);

	// Reset form when modal opens/closes or session changes
	$effect(() => {
		if (isOpen) {
			name = session?.name ?? '';
			description = session?.description ?? '';
			error = null;
		}
	});

	let isEditing = $derived(session !== null);
	let modalTitle = $derived(isEditing ? 'Edit Session' : 'Create New Session');
	let submitLabel = $derived(isEditing ? 'Save Changes' : 'Create Session');

	async function handleSubmit(event: Event) {
		event.preventDefault();

		if (!name.trim()) {
			error = 'Session name is required';
			return;
		}

		isSubmitting = true;
		error = null;

		try {
			await onSubmit({
				name: name.trim(),
				description: description.trim() || undefined
			});
			onClose();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to save session';
		} finally {
			isSubmitting = false;
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			onClose();
		}
	}
</script>

{#if isOpen}
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div class="modal-overlay" role="dialog" aria-modal="true" onkeydown={handleKeydown}>
		<div class="modal-content">
			<header class="modal-header">
				<h2>{modalTitle}</h2>
				<button class="close-btn" onclick={onClose} aria-label="Close modal">
					&times;
				</button>
			</header>

			<form onsubmit={handleSubmit}>
				{#if error}
					<div class="error-message" role="alert">{error}</div>
				{/if}

				<div class="form-field">
					<label for="session-name">Name *</label>
					<input
						type="text"
						id="session-name"
						bind:value={name}
						placeholder="Enter session name"
						disabled={isSubmitting}
						required
					/>
				</div>

				<div class="form-field">
					<label for="session-description">Description</label>
					<textarea
						id="session-description"
						bind:value={description}
						placeholder="Optional description for this session"
						disabled={isSubmitting}
						rows="3"
					></textarea>
				</div>

				<footer class="modal-footer">
					<button type="button" class="btn-cancel" onclick={onClose} disabled={isSubmitting}>
						Cancel
					</button>
					<button type="submit" class="btn-submit" disabled={isSubmitting}>
						{#if isSubmitting}
							Saving...
						{:else}
							{submitLabel}
						{/if}
					</button>
				</footer>
			</form>
		</div>
	</div>
{/if}

<style>
	.modal-overlay {
		position: fixed;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
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

	.modal-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 1.25rem 1.5rem;
		border-bottom: 1px solid #e0e0e0;
	}

	.modal-header h2 {
		margin: 0;
		font-size: 1.25rem;
		color: #333;
	}

	.close-btn {
		background: none;
		border: none;
		font-size: 1.5rem;
		color: #666;
		cursor: pointer;
		padding: 0.25rem;
		line-height: 1;
	}

	.close-btn:hover {
		color: #333;
	}

	form {
		padding: 1.5rem;
	}

	.error-message {
		background: #ffebee;
		color: #c62828;
		padding: 0.75rem 1rem;
		border-radius: 4px;
		margin-bottom: 1rem;
		font-size: 0.9rem;
	}

	.form-field {
		margin-bottom: 1.25rem;
	}

	.form-field label {
		display: block;
		margin-bottom: 0.5rem;
		font-weight: 500;
		color: #333;
	}

	.form-field input,
	.form-field textarea {
		width: 100%;
		padding: 0.75rem;
		border: 1px solid #ddd;
		border-radius: 4px;
		font-size: 1rem;
		font-family: inherit;
		box-sizing: border-box;
	}

	.form-field input:focus,
	.form-field textarea:focus {
		outline: none;
		border-color: #1976d2;
		box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.1);
	}

	.form-field input:disabled,
	.form-field textarea:disabled {
		background: #f5f5f5;
	}

	.modal-footer {
		display: flex;
		justify-content: flex-end;
		gap: 0.75rem;
		padding-top: 1rem;
		border-top: 1px solid #e0e0e0;
		margin-top: 0.5rem;
	}

	.btn-cancel,
	.btn-submit {
		padding: 0.625rem 1.25rem;
		border-radius: 4px;
		font-size: 0.9rem;
		font-weight: 500;
		cursor: pointer;
		transition: all 0.2s ease;
	}

	.btn-cancel {
		background: white;
		border: 1px solid #ddd;
		color: #666;
	}

	.btn-cancel:hover:not(:disabled) {
		background: #f5f5f5;
	}

	.btn-submit {
		background: #1976d2;
		border: 1px solid #1976d2;
		color: white;
	}

	.btn-submit:hover:not(:disabled) {
		background: #1565c0;
	}

	.btn-cancel:disabled,
	.btn-submit:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}
</style>
```

### 4.4 DeleteSessionModal Component

**File:** `src/lib/components/DeleteSessionModal.svelte`

```svelte
<script lang="ts">
	import type { SessionResponse } from '$lib/api/generated';

	interface Props {
		isOpen: boolean;
		session: SessionResponse | null;
		onClose: () => void;
		onConfirm: () => Promise<void>;
	}

	let { isOpen, session, onClose, onConfirm }: Props = $props();

	let isDeleting = $state(false);
	let error = $state<string | null>(null);

	async function handleDelete() {
		isDeleting = true;
		error = null;

		try {
			await onConfirm();
			onClose();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to delete session';
		} finally {
			isDeleting = false;
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			onClose();
		}
	}
</script>

{#if isOpen && session}
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div class="modal-overlay" role="dialog" aria-modal="true" onkeydown={handleKeydown}>
		<div class="modal-content">
			<header class="modal-header">
				<h2>Delete Session</h2>
				<button class="close-btn" onclick={onClose} aria-label="Close modal">
					&times;
				</button>
			</header>

			<div class="modal-body">
				{#if error}
					<div class="error-message" role="alert">{error}</div>
				{/if}

				<p class="warning-text">
					Are you sure you want to delete <strong>"{session.name}"</strong>?
				</p>
				<p class="info-text">
					This action will soft-delete the session. The session can be recovered by an administrator.
				</p>
			</div>

			<footer class="modal-footer">
				<button class="btn-cancel" onclick={onClose} disabled={isDeleting}>
					Cancel
				</button>
				<button class="btn-delete" onclick={handleDelete} disabled={isDeleting}>
					{#if isDeleting}
						Deleting...
					{:else}
						Delete Session
					{/if}
				</button>
			</footer>
		</div>
	</div>
{/if}

<style>
	.modal-overlay {
		position: fixed;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
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
		max-width: 450px;
		margin: 1rem;
		box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
	}

	.modal-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 1.25rem 1.5rem;
		border-bottom: 1px solid #e0e0e0;
	}

	.modal-header h2 {
		margin: 0;
		font-size: 1.25rem;
		color: #d32f2f;
	}

	.close-btn {
		background: none;
		border: none;
		font-size: 1.5rem;
		color: #666;
		cursor: pointer;
		padding: 0.25rem;
		line-height: 1;
	}

	.modal-body {
		padding: 1.5rem;
	}

	.error-message {
		background: #ffebee;
		color: #c62828;
		padding: 0.75rem 1rem;
		border-radius: 4px;
		margin-bottom: 1rem;
		font-size: 0.9rem;
	}

	.warning-text {
		margin: 0 0 0.75rem 0;
		font-size: 1rem;
		color: #333;
	}

	.info-text {
		margin: 0;
		font-size: 0.9rem;
		color: #666;
	}

	.modal-footer {
		display: flex;
		justify-content: flex-end;
		gap: 0.75rem;
		padding: 1rem 1.5rem;
		border-top: 1px solid #e0e0e0;
	}

	.btn-cancel,
	.btn-delete {
		padding: 0.625rem 1.25rem;
		border-radius: 4px;
		font-size: 0.9rem;
		font-weight: 500;
		cursor: pointer;
		transition: all 0.2s ease;
	}

	.btn-cancel {
		background: white;
		border: 1px solid #ddd;
		color: #666;
	}

	.btn-cancel:hover:not(:disabled) {
		background: #f5f5f5;
	}

	.btn-delete {
		background: #d32f2f;
		border: 1px solid #d32f2f;
		color: white;
	}

	.btn-delete:hover:not(:disabled) {
		background: #c62828;
	}

	.btn-cancel:disabled,
	.btn-delete:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}
</style>
```

---

## 5. Routes

### 5.1 Updated Layout

**File:** `src/routes/+layout.svelte`

```svelte
<script lang="ts">
	import TopBar from '$lib/components/TopBar.svelte';

	let { children } = $props();
</script>

<TopBar />
<main>
	{@render children()}
</main>

<style>
	:global(body) {
		margin: 0;
		padding: 0;
		font-family:
			system-ui,
			-apple-system,
			sans-serif;
		background: #f8f9fa;
	}

	main {
		min-height: calc(100vh - 60px);
	}
</style>
```

### 5.2 Session List Page (Landing)

**File:** `src/routes/+page.svelte`

```svelte
<script lang="ts">
	import { apiClient } from '$lib/api/client';
	import type {
		SessionResponse,
		SessionListResponse,
		CreateSessionRequest
	} from '$lib/api/generated';
	import SessionCard from '$lib/components/SessionCard.svelte';
	import CreateSessionModal from '$lib/components/CreateSessionModal.svelte';
	import DeleteSessionModal from '$lib/components/DeleteSessionModal.svelte';

	// State
	let sessions = $state<SessionResponse[]>([]);
	let isLoading = $state(true);
	let error = $state<string | null>(null);

	// Modal state
	let showCreateModal = $state(false);
	let showDeleteModal = $state(false);
	let editingSession = $state<SessionResponse | null>(null);
	let deletingSession = $state<SessionResponse | null>(null);

	// Filter state
	let statusFilter = $state<string>('all');

	// Filtered sessions
	let filteredSessions = $derived(() => {
		if (statusFilter === 'all') {
			return sessions;
		}
		return sessions.filter((s) => s.status === statusFilter);
	});

	// Load sessions on mount
	$effect(() => {
		loadSessions();
	});

	async function loadSessions() {
		isLoading = true;
		error = null;

		try {
			const response = await apiClient.get<SessionListResponse>('/api/v1/sessions');
			sessions = response.sessions;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load sessions';
			console.error('Failed to load sessions:', e);
		} finally {
			isLoading = false;
		}
	}

	async function handleCreateSession(data: CreateSessionRequest) {
		if (editingSession) {
			// Update existing session
			await apiClient.patch(`/api/v1/sessions/${editingSession.id}`, data);
		} else {
			// Create new session
			await apiClient.post('/api/v1/sessions', data);
		}
		await loadSessions();
	}

	async function handleDeleteSession() {
		if (!deletingSession) return;
		await apiClient.delete(`/api/v1/sessions/${deletingSession.id}`);
		await loadSessions();
	}

	async function handleArchiveSession(session: SessionResponse) {
		await apiClient.post(`/api/v1/sessions/${session.id}/status`, {
			status: 'archived'
		});
		await loadSessions();
	}

	function openCreateModal() {
		editingSession = null;
		showCreateModal = true;
	}

	function openEditModal(session: SessionResponse) {
		editingSession = session;
		showCreateModal = true;
	}

	function openDeleteModal(session: SessionResponse) {
		deletingSession = session;
		showDeleteModal = true;
	}

	function closeCreateModal() {
		showCreateModal = false;
		editingSession = null;
	}

	function closeDeleteModal() {
		showDeleteModal = false;
		deletingSession = null;
	}
</script>

<div class="page-container">
	<header class="page-header">
		<h1>Research Sessions</h1>
		<button class="btn-create" onclick={openCreateModal}>
			+ New Session
		</button>
	</header>

	<div class="filters">
		<label for="status-filter">Filter by status:</label>
		<select id="status-filter" bind:value={statusFilter}>
			<option value="all">All</option>
			<option value="draft">Draft</option>
			<option value="active">Active</option>
			<option value="completed">Completed</option>
			<option value="archived">Archived</option>
		</select>
	</div>

	{#if isLoading}
		<div class="loading">Loading sessions...</div>
	{:else if error}
		<div class="error" role="alert">
			<p>{error}</p>
			<button onclick={loadSessions}>Retry</button>
		</div>
	{:else if filteredSessions().length === 0}
		<div class="empty-state">
			{#if sessions.length === 0}
				<p>No research sessions yet.</p>
				<p>Create your first session to start organizing articles.</p>
				<button class="btn-create" onclick={openCreateModal}>
					Create Session
				</button>
			{:else}
				<p>No sessions match the selected filter.</p>
			{/if}
		</div>
	{:else}
		<div class="sessions-grid">
			{#each filteredSessions() as session (session.id)}
				<SessionCard
					{session}
					onEdit={openEditModal}
					onDelete={openDeleteModal}
					onArchive={handleArchiveSession}
				/>
			{/each}
		</div>
	{/if}
</div>

<CreateSessionModal
	isOpen={showCreateModal}
	session={editingSession}
	onClose={closeCreateModal}
	onSubmit={handleCreateSession}
/>

<DeleteSessionModal
	isOpen={showDeleteModal}
	session={deletingSession}
	onClose={closeDeleteModal}
	onConfirm={handleDeleteSession}
/>

<style>
	.page-container {
		max-width: 1200px;
		margin: 0 auto;
		padding: 2rem;
	}

	.page-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 2rem;
	}

	.page-header h1 {
		margin: 0;
		font-size: 2rem;
		color: #333;
	}

	.btn-create {
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

	.btn-create:hover {
		background: #1565c0;
	}

	.filters {
		margin-bottom: 1.5rem;
		display: flex;
		align-items: center;
		gap: 0.75rem;
	}

	.filters label {
		font-size: 0.9rem;
		color: #666;
	}

	.filters select {
		padding: 0.5rem 1rem;
		border: 1px solid #ddd;
		border-radius: 4px;
		font-size: 0.9rem;
		background: white;
	}

	.loading {
		text-align: center;
		padding: 3rem;
		color: #666;
	}

	.error {
		text-align: center;
		padding: 2rem;
		background: #ffebee;
		border-radius: 8px;
		color: #c62828;
	}

	.error button {
		margin-top: 1rem;
		padding: 0.5rem 1rem;
		background: #d32f2f;
		color: white;
		border: none;
		border-radius: 4px;
		cursor: pointer;
	}

	.empty-state {
		text-align: center;
		padding: 4rem 2rem;
		background: white;
		border: 2px dashed #ddd;
		border-radius: 8px;
	}

	.empty-state p {
		margin: 0.5rem 0;
		color: #666;
	}

	.empty-state .btn-create {
		margin-top: 1rem;
	}

	.sessions-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
		gap: 1.5rem;
	}
</style>
```

### 5.3 Session Detail Page

**File:** `src/routes/sessions/[id]/+page.svelte`

```svelte
<script lang="ts">
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { apiClient } from '$lib/api/client';
	import type { SessionResponse } from '$lib/api/generated';

	let sessionId = $derived($page.params.id);
	let session = $state<SessionResponse | null>(null);
	let isLoading = $state(true);
	let error = $state<string | null>(null);

	// Load session when ID changes
	$effect(() => {
		loadSession(sessionId);
	});

	async function loadSession(id: string) {
		isLoading = true;
		error = null;

		try {
			session = await apiClient.get<SessionResponse>(`/api/v1/sessions/${id}`);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load session';
			console.error('Failed to load session:', e);
		} finally {
			isLoading = false;
		}
	}

	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleDateString('en-US', {
			year: 'numeric',
			month: 'long',
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit'
		});
	}

	async function changeStatus(newStatus: string) {
		if (!session) return;

		try {
			await apiClient.post(`/api/v1/sessions/${session.id}/status`, {
				status: newStatus
			});
			await loadSession(sessionId);
		} catch (e) {
			console.error('Failed to change status:', e);
		}
	}
</script>

<div class="page-container">
	<nav class="breadcrumb">
		<a href="/">Sessions</a>
		<span>/</span>
		<span>{session?.name ?? 'Loading...'}</span>
	</nav>

	{#if isLoading}
		<div class="loading">Loading session...</div>
	{:else if error}
		<div class="error" role="alert">
			<p>{error}</p>
			<button onclick={() => goto('/')}>Back to Sessions</button>
		</div>
	{:else if session}
		<header class="session-header">
			<div class="header-info">
				<h1>{session.name}</h1>
				<span class="status-badge status-{session.status}">{session.status}</span>
			</div>

			{#if session.description}
				<p class="description">{session.description}</p>
			{/if}

			<div class="meta">
				<span>Created: {formatDate(session.created_at)}</span>
				<span>Updated: {formatDate(session.updated_at)}</span>
				<span>{session.article_count ?? 0} articles</span>
			</div>
		</header>

		<section class="status-actions">
			<h2>Status Actions</h2>
			<div class="action-buttons">
				{#if session.status === 'draft'}
					<button onclick={() => changeStatus('active')}>Start Session</button>
				{/if}
				{#if session.status === 'active'}
					<button onclick={() => changeStatus('completed')}>Mark Complete</button>
				{/if}
				{#if session.status !== 'archived'}
					<button class="btn-archive" onclick={() => changeStatus('archived')}>
						Archive Session
					</button>
				{/if}
			</div>
		</section>

		<section class="articles-section">
			<h2>Articles</h2>
			<p class="placeholder">
				Article management will be implemented in a future plan.
			</p>
		</section>
	{/if}
</div>

<style>
	.page-container {
		max-width: 1000px;
		margin: 0 auto;
		padding: 2rem;
	}

	.breadcrumb {
		margin-bottom: 1.5rem;
		font-size: 0.9rem;
		color: #666;
	}

	.breadcrumb a {
		color: #1976d2;
		text-decoration: none;
	}

	.breadcrumb a:hover {
		text-decoration: underline;
	}

	.breadcrumb span {
		margin: 0 0.5rem;
	}

	.loading {
		text-align: center;
		padding: 3rem;
		color: #666;
	}

	.error {
		text-align: center;
		padding: 2rem;
		background: #ffebee;
		border-radius: 8px;
		color: #c62828;
	}

	.error button {
		margin-top: 1rem;
		padding: 0.5rem 1rem;
		background: #d32f2f;
		color: white;
		border: none;
		border-radius: 4px;
		cursor: pointer;
	}

	.session-header {
		background: white;
		padding: 2rem;
		border-radius: 8px;
		margin-bottom: 2rem;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
	}

	.header-info {
		display: flex;
		align-items: center;
		gap: 1rem;
		margin-bottom: 1rem;
	}

	.header-info h1 {
		margin: 0;
		font-size: 1.75rem;
		color: #333;
	}

	.status-badge {
		padding: 0.25rem 0.75rem;
		border-radius: 12px;
		font-size: 0.8rem;
		font-weight: 600;
		text-transform: uppercase;
	}

	.status-draft {
		background: #f0f0f0;
		color: #666;
	}
	.status-active {
		background: #e3f2fd;
		color: #1976d2;
	}
	.status-completed {
		background: #e8f5e9;
		color: #388e3c;
	}
	.status-archived {
		background: #fafafa;
		color: #9e9e9e;
	}

	.description {
		color: #666;
		margin: 0 0 1rem 0;
		line-height: 1.6;
	}

	.meta {
		display: flex;
		flex-wrap: wrap;
		gap: 1.5rem;
		font-size: 0.9rem;
		color: #888;
	}

	.status-actions {
		background: white;
		padding: 1.5rem;
		border-radius: 8px;
		margin-bottom: 2rem;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
	}

	.status-actions h2 {
		margin: 0 0 1rem 0;
		font-size: 1.25rem;
		color: #333;
	}

	.action-buttons {
		display: flex;
		gap: 1rem;
	}

	.action-buttons button {
		padding: 0.625rem 1.25rem;
		background: #1976d2;
		color: white;
		border: none;
		border-radius: 4px;
		cursor: pointer;
		transition: background 0.2s ease;
	}

	.action-buttons button:hover {
		background: #1565c0;
	}

	.action-buttons .btn-archive {
		background: #ff9800;
	}

	.action-buttons .btn-archive:hover {
		background: #f57c00;
	}

	.articles-section {
		background: white;
		padding: 1.5rem;
		border-radius: 8px;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
	}

	.articles-section h2 {
		margin: 0 0 1rem 0;
		font-size: 1.25rem;
		color: #333;
	}

	.placeholder {
		color: #888;
		font-style: italic;
	}
</style>
```

---

## 6. State Management

### Session Store (Optional Enhancement)

**File:** `src/lib/stores/sessions.svelte.ts`

```typescript
import { apiClient } from '$lib/api/client';
import type {
	SessionResponse,
	SessionListResponse,
	CreateSessionRequest,
	UpdateSessionRequest
} from '$lib/api/generated';

/**
 * Session store for centralized state management.
 *
 * Design Decision: Using Svelte 5 Runes with module-level state
 *
 * This provides a simple, reactive store without external dependencies.
 * For more complex applications, consider using a state management library.
 */

// Store state
let sessions = $state<SessionResponse[]>([]);
let isLoading = $state(false);
let error = $state<string | null>(null);

export function useSessionStore() {
	return {
		// Getters (read-only)
		get sessions() { return sessions; },
		get isLoading() { return isLoading; },
		get error() { return error; },

		// Actions
		async loadSessions(): Promise<void> {
			isLoading = true;
			error = null;

			try {
				const response = await apiClient.get<SessionListResponse>('/api/v1/sessions');
				sessions = response.sessions;
			} catch (e) {
				error = e instanceof Error ? e.message : 'Failed to load sessions';
				throw e;
			} finally {
				isLoading = false;
			}
		},

		async createSession(data: CreateSessionRequest): Promise<SessionResponse> {
			const session = await apiClient.post<SessionResponse>('/api/v1/sessions', data);
			sessions = [...sessions, session];
			return session;
		},

		async updateSession(id: string, data: UpdateSessionRequest): Promise<SessionResponse> {
			const updated = await apiClient.patch<SessionResponse>(`/api/v1/sessions/${id}`, data);
			sessions = sessions.map(s => s.id === id ? updated : s);
			return updated;
		},

		async deleteSession(id: string): Promise<void> {
			await apiClient.delete(`/api/v1/sessions/${id}`);
			sessions = sessions.filter(s => s.id !== id);
		},

		async changeStatus(id: string, status: string): Promise<SessionResponse> {
			const updated = await apiClient.post<SessionResponse>(
				`/api/v1/sessions/${id}/status`,
				{ status }
			);
			sessions = sessions.map(s => s.id === id ? updated : s);
			return updated;
		},

		// Helpers
		getSessionById(id: string): SessionResponse | undefined {
			return sessions.find(s => s.id === id);
		},

		filterByStatus(status: string): SessionResponse[] {
			if (status === 'all') return sessions;
			return sessions.filter(s => s.status === status);
		}
	};
}
```

---

## 7. API Integration

### Generated Types (Expected from OpenAPI)

After running `make gen-api`, the following types should be available in `src/lib/api/generated.ts`:

```typescript
// Expected generated types from backend OpenAPI spec

export type SessionStatus = 'draft' | 'active' | 'completed' | 'archived';

export interface SessionResponse {
	id: string;
	name: string;
	description?: string;
	status: SessionStatus;
	article_count?: number;
	created_at: string;
	updated_at: string;
}

export interface SessionListResponse {
	sessions: SessionResponse[];
	total: number;
}

export interface CreateSessionRequest {
	name: string;
	description?: string;
}

export interface UpdateSessionRequest {
	name?: string;
	description?: string;
}

export interface ChangeStatusRequest {
	status: SessionStatus;
}
```

---

## 8. Wireframes

### 8.1 Session List Page (Landing)

```
+------------------------------------------------------------------+
|  [Logo] Article Mind                         [Sessions] (active)  |
+------------------------------------------------------------------+
|                                                                   |
|  Research Sessions                          [+ New Session]       |
|  -----------------------------------------------------------------|
|  Filter by status: [All v]                                        |
|                                                                   |
|  +-------------------------+  +-------------------------+         |
|  | My Research Project     |  | Literature Review       |         |
|  | [ACTIVE]               |  | [DRAFT]                |         |
|  |                        |  |                        |         |
|  | Working on ML papers   |  | Collecting papers on   |         |
|  |                        |  | quantum computing      |         |
|  |                        |  |                        |         |
|  | 12 articles            |  | 0 articles             |         |
|  | Created Jan 15, 2026   |  | Created Jan 18, 2026   |         |
|  | Updated Jan 19, 2026   |  |                        |         |
|  | ---------------------- |  | ---------------------- |         |
|  | [Edit] [Archive] [Del] |  | [Edit] [Archive] [Del] |         |
|  +-------------------------+  +-------------------------+         |
|                                                                   |
|  +-------------------------+  +-------------------------+         |
|  | Completed Analysis      |  | Archived Project        |         |
|  | [COMPLETED]            |  | [ARCHIVED]             |         |
|  | ...                    |  | ...                    |         |
|  +-------------------------+  +-------------------------+         |
|                                                                   |
+------------------------------------------------------------------+
```

### 8.2 Create Session Modal

```
+------------------------------------------+
|  Create New Session                   X  |
|------------------------------------------|
|                                          |
|  Name *                                  |
|  +------------------------------------+  |
|  | Enter session name                 |  |
|  +------------------------------------+  |
|                                          |
|  Description                             |
|  +------------------------------------+  |
|  | Optional description for this      |  |
|  | session                            |  |
|  |                                    |  |
|  +------------------------------------+  |
|                                          |
|  -------------------------------------- |
|                    [Cancel] [Create Session] |
+------------------------------------------+
```

### 8.3 Session Detail Page

```
+------------------------------------------------------------------+
|  [Logo] Article Mind                         [Sessions] (active)  |
+------------------------------------------------------------------+
|                                                                   |
|  Sessions / My Research Project                                   |
|                                                                   |
|  +--------------------------------------------------------------+|
|  |  My Research Project                     [ACTIVE]            ||
|  |                                                              ||
|  |  Working on machine learning papers for my thesis.           ||
|  |                                                              ||
|  |  Created: January 15, 2026 at 10:30 AM                       ||
|  |  Updated: January 19, 2026 at 2:45 PM                        ||
|  |  12 articles                                                 ||
|  +--------------------------------------------------------------+|
|                                                                   |
|  +--------------------------------------------------------------+|
|  |  Status Actions                                              ||
|  |  [Mark Complete]  [Archive Session]                          ||
|  +--------------------------------------------------------------+|
|                                                                   |
|  +--------------------------------------------------------------+|
|  |  Articles                                                    ||
|  |  Article management will be implemented in a future plan.    ||
|  +--------------------------------------------------------------+|
|                                                                   |
+------------------------------------------------------------------+
```

---

## 9. Implementation Steps

### Phase 1: Foundation (1-2 hours)

1. **Create TopBar component**
   - Basic navigation structure
   - Active state based on current route
   - Responsive styling

2. **Update +layout.svelte**
   - Include TopBar component
   - Adjust main content area

3. **Run `make gen-api`** (after Plan R2 is implemented)
   - Generate TypeScript types for session endpoints

### Phase 2: Session List Page (1.5-2 hours)

4. **Create SessionCard component**
   - Display session information
   - Status badge styling
   - Action buttons (edit, archive, delete)

5. **Update +page.svelte**
   - Session list grid
   - Loading and error states
   - Empty state

6. **Add filtering**
   - Status filter dropdown
   - Filter logic

### Phase 3: Modals (1-1.5 hours)

7. **Create CreateSessionModal**
   - Form with name and description
   - Create and edit modes
   - Form validation

8. **Create DeleteSessionModal**
   - Confirmation dialog
   - Soft delete explanation

### Phase 4: Session Detail (1 hour)

9. **Create session detail page**
   - Route: `/sessions/[id]/+page.svelte`
   - Display full session info
   - Status transition buttons
   - Breadcrumb navigation

### Phase 5: Polish & Testing (1 hour)

10. **Write unit tests**
    - SessionCard rendering
    - Modal interactions
    - Store operations

11. **Test integration**
    - Full CRUD flow
    - Error handling
    - Edge cases

---

## 10. Testing

### 10.1 SessionCard Component Test

**File:** `tests/unit/SessionCard.test.ts`

```typescript
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/svelte';
import SessionCard from '$lib/components/SessionCard.svelte';
import type { SessionResponse } from '$lib/api/generated';

describe('SessionCard', () => {
	const mockSession: SessionResponse = {
		id: '1',
		name: 'Test Session',
		description: 'A test description',
		status: 'active',
		article_count: 5,
		created_at: '2026-01-15T10:00:00Z',
		updated_at: '2026-01-19T14:00:00Z'
	};

	it('renders session name', () => {
		render(SessionCard, { props: { session: mockSession } });
		expect(screen.getByText('Test Session')).toBeTruthy();
	});

	it('renders session description', () => {
		render(SessionCard, { props: { session: mockSession } });
		expect(screen.getByText('A test description')).toBeTruthy();
	});

	it('renders status badge with correct class', () => {
		render(SessionCard, { props: { session: mockSession } });
		const badge = screen.getByText('active');
		expect(badge.classList.contains('badge-active')).toBe(true);
	});

	it('renders article count', () => {
		render(SessionCard, { props: { session: mockSession } });
		expect(screen.getByText('5 articles')).toBeTruthy();
	});

	it('links to session detail page', () => {
		render(SessionCard, { props: { session: mockSession } });
		const link = screen.getByRole('link');
		expect(link.getAttribute('href')).toBe('/sessions/1');
	});

	it('does not show edit button for archived sessions', () => {
		const archivedSession = { ...mockSession, status: 'archived' as const };
		render(SessionCard, {
			props: {
				session: archivedSession,
				onEdit: () => {}
			}
		});
		expect(screen.queryByText('Edit')).toBeNull();
	});
});
```

### 10.2 CreateSessionModal Test

**File:** `tests/unit/CreateSessionModal.test.ts`

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/svelte';
import CreateSessionModal from '$lib/components/CreateSessionModal.svelte';

describe('CreateSessionModal', () => {
	it('renders with create title when no session provided', () => {
		render(CreateSessionModal, {
			props: {
				isOpen: true,
				onClose: () => {},
				onSubmit: async () => {}
			}
		});
		expect(screen.getByText('Create New Session')).toBeTruthy();
	});

	it('renders with edit title when session provided', () => {
		render(CreateSessionModal, {
			props: {
				isOpen: true,
				session: {
					id: '1',
					name: 'Test',
					status: 'draft',
					created_at: '2026-01-15T10:00:00Z',
					updated_at: '2026-01-15T10:00:00Z'
				},
				onClose: () => {},
				onSubmit: async () => {}
			}
		});
		expect(screen.getByText('Edit Session')).toBeTruthy();
	});

	it('shows error when submitting empty name', async () => {
		render(CreateSessionModal, {
			props: {
				isOpen: true,
				onClose: () => {},
				onSubmit: async () => {}
			}
		});

		const submitButton = screen.getByText('Create Session');
		await fireEvent.click(submitButton);

		expect(screen.getByText('Session name is required')).toBeTruthy();
	});

	it('calls onSubmit with form data', async () => {
		const onSubmit = vi.fn().mockResolvedValue(undefined);

		render(CreateSessionModal, {
			props: {
				isOpen: true,
				onClose: () => {},
				onSubmit
			}
		});

		const nameInput = screen.getByLabelText('Name *');
		await fireEvent.input(nameInput, { target: { value: 'New Session' } });

		const submitButton = screen.getByText('Create Session');
		await fireEvent.click(submitButton);

		expect(onSubmit).toHaveBeenCalledWith({
			name: 'New Session',
			description: undefined
		});
	});
});
```

---

## 11. Acceptance Criteria

### Functional Requirements

- [ ] **AC1:** Top navigation bar displays "Sessions" menu item
- [ ] **AC2:** Default landing page (`/`) shows list of research sessions
- [ ] **AC3:** Users can create new sessions via "New Session" button
- [ ] **AC4:** Session cards display: name, description, status, article count, dates
- [ ] **AC5:** Users can rename sessions via edit modal
- [ ] **AC6:** Users can delete sessions with confirmation dialog
- [ ] **AC7:** Users can archive sessions
- [ ] **AC8:** Clicking a session navigates to detail page
- [ ] **AC9:** Status filter works correctly
- [ ] **AC10:** Session detail page shows full information
- [ ] **AC11:** Status transitions work (draft -> active -> completed -> archived)

### Non-Functional Requirements

- [ ] **AC12:** All API calls use generated TypeScript types
- [ ] **AC13:** Loading states display during API calls
- [ ] **AC14:** Error states display with retry option
- [ ] **AC15:** Empty states display appropriate messaging
- [ ] **AC16:** Components follow Svelte 5 Runes patterns
- [ ] **AC17:** All components have accessibility attributes (ARIA)
- [ ] **AC18:** Unit tests pass with 80%+ coverage
- [ ] **AC19:** TypeScript compilation passes with no errors

---

## 12. Common Pitfalls

### Pitfall 1: Not Regenerating Types

**Symptom:** TypeScript errors about missing properties

**Solution:**
```bash
cd article-mind-ui
make gen-api
```

### Pitfall 2: Svelte 4 Syntax in Svelte 5

**Symptom:** Components don't react to state changes

**Wrong:**
```svelte
<script>
  export let session;  // Svelte 4 syntax
</script>
```

**Correct:**
```svelte
<script>
  let { session } = $props();  // Svelte 5 Runes
</script>
```

### Pitfall 3: Missing Error Handling

**Symptom:** API errors cause unhandled promise rejections

**Solution:** Always wrap API calls in try-catch and update error state.

### Pitfall 4: Not Using $derived for Computed Values

**Symptom:** Values don't update when dependencies change

**Wrong:**
```svelte
<script>
  let sessions = $state([]);
  let filteredSessions = sessions.filter(s => s.status === 'active');  // Static!
</script>
```

**Correct:**
```svelte
<script>
  let sessions = $state([]);
  let filteredSessions = $derived(sessions.filter(s => s.status === 'active'));
</script>
```

---

## 13. Next Steps

After completing this plan:

1. **Implement Plan R2** (Session CRUD API) to provide backend endpoints
2. **Run `make gen-api`** to generate TypeScript types
3. **Test full integration** between frontend and backend
4. **Add article management** to session detail page (future plan)
5. **Add search and filtering** enhancements

---

**Plan Status:** Ready for implementation
**Last Updated:** 2026-01-19
